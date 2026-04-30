import ctypes
import errno
import logging
import os
import pickle
import socket
import struct
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import get_world_group

logger = logging.getLogger(__name__)

# ---- PP control command constants ----
CMD_PUT_META = 2
CMD_LAYERWISE = 3
CMD_STORE_COMPLETE = 5


class FlexKVComm:
    """FlexKV hierarchical communication on 3D topology (PP x CP x TP).

    Public API: scatter, scatter_pp, barrier, all_reduce_min.
    Public read-only attributes: is_sync_leader, needs_sync, is_pp_active,
    is_pp_sender, is_pp_receiver.

    Communication hierarchy (3 dimensions, fan-out / aggregate):

        Scatter (async):   PP_leader --isend--> PP_stage_leaders
                                     --isend--> CP_ranks  (per PP stage)
                                     --isend--> TP_ranks  (per CP group)

        AllReduce (sync):  TP group all_reduce
                           -> CP group all_reduce
                           -> PP P2P reduce (stage leaders)
                           -> bcast result back down

        scatter_pp (async): PP0 stage leader --isend--> PP1+ stage leaders
        barrier (sync):     hierarchical barrier: TP -> CP -> PP
    """

    # ---- Tags for P2P scatter on world_cpu_group ----
    _TAG_SCATTER = int.from_bytes(b"FxSc", byteorder="big")
    _TAG_PP      = int.from_bytes(b"FxPP", byteorder="big")
    _TAG_CP      = int.from_bytes(b"FxCP", byteorder="big")
    _TAG_TP      = int.from_bytes(b"FxTP", byteorder="big")
    # ---- Tags for PP P2P all-reduce / barrier on world_cpu_group ----
    _TAG_PP_AR_MIN = int.from_bytes(b"FxA2", byteorder="big")
    _TAG_PP_BARRIER = int.from_bytes(b"FxB2", byteorder="big")
    _TAG_PP_BARRIER_BCAST = int.from_bytes(b"FxB3", byteorder="big")
    _TAG_AR_BCAST = int.from_bytes(b"FxAR", byteorder="big")

    def __init__(
        self,
        flexkv_model_config,
        world_rank: int,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ):
        self.config = flexkv_model_config
        self.world_rank = world_rank
        self._async_works: List = []

        # ---- Extract cpu_group from wrapper objects if present ----
        self.pp_cpu_group = (
            getattr(pp_group, "cpu_group", pp_group) if pp_group is not None else None
        )
        self.attn_tp_cpu_group = (
            getattr(attn_tp_group, "cpu_group", attn_tp_group)
            if attn_tp_group is not None
            else None
        )
        self.attn_cp_cpu_group = (
            getattr(attn_cp_group, "cpu_group", attn_cp_group)
            if attn_cp_group is not None
            else None
        )

        # ---- Dimension sizes ----
        self.pp_size = flexkv_model_config.pp_size
        self.attn_tp_size = flexkv_model_config.attn_tp_size
        self.attn_cp_size = flexkv_model_config.attn_cp_size

        # ---- 3D coordinate ----
        self.pp_rank = flexkv_model_config.pp_rank
        self.attn_tp_rank = flexkv_model_config.attn_tp_rank
        self.attn_cp_rank = flexkv_model_config.attn_cp_rank
        self.is_nsa_cp = flexkv_model_config.is_nsa_cp

        # ---- Role resolution ----
        self.is_pp_stage_leader = (self.attn_tp_rank == 0 and self.attn_cp_rank == 0)
        self.is_sync_leader = (
            self.pp_rank == 0 and self.is_pp_stage_leader
        )
        self.is_pp_leader = (self.pp_rank == 0 and self.is_pp_stage_leader)
        self.is_cp_leader = (self.attn_cp_rank == 0)
        self.is_tp_leader = (self.attn_tp_rank == 0)

        # ---- Rank mapping for point-to-point scatter ----
        # PP stage leaders: one per PP stage (tp=0, cp=0)
        stride = self.attn_tp_size * self.attn_cp_size
        self._pp_stage_leader_ranks = [s * stride for s in range(self.pp_size)]
        # CP leaders: tp_rank=0 of each CP group within this PP stage
        pp_stage_offset = self.pp_rank * stride
        self._cp_leader_ranks = (
            [pp_stage_offset + cp * self.attn_tp_size for cp in range(self.attn_cp_size)]
            if self.attn_cp_size > 1 else []
        )
        # TP group ranks (pre-computed once)
        if self.attn_tp_size > 1:
            if self.attn_tp_cpu_group is None:
                raise RuntimeError(
                    f"[FlexKV] attn_tp_size={self.attn_tp_size} > 1 but "
                    f"attn_tp_cpu_group is None — TP group is required for "
                    f"scatter/collective operations"
                )
            self._tp_group_ranks = [
                dist.get_global_rank(self.attn_tp_cpu_group, i)
                for i in range(self.attn_tp_cpu_group.size())
            ]
        else:
            self._tp_group_ranks = []
        # PP group ranks for scatter_pp (pre-computed once)
        self._pp_group_global_ranks = (
            [dist.get_global_rank(self.pp_cpu_group, i)
             for i in range(self.pp_cpu_group.size())]
            if self.pp_size > 1 and self.pp_cpu_group is not None else []
        )
        # PP stage member ranks (all ranks in same PP stage)
        self._pp_stage_member_ranks = list(
            range(pp_stage_offset, pp_stage_offset + stride)
        )

        # ---- Whether sync is needed (any dimension > 1) ----
        self.needs_sync = (self.pp_size > 1 or self.attn_tp_size > 1 or self.attn_cp_size > 1)

        # ==================================================================
        # Communication group strategy
        # ----------------------------
        # P2P operations (send/recv/isend/irecv) on CPU tensors:
        #   -> use world_group.cpu_group (gloo-backed, full TCP mesh).
        #      Sub-group cpu_groups have unreliable TCP pairs for P2P.
        #
        # Collective operations (all_reduce/barrier):
        #   -> use sglang's sub-group _cpu_groups (fine for collectives).
        # ==================================================================

        self._world_cpu_group = get_world_group().cpu_group

        # ---- PP scatter role flags (used by flexkv_connector) ----
        self.pp_group = self.pp_cpu_group if (self.pp_size > 1 and self.is_pp_stage_leader) else None
        self.is_pp_active = self.pp_size > 1
        self.is_pp_sender = self.is_pp_leader
        self.is_pp_receiver = self.is_pp_stage_leader and not self.is_pp_leader

        logger.info(
            f"[FlexKV] Comm init: rank={world_rank}, "
            f"pp={self.pp_rank}/{self.pp_size}, "
            f"tp={self.attn_tp_rank}/{self.attn_tp_size}, "
            f"cp={self.attn_cp_rank}/{self.attn_cp_size}, "
            f"sync_leader={self.is_sync_leader}, "
            f"stage_leader={self.is_pp_stage_leader}"
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def scatter(self, data: Any, blocking: bool = False) -> Any:
        """Hierarchical scatter: PP -> CP -> TP (async isend).

        Fan-out in 3 stages, each uses isend/irecv on world_cpu_group:
          1. PP: sync_leader -> each PP stage's stage_leader
          2. CP: each stage's cp_leader -> other CP ranks in same stage
          3. TP: each CP group's tp_leader -> other TP ranks
        """
        # Stage 1: PP scatter (sync_leader -> stage leaders)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            data = self._scatter_group(
                data, self._pp_stage_leader_ranks,
                self.is_pp_leader, self._TAG_PP, blocking,
            )

        # Stage 2: CP scatter (cp_leader -> other CP leaders in same PP stage)
        if self._cp_leader_ranks:
            data = self._scatter_group(
                data, self._cp_leader_ranks,
                self.is_cp_leader, self._TAG_CP, blocking,
            )

        # Stage 3: TP scatter (tp_leader -> other TP ranks)
        if self._tp_group_ranks:
            data = self._scatter_group(
                data, self._tp_group_ranks,
                self.is_tp_leader, self._TAG_TP, blocking,
            )

        return data

    def scatter_pp(self, data: Any) -> Any:
        """Fan-out across PP stages (PP0 stage leader -> PP1+ stage leaders).

        Only PP stage leaders participate. Non-leaders are no-ops.
        """
        if not self._pp_group_global_ranks:
            return data
        is_leader = (self._pp_group_global_ranks[0] == self.world_rank)
        return self._scatter_group(
            data, self._pp_group_global_ranks,
            is_leader, self._TAG_SCATTER, blocking=False,
        )

    def all_reduce_min(self, value: int) -> int:
        """Hierarchical all-reduce MIN across TP, CP, and PP dimensions.

        Every rank participates in each collective layer it belongs to:
          Layer 1  TP all_reduce   all attn_tp group members
          Layer 2  CP all_reduce   all attn_cp group members
          Layer 3  PP P2P reduce   only PP stage leaders
          Layer 4  bcast result    stage leaders -> non-stage-leaders
        """
        logger.debug(
            f"[FlexKV] all_reduce_min rank={self.world_rank} value={value}"
        )

        tensor = torch.tensor(value, dtype=torch.int64)

        # Layer 1: TP all_reduce
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)

        # Layer 2: CP all_reduce
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_cp_cpu_group)

        # Layer 3: PP all_reduce via P2P (stage leaders only)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_all_reduce_min_p2p(tensor)

        # Layer 4: broadcast PP result to non-stage-leaders
        if self.pp_size > 1:
            self._bcast_to_stage_members(tensor, self._TAG_AR_BCAST)

        result = tensor.item()
        logger.debug(
            f"[FlexKV] all_reduce_min rank={self.world_rank} "
            f"value={value} -> {result}"
        )
        return result

    def barrier(self):
        """Hierarchical global barrier: TP -> CP -> PP -> bcast."""
        logger.debug(f"[FlexKV] barrier ENTER rank={self.world_rank}")

        # Layer 1: TP barrier
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.barrier(group=self.attn_tp_cpu_group)

        # Layer 2: CP barrier
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.barrier(group=self.attn_cp_cpu_group)

        # Layer 3: PP barrier (stage leaders, P2P)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_barrier_p2p()

        # Layer 4: broadcast PP barrier completion to non-stage-leaders
        if self.pp_size > 1:
            dummy = torch.tensor([0], dtype=torch.int64)
            self._bcast_to_stage_members(dummy, self._TAG_PP_BARRIER_BCAST)

        logger.debug(f"[FlexKV] barrier EXIT  rank={self.world_rank}")

    # ==================================================================
    # Unified scatter helper
    # ==================================================================

    def _scatter_group(
        self,
        data: Any,
        group_ranks: List[int],
        is_leader: bool,
        tag: int,
        blocking: bool = False,
    ) -> Any:
        """Scatter within a group: leader sends to all others, followers recv.

        If current rank is not in group_ranks, returns data unchanged.
        """
        if not group_ranks or self.world_rank not in group_ranks:
            return data

        if is_leader:
            dsts = [r for r in group_ranks if r != self.world_rank]
            works = []
            for dst in dsts:
                works.extend(self._isend(dst, data, tag, self._world_cpu_group))
            if blocking:
                for w in works:
                    w.wait()
            else:
                # Drain completed works when list grows beyond threshold
                # to prevent unbounded growth, but do NOT wait eagerly —
                # that would deadlock when gloo isend.wait() blocks until
                # the peer posts a matching irecv.
                if len(self._async_works) > 4096:
                    self._drain_async_works()
                self._async_works.extend(works)
            return data
        else:
            return self._recv(group_ranks[0], tag, self._world_cpu_group)

    # ==================================================================
    # Async work management
    # ==================================================================

    def _drain_async_works(self):
        """Wait for all pending async works to complete and clear the list.

        First removes already-completed works without blocking, then waits
        for the remaining ones.  This avoids unnecessary blocking on works
        whose peer has already posted a matching recv.
        """
        still_pending = []
        for w in self._async_works:
            if w.is_completed():
                continue
            still_pending.append(w)
        for w in still_pending:
            w.wait()
        self._async_works.clear()

    # ==================================================================
    # Low-level send / recv
    # ==================================================================

    def _isend(self, dst: int, data: Any, tag: int = 0, group=None) -> list:
        """Non-blocking send via gloo-backed cpu_group (CPU tensor P2P)."""
        serialized = bytearray(pickle.dumps(data))
        t_size = torch.tensor([len(serialized)], dtype=torch.long)
        t_data = torch.frombuffer(serialized, dtype=torch.uint8)
        return [
            dist.isend(t_size, dst=dst, tag=tag, group=group),
            dist.isend(t_data, dst=dst, tag=tag, group=group),
        ]

    def _recv(self, src: int, tag: int = 0, group=None) -> Any:
        """Blocking recv via gloo-backed cpu_group (CPU tensor P2P)."""
        t_size = torch.tensor([0], dtype=torch.long)
        dist.irecv(t_size, src=src, tag=tag, group=group).wait()
        size = t_size.item()
        if size == 0:
            return []
        t_data = torch.empty(size, dtype=torch.uint8)
        dist.irecv(t_data, src=src, tag=tag, group=group).wait()
        return pickle.loads(t_data.numpy().tobytes())

    def _send_tensor(self, tensor: torch.Tensor, dst: int, tag: int = 0, group=None):
        dist.send(tensor, dst=dst, tag=tag, group=group)

    def _recv_tensor(self, tensor: torch.Tensor, src: int, tag: int = 0, group=None):
        dist.recv(tensor, src=src, tag=tag, group=group)

    # ==================================================================
    # Intra-stage broadcast (stage leader -> non-leaders in same PP stage)
    # ==================================================================

    def _bcast_to_stage_members(self, tensor: torch.Tensor, tag: int):
        if not self.is_pp_stage_leader:
            self._recv_tensor(
                tensor, src=self._pp_stage_leader_ranks[self.pp_rank],
                tag=tag, group=self._world_cpu_group,
            )
        else:
            for rank in self._pp_stage_member_ranks:
                if rank != self.world_rank:
                    self._send_tensor(
                        tensor, dst=rank, tag=tag, group=self._world_cpu_group,
                    )

    # ==================================================================
    # PP-level P2P collectives (cross-node safe on world_cpu_group)
    # ==================================================================

    def _pp_all_reduce_min_p2p(self, tensor: torch.Tensor):
        """PP all_reduce MIN via star-pattern P2P on world_cpu_group."""
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_AR_MIN

        if self.world_rank == leader_rank:
            result = tensor.item()
            for src in other_leaders:
                other = torch.tensor(0, dtype=torch.int64)
                self._recv_tensor(other, src=src, tag=tag, group=self._world_cpu_group)
                result = min(result, other.item())
            tensor.fill_(result)
            for dst in other_leaders:
                self._send_tensor(tensor, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(tensor, dst=leader_rank, tag=tag, group=self._world_cpu_group)
            self._recv_tensor(tensor, src=leader_rank, tag=tag, group=self._world_cpu_group)

    def _pp_barrier_p2p(self):
        """PP barrier via star-pattern P2P on world_cpu_group."""
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_BARRIER
        dummy = torch.tensor([1], dtype=torch.int64)

        if self.world_rank == leader_rank:
            for src in other_leaders:
                self._recv_tensor(dummy, src=src, tag=tag, group=self._world_cpu_group)
            for dst in other_leaders:
                self._send_tensor(dummy, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(dummy, dst=leader_rank, tag=tag, group=self._world_cpu_group)
            self._recv_tensor(dummy, src=leader_rank, tag=tag, group=self._world_cpu_group)


# ===================================================================
# libc / eventfd helpers (module-private)
# ===================================================================

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
_libc.eventfd.restype = ctypes.c_int
_libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.read.restype = ctypes.c_ssize_t
_libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval=0, flags=0):
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd, val):
    v = ctypes.c_uint64(val)
    n = _libc.write(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd):
    v = ctypes.c_uint64()
    n = _libc.read(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == errno.EAGAIN:
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


# ===================================================================
# Layer-wise transfer components
# ===================================================================


class FlexKVLayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE) for _ in range(num_layers)
        ]
        self._finished = True
        self.wait_remaining: List[int] = [1] * num_layers

    def reset_for_new_transfer(self):
        self._finished = False
        self.wait_remaining = [1] * self._num_layers

    def wait(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        eventfd_read(self.load_event_fds[layer_index])
        if layer_index == self._num_layers - 1:
            self._finished = True

    def close(self):
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self):
        self.close()


class FlexKVLayerDoneCounter:
    """Triple-buffered layer-wise transfer counter using eventfds."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.num_counters = 3
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer: Dict[int, int] = {}

    def register_task(self, task_id: int, producer_id: int):
        self._task_to_producer[task_id] = producer_id

    def register_task_with_explicit_counter_id(self, task_id: int, counter_id: int):
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"Invalid counter_id={counter_id}, must be in [0, {self.num_counters})"
            )
        self._task_to_producer[task_id] = counter_id
        self.events[counter_id].reset_for_new_transfer()

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        return self.producer_index

    def set_consumer(self, task_id: int):
        if task_id < 0:
            self.consumer_index = -1
            return
        producer_id = self._task_to_producer.pop(task_id, None)
        if producer_id is not None:
            self.consumer_index = producer_id
        else:
            self.consumer_index = -1

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        if event.wait_remaining[threshold] <= 0:
            return
        event.wait_remaining[threshold] -= 1
        event.wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer.clear()

    def __del__(self):
        for event in self.events:
            event.close()
        self.events.clear()
