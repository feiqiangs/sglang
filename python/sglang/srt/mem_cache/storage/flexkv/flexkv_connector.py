import ctypes
import logging
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.utils import broadcast_pyobj

try:
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)


# ---- libc / eventfd ----
libc = ctypes.CDLL("libc.so.6", use_errno=True)

libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
libc.eventfd.restype = ctypes.c_int

libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_ssize_t

libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval=0, flags=0):
    fd = libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd, val):
    v = ctypes.c_uint64(val)
    buf = ctypes.byref(v)
    n = libc.write(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd):
    """Blocking read from eventfd."""
    v = ctypes.c_uint64()
    buf = ctypes.byref(v)
    n = libc.read(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == 11:  # EAGAIN
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    """Send multiple fds + extra_data via Unix domain socket."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


def recv_fds(sock: socket.socket, num_fds: int):
    """Receive multiple fds + extra_data via Unix domain socket."""
    data_buf = bytearray(256)
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))

    nbytes, ancdata, flags, addr = sock.recvmsg_into(
        [data_buf], anc_buf_size, 0
    )
    data = bytes(data_buf[:nbytes])

    fds = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            num_received = len(cdata) // struct.calcsize("i")
            fds = list(
                struct.unpack(
                    f"{num_received}i", cdata[: num_received * struct.calcsize("i")]
                )
            )
            break
    if not fds:
        raise RuntimeError("did not receive fds via SCM_RIGHTS")
    return fds, data


# ---- CUDA Runtime (via ctypes) ----
def load_cudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for lib in candidates:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None


cudart = load_cudart()

if cudart:
    cudart.cudaLaunchHostFunc.argtypes = [
        ctypes.c_void_p,
        ctypes.CFUNCTYPE(None, ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int


# ---- Layer-wise transfer components ----


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
    """Triple-buffered layer-wise transfer counter using eventfds.

    Provides the same ``set_consumer`` / ``wait_until`` interface expected by
    the KV cache memory pool so that the attention backend can synchronize
    per-layer with an in-flight host->device transfer.

    Because ``ExtendedRadixCache`` assigns monotonically increasing *task_ids*
    while this counter cycles through a fixed number of producer slots, a
    ``_task_to_producer`` mapping translates between the two id spaces.
    """

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

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        return self.producer_index

    def set_consumer(self, index: int):
        if index < 0:
            self.consumer_index = -1
            return
        producer_id = self._task_to_producer.pop(index, None)
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


# ---- FlexKV Connector ----


class FlexKVConnector(BaseKVConnector):
    """KV cache connector backed by FlexKV's distributed cache system.

    Implements ``BaseKVConnector`` so it can be used with
    ``ExtendedRadixCache`` via ``--kv-connector-cls``.
    """

    def __init__(
        self,
        params: Any = None,
        server_args: Any = None,
        tp_group: Any = None,
        tp_rank: int = 0,
        cp_group: Any = None,
        cp_rank: int = 0,
        kvcache: Any = None,
    ):
        super().__init__(params, server_args, tp_group, tp_rank, cp_group, cp_rank, kvcache)

        model_config = ModelConfig.from_server_args(server_args)

        cp_size = server_args.attn_cp_size if server_args.attn_cp_size > 0 else 1
        pp_size = server_args.pp_size if server_args.pp_size > 0 else 1
        pp_rank = getattr(server_args, "_pp_rank", 0)
        # Normalise None to safe defaults (PP-only mode may pass None)
        dp_size = getattr(server_args, "_dp_size", None)
        dp_size = dp_size if dp_size is not None else getattr(server_args, "dp_size", 1)
        dp_rank = getattr(server_args, "_dp_rank", None)
        dp_rank = dp_rank if dp_rank is not None else 0

        # Compute num_local_layers for PP
        if pp_size > 1:
            from sglang.srt.distributed.utils import get_pp_indices
            total_layers = int(getattr(model_config, "num_hidden_layers", 0))
            start_layer, end_layer = get_pp_indices(total_layers, pp_rank, pp_size)
            num_local_layers = end_layer - start_layer
        else:
            num_local_layers = 0

        self.flexkv_config = FlexKVConfig.from_env()
        self.flexkv_config.post_init_from_sglang_config(
            sglang_config=model_config,
            tp_size=server_args.tp_size,
            page_size=params.page_size,
            num_local_layers=num_local_layers,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            is_nsa_cp=server_args.enable_nsa_prefill_context_parallel,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.tp_cpu_group = getattr(tp_group, "cpu_group", tp_group) if tp_group is not None else None
        self.cp_size = cp_size
        # self.cp_rank already set by base connector class
        self.cp_cpu_group = getattr(cp_group, "cpu_group", cp_group) if cp_group is not None else None
        self.page_size = params.page_size

        # INFO: self.rank answers "who am I in my TP/CP group" (local)?
        #       self.src_rank answers "what is the global rank of the group leader" (used as the broadcast source)?
        #       self.global_rank answers "who am I in the world" (global)?
        self.rank = self.cp_rank if cp_size > 1 else self.tp_rank # WARN: Either CP or TP under a DP group, not both
        # Compute global_rank and src_rank for correct broadcast in PP scenarios
        self.global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if cp_size > 1:
            self.src_rank = torch.distributed.get_global_rank(self.cp_cpu_group, 0)
        elif self.tp_cpu_group is not None and self.tp_size > 1:
            self.src_rank = torch.distributed.get_global_rank(self.tp_cpu_group, 0)
        else:
            self.src_rank = 0        

        # Build unified kv_caches list (MLA vs MHA)
        indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            logger.info(
                f"[FlexKV] Detected sparse attention indexer cache with "
                f"{len(indexer_buffers)} indexer layers, "
                f"shape={indexer_buffers[0].shape}"
            )

        if hasattr(kvcache, "kv_buffer"):
            # MLA: K and V share the same buffer, register once per layer
            kv_caches = kvcache.kv_buffer
        elif hasattr(kvcache, "k_buffer"):
            # MHA: separate K and V buffers, concat as [k_layers..., v_layers...]
            kv_caches = kvcache.k_buffer + kvcache.v_buffer
        else:
            raise AttributeError(
                f"Unsupported KV cache type {type(kvcache).__name__}: "
                f"expected 'kv_buffer' (MLA/NSA) or 'k_buffer'/'v_buffer' (MHA)."
            )

        # Build rank label for structured logging
        rank_parts = []
        if cp_size > 1:
            rank_parts.append(f"cp_rank={int(cp_rank)}")
        elif int(self.tp_size) > 1:
            rank_parts.append(f"tp_rank={int(tp_rank)}")
        if int(pp_size) > 1:
            rank_parts.append(f"pp_rank={int(pp_rank)}")
        if int(dp_size) > 1:
            rank_parts.append(f"dp_rank={int(dp_rank)}")
        self._rank_label = f" [{', '.join(rank_parts)}]" if rank_parts else ""

        if self.rank == 0:
            self.kv_manager = KVManager(
                model_config=self.flexkv_config.model_config,
                cache_config=self.flexkv_config.cache_config,
                dp_client_id=int(dp_rank),
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()
            logger.info(
                f"[FlexKV] Creating KVManager{self._rank_label}: "
                f"server_recv_port={self.flexkv_config.server_recv_port}, "
                f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        # Use globally unique device_id: dp_rank * {cp|tp}_size + {cp|tp}_rank
        # so that GPUs from different DP ranks don't collide in TransferManager.
        # NOTE: This only works for single-node DP. Multi-node DP is not
        # considered here.
        if self.cp_size > 1:
            global_device_id = int(dp_rank) * int(self.cp_size) + int(self.cp_rank) # Either CP or TP
        else:
            global_device_id = int(dp_rank) * int(self.tp_size) + int(self.tp_rank)
        self.tp_client = KVTPClient(self.flexkv_config.gpu_register_port, int(dp_rank), global_device_id) # WARN: Reuse TP client for CP
        logger.info(
            (f"[FlexKV] Use KVTPClient on behalf of CP\n" if self.cp_size > 1 else "") +
            f"[FlexKV] KVTPClient created{self._rank_label}: "
            f"gpu_register_port={self.flexkv_config.gpu_register_port}")
        self._register_to_server(kv_caches, indexer_buffers)
        logger.info(
            f"[FlexKV] KVTPClient registered to server{self._rank_label}: "
            f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        self.num_layers = self.flexkv_config.model_config.num_layers
        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
        base_eventfd_socket = os.getenv(
            "FLEXKV_LAYERWISE_EVENTFD_SOCKET", "/tmp/flexkv_layerwise_eventfd.sock"
        )
        _pp_size = int(self.flexkv_config.model_config.pp_size or 1)
        _pp_rank = int(self.flexkv_config.model_config.pp_rank or 0)
        _dp_size = int(self.flexkv_config.model_config.dp_size or 1)
        _dp_rank = int(dp_rank)
        sock_suffix = ""
        if _pp_size > 1:
            sock_suffix += f"_pp{_pp_rank}"
        if _dp_size > 1:
            sock_suffix += f"_dp{_dp_rank}"
        if sock_suffix:
            root, ext = os.path.splitext(base_eventfd_socket)
            self.layerwise_eventfd_socket = f"{root}{sock_suffix}{ext}"
        else:
            self.layerwise_eventfd_socket = base_eventfd_socket
        logger.info(
            f"[FlexKV] Eventfd socket path configured{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, "
            f"layerwise_transfer={self.enable_layerwise_transfer}")
        self.layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.getenv("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )
        self._layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False

        self._init_layer_transfer_components()

        if self._layer_done_counter is not None and kvcache is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

        # rid -> flexkv_task_id (pending loads awaiting start_load_kv)
        self._pending_loads: Dict[str, int] = {}
        # ext_task_id -> producer_id (layerwise loads in flight)
        self._ongoing_loads: Dict[int, int] = {}
        # ext_task_ids whose load has completed
        self._completed_loads: List[int] = []
        # ext_task_id -> flexkv_task_id (stores in flight, rank 0 only)
        self._ongoing_stores: Dict[int, int] = {}
        # ext_task_ids whose store has completed or was skipped (rank 0 only)
        self._completed_stores: List[int] = []
        # flexkv task ids for periodic drain to prevent pipe deadlock
        self._load_fkv_tids: List[int] = []

        if self.rank == 0:
            wait_count = 0
            while not self.kv_manager.is_ready():
                time.sleep(10)
                wait_count += 1
                # Collect diagnostic info for debugging
                diag_parts = []
                # Check IPC socket file existence
                gpu_port = self.flexkv_config.gpu_register_port
                if gpu_port.startswith("ipc://"):
                    ipc_path = gpu_port[len("ipc://"):]
                    ipc_exists = os.path.exists(ipc_path)
                    diag_parts.append(f"ipc_socket={ipc_path} exists={ipc_exists}")
                # Check TransferManager subprocess status
                task_engine = getattr(self.kv_manager, 'kv_task_engine', None)
                if task_engine is not None:
                    for i, th in enumerate(getattr(task_engine, 'transfer_handles', [])):
                        handle = getattr(th, '_handle', None)
                        if handle is not None:
                            parts = []
                            start_evt = getattr(handle, 'start_event', None)
                            ready_evt = getattr(handle, 'ready_event', None)
                            proc = getattr(handle, 'process', None)
                            if start_evt is not None:
                                parts.append(f"started={start_evt.is_set()}")
                            if ready_evt is not None:
                                parts.append(f"ready={ready_evt.is_set()}")
                            if proc is not None:
                                parts.append(f"alive={proc.is_alive()}")
                            if parts:
                                diag_parts.append(f"transfer_handle[{i}]: {', '.join(parts)}")
                diag_str = "; ".join(diag_parts) if diag_parts else "no diagnostics available"
                logger.info(
                    f"[FlexKV] Waiting for FlexKV to be ready{self._rank_label}... "
                    f"(waited {wait_count * 10}s, {diag_str})"
                )
            logger.info(f"[FlexKV] FlexKV is ready{self._rank_label}")

        logger.info(
            f"[FlexKV] Connector initialized{self._rank_label}: "
            f"layerwise_transfer={self.enable_layerwise_transfer}"
        )

    # ---- BaseKVConnector abstract methods ----

    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
    ) -> int:
        hit_length = 0
        flexkv_task_id = -1

        # INFO: TP/CP group is strictly synchronous, so TP/CP ranks are symmetric. This means they
        #       have identical dst GPU blocks. Hence, let TP/CP rank 0 do prefix matching on the
        #       TP/CP group's behalf and broadcast the result to the rest of the group.
        if self.rank == 0:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            result = self.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )
            # get_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Guard against unpacking a
            # None result to avoid crashing the scheduler.
            if result is None:
                logger.warning("[FlexKV] get_match returned None, treating as no hit")
                flexkv_task_id = -1
                hit_length = 0
            else:
                flexkv_task_id, matched_mask = result
                hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
            if not update_state_for_load and flexkv_task_id >= 0:
                self.kv_manager.cancel([flexkv_task_id])
            else:
                ## GPU hit length is the zero length of token masks
                gpu_hit_length = torch.logical_not(token_mask).sum()
                logger.info(f"[FlexKV Connector] gpu hit length: {gpu_hit_length}, Flexkv hit length: {hit_length}")

        if self.cp_cpu_group is not None and self.cp_size > 1:
            data = broadcast_pyobj(
                [{"hit_length": hit_length, "task_id": flexkv_task_id}],
                self.global_rank,
                self.cp_cpu_group,
                src=self.src_rank
            )[0]
            hit_length = data['hit_length']
            flexkv_task_id = data['task_id']
        elif self.tp_cpu_group is not None and self.tp_size > 1:
            data = broadcast_pyobj(
                [{"hit_length": hit_length, "task_id": flexkv_task_id}],
                self.global_rank,
                self.tp_cpu_group,
                src=self.src_rank,
            )[0]
            hit_length = data["hit_length"]
            flexkv_task_id = data["task_id"]

        # Page-align host_hit_length: ensure GET loads complete pages
        if hit_length > 0 and self.page_size > 1:
            aligned_hit = (hit_length // self.page_size) * self.page_size
            if aligned_hit < hit_length:
                logger.debug(
                    "[FlexKV] get_new_hit_length: host_hit_length page_align %d -> %d (page_size=%d)",
                    hit_length, aligned_hit, self.page_size,
                )
                hit_length = aligned_hit

        if update_state_for_load and rid is not None and hit_length > 0:
            self._pending_loads[rid] = flexkv_task_id
        elif update_state_for_load and flexkv_task_id >= 0 and self.tp_rank == 0:
            # Task was not cancelled earlier, but won't be used — cancel it now
            # to avoid resource leak (e.g. hit_length page-aligned to 0, or rid is None)
            self.kv_manager.cancel([flexkv_task_id])
        return hit_length

    def release_load_state(self, rid: str) -> None:
        fkv_tid = self._pending_loads.pop(rid, -1)
        if fkv_tid >= 0 and self.tp_rank == 0:
            self.kv_manager.cancel([fkv_tid])

    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        flexkv_task_ids: List[int] = []
        slot_mappings: List[torch.Tensor] = []

        for op in load_ops:
            fkv_tid = self._pending_loads.pop(op.rid, -1)
            if fkv_tid < 0:
                continue
            flexkv_task_ids.append(fkv_tid)
            indices = op.device_indices
            slot_mapping_cpu = indices.cpu() if indices.is_cuda else indices
            slot_mapping_cpu = slot_mapping_cpu.to(torch.int64)
            slot_mappings.append(slot_mapping_cpu)

        if not flexkv_task_ids:
            self._completed_loads.append(task_id)
            return

        if self.enable_layerwise_transfer and self._layer_done_counter is not None:
            producer_id = self._layer_done_counter.update_producer()
            self._layer_done_counter.events[producer_id].reset_for_new_transfer()
            self._layer_done_counter.register_task(task_id, producer_id)

            if self.rank == 0:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )

            if self.rank == 0:
                self._load_fkv_tids.extend(flexkv_task_ids)
            self._ongoing_loads[task_id] = producer_id
        else:
            if self.rank == 0:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=False,
                )
                response = self.kv_manager.wait(flexkv_task_ids, timeout=30.0)
                if not all(
                    tid in response and response[tid].status == KVResponseStatus.SUCCESS
                    for tid in flexkv_task_ids
                ):
                    logger.warning(
                        "[FlexKV] Some tasks failed in non-layerwise transfer"
                    )

            if self.cp_cpu_group is not None and self.cp_size > 1:
                torch.distributed.barrier(self.cp_cpu_group)
            elif self.tp_cpu_group is not None and self.tp_size > 1:
                torch.distributed.barrier(self.tp_cpu_group)

            self._completed_loads.append(task_id)

    def check_completed_load_tasks(self) -> List[int]:
        if self.rank == 0 and len(self._load_fkv_tids) >= 100:
            self.kv_manager.try_wait(task_ids=self._load_fkv_tids)
            self._load_fkv_tids.clear()

        if self._layer_done_counter is not None:
            for ext_tid, producer_id in list(self._ongoing_loads.items()):
                if self._layer_done_counter.events[producer_id]._finished:
                    self._completed_loads.append(ext_tid)
                    del self._ongoing_loads[ext_tid]

        result = list(self._completed_loads)
        self._completed_loads.clear()
        return result

    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:
        if self.rank != 0:
            return

        try:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            assert len(token_ids) == len(kv_indices), (
                f"len(token_ids)={len(token_ids)} != len(kv_indices)={len(kv_indices)}, "
                f"task_id={task_id}, page_size={self.page_size}, "
                f"kv_indices_shape={kv_indices.shape if hasattr(kv_indices, 'shape') else 'N/A'}"
            )

            # Page-align token_ids and kv_indices BEFORE put_match so that
            # put_match allocates dst_block_ids consistent with the slot_mapping
            # we will later pass to launch().
            original_len = len(token_ids_np)
            if self.page_size > 1:
                aligned_len = (original_len // self.page_size) * self.page_size
                if aligned_len == 0:
                    self._completed_stores.append(task_id)
                    return
                if aligned_len < original_len:
                    token_ids_np = token_ids_np[:aligned_len]
                    kv_indices = kv_indices[:aligned_len]

            result = self.kv_manager.put_match(
                token_ids=token_ids_np, token_mask=None
            )
            # put_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Treat as a failed store.
            if result is None:
                logger.warning("[FlexKV] put_match returned None, skipping store for task %d", task_id)
                self._completed_stores.append(task_id)
                return
            fkv_task_id, unmatched_mask = result

            logger.info(f"[FlexKV] start_store_kv: token_ids length: {len(token_ids)}, kv_indices length: {len(kv_indices)}, fkv_task_id: {fkv_task_id}, unmatched_mask: {unmatched_mask}")

            if unmatched_mask.sum() > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                slot_mapping = slot_mapping.to(torch.int64)

                self.kv_manager.launch(
                    task_ids=[fkv_task_id], slot_mappings=[slot_mapping]
                )
                self._ongoing_stores[task_id] = fkv_task_id
            else:
                self._completed_stores.append(task_id)
        except Exception as e:
            logger.error("[FlexKV] start_store_kv failed: %s", e, exc_info=True)
            self._completed_stores.append(task_id)

    def check_completed_store_tasks(self) -> List[int]:
        completed_ext_ids = list(self._completed_stores)
        self._completed_stores.clear()

        if self.rank == 0 and self._ongoing_stores:
            fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
            completed_dict = self.kv_manager.try_wait(task_ids=list(fk_to_ext.keys()))
            for fk_tid in completed_dict:
                ext_tid = fk_to_ext[fk_tid]
                completed_ext_ids.append(ext_tid)
                del self._ongoing_stores[ext_tid]

        if self.cp_cpu_group is not None and self.cp_size > 1:
            completed_ext_ids = broadcast_pyobj(
                [completed_ext_ids] if self.rank == 0 else [None],
                self.global_rank,
                self.cp_cpu_group,
                src=self.src_rank,
            )[0]
        elif self.tp_cpu_group is not None and self.tp_size > 1:
            completed_ext_ids = broadcast_pyobj(
                [completed_ext_ids] if self.rank == 0 else [None],
                self.global_rank,
                self.tp_cpu_group,
                src=self.src_rank,
            )[0]

        return completed_ext_ids

    # ---- Optional overrides ----

    def cancel_prefetch(self, rid: str) -> None:
        fkv_tid = self._pending_loads.pop(rid, -1)
        if fkv_tid >= 0 and self.tp_rank == 0:
            self.kv_manager.cancel([fkv_tid])

    @property
    def layer_done_counter(self) -> Any:
        return self._layer_done_counter

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        if self._layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

    def reset(self) -> None:
        if self.tp_rank == 0 and self._pending_loads:
            pending_tids = [tid for tid in self._pending_loads.values() if tid >= 0]
            if pending_tids:
                self.kv_manager.cancel(pending_tids)
        self._pending_loads.clear()
        self._ongoing_loads.clear()
        self._completed_loads.clear()
        self._load_fkv_tids.clear()

        if self.rank == 0:
            for fk_tid in list(self._ongoing_stores.values()):
                if fk_tid >= 0:
                    self._wait_flexkv_task(fk_tid)
        self._ongoing_stores.clear()
        self._completed_stores.clear()

        if self._layer_done_counter is not None:
            self._layer_done_counter.reset()

    def shutdown(self) -> None:
        if self.rank == 0:
            self.kv_manager.shutdown()

    # ---- Private helpers ----

    def _wait_flexkv_task(self, fk_task_id: int, timeout: float = 20.0) -> bool:
        if fk_task_id < 0 or self.rank != 0:
            return True
        try:
            response = self.kv_manager.wait([fk_task_id], timeout=timeout)
            return (
                fk_task_id in response
                and response[fk_task_id].status == KVResponseStatus.SUCCESS
            )
        except Exception as e:
            logger.error("[FlexKV] wait task failed: %s", e, exc_info=True)
            return False

    def _register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Register GPU KV cache buffers to FlexKV server.

        Args:
            kv_caches: Unified KV cache tensor list.
                - MLA: num_layer tensors (K and V share the same buffer).
                - MHA: 2 * num_layer tensors (K buffers followed by V buffers).
            indexer_buffers: Optional sparse attention indexer buffers.
        """
        assert len(kv_caches) > 0
        assert kv_caches[0].ndim == 3, f"Expected 3D tensor, got shape={kv_caches[0].shape}"

        is_mla = self.flexkv_config.model_config.use_mla
        if not is_mla:
            assert len(kv_caches) % 2 == 0, (
                f"MHA mode expects even number of kv_caches (k_buffers + v_buffers), "
                f"got {len(kv_caches)}"
            )
        num_layer = len(kv_caches) if is_mla else len(kv_caches) // 2
        num_blocks, num_kv_heads, head_size = kv_caches[0].shape

        # GPU layout uses page_size as tokens_per_block so that the transfer
        # engine's block_stride covers an entire page of tokens.  The physical
        # GPU tensor shape is [num_blocks, num_kv_heads, head_size] where each
        # slot stores 1 token, but we present it to FlexKV as
        # [num_blocks/page_size, page_size, num_kv_heads, head_size] so that
        # block_id * block_stride correctly addresses the start of a page.
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layer,
            num_block=num_blocks // self.page_size,
            tokens_per_block=self.page_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=is_mla,
        )

        # Build indexer layout if indexer buffers are present
        indexer_layout = None
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            indexer_tensor = indexer_buffers[0]
            assert indexer_tensor.ndim == 2, (
                f"Expected 2D indexer tensor (num_pages, page_stride_size), "
                f"got shape={indexer_tensor.shape}"
            )
            # sglang's NSA indexer buffer is 2D: (num_pages, page_stride_size),
            # where page_stride_size = page_size * (index_head_dim + scale_bytes).
            # All tokens within a page are flattened into a single contiguous
            # vector, so from FlexKV's perspective each page is one indivisible
            # block with tokens_per_block=1.  The resulting block_stride
            # (= 1 * 1 * page_stride_size) correctly addresses each page.
            indexer_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(indexer_buffers),
                num_block=indexer_tensor.shape[0],
                tokens_per_block=1,
                num_head=1,
                head_size=indexer_tensor.shape[1],
                is_mla=True,
            )
            logger.debug(
                "[FlexKV] Indexer layout: num_layer=%d, num_block=%d, "
                "tokens_per_block=%d, head_size=%d",
                len(indexer_buffers), indexer_tensor.shape[0],
                1, indexer_tensor.shape[1],
            )
            # Consistency check: indexer num_block should equal main KV num_block
            # (1:1 mapping since tokens_per_block = page_size)
            indexer_config = self.flexkv_config.cache_config.indexer
            if indexer_config is not None:
                expected_indexer_blocks = num_blocks // self.page_size
                assert indexer_tensor.shape[0] == expected_indexer_blocks, (
                    f"[FlexKV] Indexer num_block mismatch: indexer has {indexer_tensor.shape[0]} pages, "
                    f"but main KV has {num_blocks} slots / page_size {self.page_size} "
                    f"= {expected_indexer_blocks} expected blocks"
                )

        # Register KV caches (and optional indexer buffers) to FlexKV server
        self.tp_client.register_to_server(
            kv_caches=kv_caches,
            kv_layout=gpu_layout,
            indexer_buffers=indexer_buffers,
            indexer_layout=indexer_layout,
        )
        logger.info("[FlexKV] Registered KV caches to server")

    def _init_layer_transfer_components(self):
        if not self.enable_layerwise_transfer:
            self._layer_done_counter = None
            self._worker_connected = False
            logger.info(f"[FlexKV] Layerwise transfer disabled{self._rank_label}")
            return

        self._layer_done_counter = FlexKVLayerDoneCounter(self.num_layers)
        self._send_eventfds_to_worker()
        logger.info(f"[FlexKV] Initialized layerwise transfer{self._rank_label}")

    def _send_eventfds_to_worker(
        self, retry_interval: float = 1.0
    ):
        max_retries = self.layerwise_eventfd_connect_max_retries
        # Allow up to 3 full connect+send attempts before giving up.
        max_send_retries = 3
        logger.info(
            f"[FlexKV] Attempting eventfd connection{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, max_retries={max_retries}")

        last_error = None
        for send_attempt in range(max_send_retries):
            # Phase 1: Connect to the worker socket (retry until ready).
            sock = None
            for attempt in range(max_retries):
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.connect(self.layerwise_eventfd_socket)
                    logger.info(
                        f"[FlexKV] Eventfd connected{self._rank_label}: "
                        f"socket={self.layerwise_eventfd_socket}, "
                        f"attempts={attempt + 1}"
                        f"{f', send_retry={send_attempt}' if send_attempt > 0 else ''}")
                    break
                except (FileNotFoundError, ConnectionRefusedError) as e:
                    sock.close()
                    sock = None
                    if attempt == max_retries - 1:
                        logger.error(
                            f"[FlexKV] Eventfd connection failed{self._rank_label}: "
                            f"socket={self.layerwise_eventfd_socket}, "
                            f"attempts={max_retries}, error={type(e).__name__}")
                        raise RuntimeError(
                            f"[FlexKV] Failed to connect to eventfd socket "
                            f"{self.layerwise_eventfd_socket} after {max_retries} attempts"
                        )
                    if attempt % 10 == 0:
                        socket_exists = os.path.exists(self.layerwise_eventfd_socket)
                        logger.debug(
                            f"[FlexKV] Eventfd connect retry{self._rank_label}: "
                            f"socket={self.layerwise_eventfd_socket}, "
                            f"attempt={attempt + 1}/{max_retries}, "
                            f"error={type(e).__name__}, socket_exists={socket_exists}")
                    time.sleep(retry_interval)

            # Phase 2: Send metadata + eventfds over the connected socket.
            try:
                sock.connect(self.layerwise_eventfd_socket)
                logger.info(
                    f"[FlexKV] Eventfd connected{self._rank_label}: "
                    f"socket={self.layerwise_eventfd_socket}, "
                    f"attempts={attempt + 1}")
                break
            except (FileNotFoundError, ConnectionRefusedError) as e:
                sock.close()
                sock = None
                if attempt == max_retries - 1:
                    logger.error(
                        f"[FlexKV] Eventfd connection failed{self._rank_label}: "
                        f"socket={self.layerwise_eventfd_socket}, "
                        f"attempts={max_retries}, error={type(e).__name__}")
                    raise RuntimeError(
                        f"[FlexKV] Failed to connect to eventfd socket "
                        f"{self.layerwise_eventfd_socket} after {max_retries} attempts"
                    )
                if attempt % 10 == 0:
                    socket_exists = os.path.exists(self.layerwise_eventfd_socket)
                    logger.debug(
                        f"[FlexKV] Eventfd connect retry{self._rank_label}: "
                        f"socket={self.layerwise_eventfd_socket}, "
                        f"attempt={attempt + 1}/{max_retries}, "
                        f"error={type(e).__name__}, socket_exists={socket_exists}")
                time.sleep(retry_interval)

        try:
            num_counters = self._layer_done_counter.num_counters
            metadata = struct.pack(
                "iiiiii", self.tp_rank, self.tp_size, self.cp_rank, self.cp_size, self.num_layers, num_counters
            )
            sock.sendall(metadata)
            logger.debug(
                f"[FlexKV] Eventfd metadata sent{self._rank_label}: "
                f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, cp_rank={self.cp_rank}, cp_size={self.cp_size}, "
                f"num_layers={self.num_layers}, num_counters={num_counters}")

            for counter_id in range(num_counters):
                fds = self._layer_done_counter.events[counter_id].load_event_fds
                send_fds(sock, fds, struct.pack("i", counter_id))
                logger.debug(
                    f"[FlexKV] Eventfd metadata sent{self._rank_label}: "
                    f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
                    f"num_layers={self.num_layers}, num_counters={num_counters}")

                for counter_id in range(num_counters):
                    fds = self._layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                    logger.debug(
                        f"[FlexKV] Eventfd fds sent{self._rank_label}: "
                        f"counter_id={counter_id}, num_fds={len(fds)}")

                # Wait for ACK from server to confirm fds were received
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout:
                    raise RuntimeError("Timed out waiting for ACK from FlexKV worker")
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV worker NACK'd eventfd transfer (ack={ack!r})")

                self._worker_connected = True
                logger.info(
                    f"[FlexKV] Eventfd setup complete{self._rank_label}: "
                    f"socket={self.layerwise_eventfd_socket}, "
                    f"counters={num_counters}, layers={self.num_layers}")
                return  # Success

        except Exception as e:
            last_error = e
            logger.warning(
                f"[FlexKV] Failed to send eventfds{self._rank_label} "
                f"(send_attempt {send_attempt + 1}/{max_send_retries}): "
                f"socket={self.layerwise_eventfd_socket}, error={e}. "
                f"Will reconnect and retry...")
        finally:
            if sock is not None:
                sock.close()
                sock = None

            # Brief pause before reconnecting
            time.sleep(retry_interval)

        # All send retries exhausted
        logger.error(
            f"[FlexKV] Failed to send eventfds{self._rank_label} after "
            f"{max_send_retries} attempts: "
            f"socket={self.layerwise_eventfd_socket}, last_error={last_error}",
            exc_info=True)
        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self.layerwise_eventfd_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
