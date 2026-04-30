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
from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
    CMD_PUT_META,
    CMD_LAYERWISE,
    CMD_STORE_COMPLETE,
    FlexKVLayerLoadingEvent,
    FlexKVLayerDoneCounter,
    FlexKVComm,
    send_fds,
)

try:
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
    from flexkv.transfer.layerwise import build_layerwise_eventfd_socket_path
    from flexkv.transfer_manager import TransferManagerOnRemote
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)

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


# ---- FlexKV Connector ----


class FlexKVConnector(BaseKVConnector):
    """KV cache connector backed by FlexKV's distributed cache system.

    Implements ``BaseKVConnector`` so it can be used with
    ``ExtendedRadixCache`` via ``--kv-connector-cls``.
    """

    def __init__(
        self,
        params: Any,
        server_args: Any,
        tp_rank: int = 0,
        dp_rank: Optional[int] = 0,
        attn_cp_rank: Optional[int] = 0,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ):
        super().__init__(
            params=params,
            server_args=server_args,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )

        # ---- Primitive variables (from params / constructor) ----
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()

        sglang_model_config = ModelConfig.from_server_args(server_args)

        # ---- Initialize FlexKV config ----
        self.flexkv_config = FlexKVConfig.from_env()
        self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sglang_model_config,
            server_args=server_args,
            page_size=self.page_size,
            tp_rank=tp_rank,
            pp_rank=params.pp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
        )

        flexkv_model_config = self.flexkv_config.model_config

        # Structured logging label
        rank_parts = []
        if flexkv_model_config.tp_size > 1:
            rank_parts.append(f"tp_rank={flexkv_model_config.tp_rank}")
        if flexkv_model_config.pp_size > 1:
            rank_parts.append(f"pp_rank={flexkv_model_config.pp_rank}")
        if flexkv_model_config.dp_size > 1:
            rank_parts.append(f"dp_rank={flexkv_model_config.dp_rank}")
        if flexkv_model_config.attn_cp_size > 1:
            rank_parts.append(f"attn_cp_rank={flexkv_model_config.attn_cp_rank}")
        if flexkv_model_config.attn_tp_size > 1:
            rank_parts.append(f"attn_tp_rank={flexkv_model_config.attn_tp_rank}")
        self._rank_label = f" [{', '.join(rank_parts)}]" if rank_parts else ""

        # ---- Communication / sync context ----
        self._sync_ctx = FlexKVComm(
            flexkv_model_config=flexkv_model_config,
            world_rank=(
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            ),
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )
        logger.debug(
            f"[FlexKV] sync_context{self._rank_label}: "
            f"is_sync_leader={self._sync_ctx.is_sync_leader}, "
            f"needs_sync={self._sync_ctx.needs_sync}, "
            f"is_pp_active={self._sync_ctx.is_pp_active}"
        )


        # ---- Align block counts on unified group (single all_reduce MIN) ----
        for _attr in ("num_cpu_blocks", "num_ssd_blocks"):
            _orig = getattr(self.flexkv_config.cache_config, _attr)
            if _orig <= 0:
                continue
            _aligned = self._sync_ctx.all_reduce_min(_orig)
            logger.debug(
                f"[FlexKV] Block count alignment{self._rank_label}: "
                f"attr={_attr}, {_orig} -> {_aligned}"
            )
            if _aligned != _orig:
                logger.info(
                    f"[FlexKV] Block count MIN alignment '{_attr}': "
                    f"{_orig} -> {_aligned}"
                )
            setattr(self.flexkv_config.cache_config, _attr, _aligned)

        if flexkv_model_config.nnodes > 1:
            logger.info(
                f"[FlexKV] Multi-node detected{self._rank_label}: "
                f"tp_size={flexkv_model_config.tp_size}, tp_size_per_node={flexkv_model_config.tp_size_per_node}, "
                f"node_rank={flexkv_model_config.node_rank}, "
                f"attn_tp_size={flexkv_model_config.attn_tp_size}, "
                f"attn_cp_size={flexkv_model_config.attn_cp_size}, "
                f"dp_size={flexkv_model_config.dp_size}, pp_size={flexkv_model_config.pp_size}, "
                f"nnodes={flexkv_model_config.nnodes}"
            )

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

        # ---- Node B: Launch TransferManagerOnRemote ----
        self._remote_process = None
        if flexkv_model_config.nnodes > 1 and flexkv_model_config.node_rank > 0 and flexkv_model_config.local_rank == 0:
            logger.debug(
                f"[FlexKV] Launching TransferManagerOnRemote{self._rank_label}: "
                f"master_host={self.flexkv_config.model_config.master_host}")
            self._remote_process = TransferManagerOnRemote.create_process(
                master_host=self.flexkv_config.model_config.master_host,
            )
            logger.info(
                f"[FlexKV] Launched TransferManagerOnRemote on node_rank={flexkv_model_config.node_rank}"
                f"{self._rank_label}"
            )

        if self._sync_ctx.is_sync_leader:
            self.kv_manager = KVManager(
                model_config=self.flexkv_config.model_config,
                cache_config=self.flexkv_config.cache_config,
                dp_client_id=flexkv_model_config.dp_rank,
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()
            logger.info(
                f"[FlexKV] Creating KVManager{self._rank_label}: "
                f"server_recv_port={self.flexkv_config.server_recv_port}, "
                f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        # ---- GPU Registration Routing ----
        self.tp_client = KVTPClient(
            self.flexkv_config.gpu_register_port,
            dp_rank=flexkv_model_config.dp_rank,
            pp_rank=flexkv_model_config.pp_rank,
            device_id=flexkv_model_config.local_rank,
        )

        # ---- GPU Registration (with retry) ----
        self._register_with_retry(kv_caches, indexer_buffers)
        logger.info(
            f"[FlexKV] KVTPClient registered to server{self._rank_label}: "
            f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )

        model_config = self.flexkv_config.model_config
        self.layerwise_eventfd_socket = build_layerwise_eventfd_socket_path(
            pp_rank=model_config.pp_rank,
            dp_rank=model_config.dp_rank,
            pp_size=model_config.pp_size,
            dp_size=model_config.dp_size,
        )
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
        # ext_task_id -> flexkv_task_id (stores in flight)
        self._ongoing_stores: Dict[int, int] = {}
        # ext_task_ids whose store has completed or was skipped
        self._completed_stores: List[int] = []
        # flexkv task ids for periodic drain to prevent pipe deadlock
        self._load_fkv_tids: List[int] = []
        # rid -> flexkv_task_id (prefetch in flight)
        self._ongoing_prefetches: Dict[str, int] = {}
        cache_cfg = self.flexkv_config.cache_config
        self._prefetch_enabled = bool(
            cache_cfg.enable_ssd
            or cache_cfg.enable_remote
            or cache_cfg.enable_kv_sharing
        )

        if self._sync_ctx.is_sync_leader:
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
        elif flexkv_model_config.nnodes > 1 and flexkv_model_config.node_rank > 0:
            # Node B: no KVManager to wait for, GPU registration retry handles readiness
            logger.info(f"[FlexKV] Node B skipping is_ready wait{self._rank_label}")

        logger.info(
            f"[FlexKV] Connector initialized{self._rank_label}: "
            f"layerwise_transfer={self.enable_layerwise_transfer}, "
            f"prefetch_enabled={self._prefetch_enabled}, "
            f"tp_size={flexkv_model_config.tp_size}, dp_size={flexkv_model_config.dp_size}, "
            f"pp_size={flexkv_model_config.pp_size}, nnodes={flexkv_model_config.nnodes}, "
            f"attn_tp_size={flexkv_model_config.attn_tp_size}, "
            f"attn_cp_size={flexkv_model_config.attn_cp_size}, "
            f"tp_size_per_node={flexkv_model_config.tp_size_per_node}"
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
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            result = self.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
                dp_rank=self.flexkv_config.model_config.dp_rank,
                pp_rank=self.flexkv_config.model_config.pp_rank,
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
                # Only cancel if the task actually has pending work.  When
                # hit_length == 0 the transfer graph is empty and the task was
                # already marked COMPLETED synchronously inside get_match →
                # _process_empty_graph, so cancelling would be a no-op that
                # triggers a spurious "already completed" warning.
                if hit_length > 0:
                    self.kv_manager.cancel([flexkv_task_id])
            else:
                ## GPU hit length is the zero length of token masks
                gpu_hit_length = torch.logical_not(token_mask).sum()
                logger.debug(f"[FlexKV Connector] gpu hit length: {gpu_hit_length}, Flexkv hit length: {hit_length}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"hit_length": hit_length, "task_id": flexkv_task_id},
            )
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
        elif update_state_for_load and flexkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            # Task was not cancelled earlier, but won't be used — cancel it now
            # to avoid resource leak (e.g. hit_length page-aligned to 0, or rid is None).
            # Skip cancel when hit_length == 0: the task's transfer graph was
            # empty and _process_empty_graph already marked it COMPLETED.
            if hit_length > 0:
                self.kv_manager.cancel([flexkv_task_id])
        return hit_length

    def release_load_state(self, rid: str) -> None:
        fkv_tid = self._pending_loads.pop(rid, -1)
        if fkv_tid >= 0 and self._sync_ctx.is_sync_leader:
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

        logger.debug(f"[FlexKV] start_load_kv: resolved {len(flexkv_task_ids)} flexkv tasks")
        if not flexkv_task_ids:
            self._completed_loads.append(task_id)
            return

        flexkv_model_config = self.flexkv_config.model_config
        if self._sync_ctx.is_pp_receiver:
            logger.debug(f"[FlexKV] start_load_kv: sending slot_mapping for pp_receiver")
            for fkv_tid, slot_map in zip(flexkv_task_ids, slot_mappings):
                self.send_slot_mapping_to_remote(fkv_tid, slot_map)

        if self.enable_layerwise_transfer and self._layer_done_counter is not None:
            # PP1+: receive counter_id from PP0 first
            if self._sync_ctx.is_pp_receiver:
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_LAYERWISE:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_LAYERWISE}, got {payload.get('cmd')}")
                producer_id = payload["counter_id"]
                self._layer_done_counter.register_task_with_explicit_counter_id(task_id, producer_id)
            else:
                # Original logic: every rank independently updates producer
                producer_id = self._layer_done_counter.update_producer()
                self._layer_done_counter.events[producer_id].reset_for_new_transfer()
                self._layer_done_counter.register_task(task_id, producer_id)

            # PP0 sync leader: send counter_id to PP1+
            if self._sync_ctx.is_pp_sender:
                self._sync_ctx.scatter_pp(
                    {"cmd": CMD_LAYERWISE, "fkv_task_id": flexkv_task_ids[0], "counter_id": producer_id},
                )

            if self._sync_ctx.is_sync_leader:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )
                self._load_fkv_tids.extend(flexkv_task_ids)
            self._ongoing_loads[task_id] = producer_id
        else:
            if self._sync_ctx.is_sync_leader:
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

            if self._sync_ctx.needs_sync:
                self._sync_ctx.barrier()

            self._completed_loads.append(task_id)

    def check_completed_load_tasks(self) -> List[int]:
        if self._sync_ctx.is_sync_leader and len(self._load_fkv_tids) >= 100:
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

        def _send_pp_put_meta(fkv_task_id: int, unmatched_mask):
            if not self._sync_ctx.is_pp_active:
                return
            mask_list = (
                unmatched_mask.cpu().tolist()
                if hasattr(unmatched_mask, "is_cuda") and unmatched_mask.is_cuda
                else (unmatched_mask.tolist() if hasattr(unmatched_mask, "tolist") else [])
            )
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_PUT_META, "fkv_task_id": fkv_task_id, "unmatched_mask": mask_list},
            )

        if not self._sync_ctx.is_sync_leader:
            if self._sync_ctx.is_pp_receiver:
                logger.debug(
                    f"[FlexKV-Connector] start_store_kv: PP1+ scatter recv PUT_META"
                )
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_PUT_META:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_PUT_META}, got {payload.get('cmd')}")
                fkv_task_id = payload["fkv_task_id"]
                unmatched_mask = torch.tensor(payload["unmatched_mask"])
                if unmatched_mask.sum() > 0 and fkv_task_id >= 0:
                    filtered = kv_indices[unmatched_mask]
                    slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                    slot_mapping = slot_mapping.to(torch.int64)
                    self.send_slot_mapping_to_remote(fkv_task_id, slot_mapping)
                    self._ongoing_stores[task_id] = fkv_task_id
                else:
                    self._completed_stores.append(task_id)
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
                    _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                    self._completed_stores.append(task_id)
                    return
                if aligned_len < original_len:
                    token_ids_np = token_ids_np[:aligned_len]
                    kv_indices = kv_indices[:aligned_len]

            result = self.kv_manager.put_match(
                token_ids=token_ids_np,
                token_mask=None,
                dp_rank=self.flexkv_config.model_config.dp_rank,
                pp_rank=self.flexkv_config.model_config.pp_rank,
            )
            # put_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Treat as a failed store.
            if result is None:
                logger.warning("[FlexKV] put_match returned None, skipping store for task %d", task_id)
                _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                self._completed_stores.append(task_id)
                return
            fkv_task_id, unmatched_mask = result

            logger.debug(
                f"[FlexKV] start_store_kv: tokens={len(token_ids)}, "
                f"kv_indices={len(kv_indices)}, fkv_task_id={fkv_task_id}, "
                f"unmatched={unmatched_mask.sum().item() if hasattr(unmatched_mask, 'sum') else len(unmatched_mask)}"
            )

            _send_pp_put_meta(fkv_task_id, unmatched_mask)

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
            _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
            self._completed_stores.append(task_id)

    def check_completed_store_tasks(self) -> List[int]:
        completed_ext_ids = list(self._completed_stores)
        self._completed_stores.clear()

        completed_dict = {}
        if self._sync_ctx.is_sync_leader and self._ongoing_stores:
            fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
            completed_dict = self.kv_manager.try_wait(task_ids=list(fk_to_ext.keys()))
            for fk_tid in completed_dict:
                ext_tid = fk_to_ext[fk_tid]
                completed_ext_ids.append(ext_tid)
                del self._ongoing_stores[ext_tid]

        if self._sync_ctx.is_pp_sender:
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_STORE_COMPLETE, "completed_fk_ids": list(completed_dict)},
            )
        elif self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_STORE_COMPLETE:
                raise RuntimeError(f"Tag mismatch: expected {CMD_STORE_COMPLETE}, got {payload.get('cmd')}")
            fk_ids = payload["completed_fk_ids"]
            if fk_ids and self._ongoing_stores:
                fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
                for fk_tid in fk_ids:
                    if fk_tid in fk_to_ext:
                        ext_tid = fk_to_ext[fk_tid]
                        completed_ext_ids.append(ext_tid)
                        del self._ongoing_stores[ext_tid]

        if self._sync_ctx.needs_sync:
            completed_ext_ids = self._sync_ctx.scatter(completed_ext_ids)

        return completed_ext_ids

    # ---- Optional overrides ----

    def prefetch(self, rid: str, token_ids: List[int]) -> None:
        if not self._prefetch_enabled:
            return
        if not rid:
            return

        prefetch_task_id = -1
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            prefetch_task_id = self.kv_manager.prefetch_async(
                token_ids=token_ids_np,
                pp_rank=self.flexkv_config.model_config.pp_rank,
            )
            logger.debug(f"[FlexKV] prefetch: launched task_id={prefetch_task_id}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"task_id": prefetch_task_id},
            )
            prefetch_task_id = data["task_id"]

        if prefetch_task_id >= 0:
            self._ongoing_prefetches[rid] = prefetch_task_id

    def check_prefetch_progress(self, rid: str) -> bool:
        if not self._prefetch_enabled:
            return True

        prefetch_task_id = self._ongoing_prefetches.get(rid, -1)
        if prefetch_task_id < 0:
            return True

        is_completed = False
        if self._sync_ctx.is_sync_leader:
            completed = self.kv_manager.try_wait(task_ids=[prefetch_task_id])
            if prefetch_task_id in completed:
                status = completed[prefetch_task_id].status
                if status != KVResponseStatus.SUCCESS:
                    logger.warning(
                        "[FlexKV] prefetch task %d for rid=%s finished with status=%s",
                        prefetch_task_id,
                        rid,
                        status,
                    )
                is_completed = True

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"is_completed": is_completed, "loaded_tokens": 0},
            )
            is_completed = data["is_completed"]

        if is_completed:
            self._ongoing_prefetches.pop(rid, None)
        return is_completed

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        # TODO: Implement this
        return 0

    def cancel_prefetch(self, rid: str) -> None:
        self._pending_loads.pop(rid, None)
        prefetch_task_id = self._ongoing_prefetches.pop(rid, -1)
        if self._sync_ctx.is_sync_leader and prefetch_task_id >= 0:
            # Flexkv not support cancel prefetch task yet
            pass

    @property
    def layer_done_counter(self) -> Any:
        return self._layer_done_counter

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        if self._layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

    def reset(self) -> None:
        if self._sync_ctx.is_sync_leader and self._pending_loads:
            pending_tids = [tid for tid in self._pending_loads.values() if tid >= 0]
            if pending_tids:
                self.kv_manager.cancel(pending_tids)
        self._pending_loads.clear()
        self._ongoing_prefetches.clear()
        self._ongoing_loads.clear()
        self._completed_loads.clear()
        self._load_fkv_tids.clear()

        if self._sync_ctx.is_sync_leader:
            for fk_tid in list(self._ongoing_stores.values()):
                if fk_tid >= 0:
                    self._wait_flexkv_task(fk_tid)
        self._ongoing_stores.clear()
        self._completed_stores.clear()

        if self._layer_done_counter is not None:
            self._layer_done_counter.reset()

    def shutdown(self) -> None:
        if self._sync_ctx.is_sync_leader:
            self.kv_manager.shutdown()

        # Shutdown TransferManagerOnRemote process on Node B
        if self._remote_process is not None:
            try:
                self._remote_process.terminate()
                self._remote_process.join(timeout=5.0)
                if self._remote_process.is_alive():
                    logger.warning(
                        f"[FlexKV] TransferManagerOnRemote did not terminate gracefully, "
                        f"killing{self._rank_label}")
                    self._remote_process.kill()
                    self._remote_process.join()
            except Exception as e:
                logger.warning(
                    f"[FlexKV] Error shutting down TransferManagerOnRemote{self._rank_label}: {e}")
            self._remote_process = None

    # ---- Private helpers ----

    def _wait_flexkv_task(self, fk_task_id: int, timeout: float = 20.0) -> bool:
        if fk_task_id < 0 or not self._sync_ctx.is_sync_leader:
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

    def _register_with_retry(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
        max_retries: int = 360,
    ) -> None:
        """Register GPU with retry for Node B (wait for TransferManagerOnRemote).

        Node B's non-leader ranks may attempt to register before
        TransferManagerOnRemote has finished initializing. This method
        retries the registration up to ``max_retries`` times (default 360,
        i.e. 6 minutes at 1 s intervals).
        """
        for attempt in range(max_retries):
            try:
                self._register_to_server(kv_caches, indexer_buffers)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if attempt % 30 == 0:
                    logger.info(
                        f"[FlexKV] GPU register retry{self._rank_label}: "
                        f"attempt={attempt+1}/{max_retries}, error={e}"
                    )
                time.sleep(1.0)

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
        # In PP mode, each stage only owns a subset of layers.
        # Use the framework's get_pp_indices-derived per-stage count
        # rather than naively dividing, because layer distribution may
        # be non-uniform or controlled by SGLANG_PP_LAYER_PARTITION.
        per_stage_layers = self.flexkv_config.model_config.num_layers_per_pp_stage
        num_blocks, num_kv_heads, head_size = kv_caches[0].shape

        # GPU layout uses page_size as tokens_per_block so that the transfer
        # engine's block_stride covers an entire page of tokens.  The physical
        # GPU tensor shape is [num_blocks, num_kv_heads, head_size] where each
        # slot stores 1 token, but we present it to FlexKV as
        # [num_blocks/page_size, page_size, num_kv_heads, head_size] so that
        # block_id * block_stride correctly addresses the start of a page.
        # NOTE: num_layer must be per-stage so that the transfer engine's
        # gpu_block_type detection (len(gpu_blocks[0]) vs num_layer) matches.
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=per_stage_layers,
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
            logger.debug(f"[FlexKV] Layerwise transfer disabled{self._rank_label}")
            return

        self._layer_done_counter = FlexKVLayerDoneCounter(self.flexkv_config.model_config.num_layers_per_pp_stage)
        self._send_eventfds_to_worker()
        logger.info(f"[FlexKV] Initialized layerwise transfer{self._rank_label}")

    def send_slot_mapping_to_remote(self, task_id: int, slot_mapping: torch.Tensor) -> None:
        """Send slot_mapping to TransferManagerOnRemote via existing ZMQ channel (PP1 side only).

        In cross-node PP mode, PP1's FlexKVConnector sends slot_mapping
        via KVTPClient.send_to_server -> TransferManagerOnRemote.command_socket,
        so that the remote side can call set_gpu_blocks() with its own local GPU block_ids.
        """
        slot_mapping_np = slot_mapping.cpu().to(torch.int64).numpy() if slot_mapping.is_cuda else slot_mapping.numpy()
        self.tp_client.set_slot_mapping(task_id, slot_mapping_np)
        logger.debug(
            f"[FlexKV] send_slot_mapping_to_remote: "
            f"sent task_id={task_id} to TransferManagerOnRemote"
        )

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
            sock = None
            try:
                # Phase 1: Connect to the worker socket (retry until ready).
                for attempt in range(max_retries):
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        sock.connect(self.layerwise_eventfd_socket)
                        logger.info(
                            f"[FlexKV] Eventfd connected{self._rank_label}: "
                            f"socket={self.layerwise_eventfd_socket}, "
                            f"attempts={attempt + 1}"
                            f"{f', send_retry={send_attempt}' if send_attempt > 0 else ''}"
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError) as e:
                        sock.close()
                        sock = None
                        if attempt == max_retries - 1:
                            logger.error(
                                f"[FlexKV] Eventfd connection failed{self._rank_label}: "
                                f"socket={self.layerwise_eventfd_socket}, "
                                f"attempts={max_retries}, error={type(e).__name__}"
                            )
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
                                f"error={type(e).__name__}, socket_exists={socket_exists}"
                            )
                        time.sleep(retry_interval)

                if sock is None:
                    raise RuntimeError(
                        f"[FlexKV] Eventfd socket unavailable after {max_retries} attempts: "
                        f"{self.layerwise_eventfd_socket}"
                    )

                # Phase 2: Send metadata + eventfds over the connected socket.
                # UDS is node-local, so use _per_node TP rank/size so that
                # LayerwiseWorker builds the correct eventfd tensor shape.
                num_counters = self._layer_done_counter.num_counters

                # Send 16-byte metadata: tp_rank_per_node, tp_size_per_node, num_layers, num_counters
                metadata = struct.pack(
                    "iiii",
                    self.flexkv_config.model_config.tp_rank_per_node,
                    self.flexkv_config.model_config.tp_size_per_node,
                    self.flexkv_config.model_config.num_layers_per_pp_stage,
                    num_counters,
                )
                sock.sendall(metadata)
                logger.debug(
                    f"[FlexKV] Eventfd metadata sent{self._rank_label}: "
                    f"tp_rank_per_node={self.flexkv_config.model_config.tp_rank_per_node}, "
                    f"tp_size_per_node={self.flexkv_config.model_config.tp_size_per_node}, "
                    f"num_layers={self.flexkv_config.model_config.num_layers_per_pp_stage}, num_counters={num_counters}"
                )

                for counter_id in range(num_counters):
                    fds = self._layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                    logger.debug(
                        f"[FlexKV] Eventfd fds sent{self._rank_label}: "
                        f"counter_id={counter_id}, num_fds={len(fds)}"
                    )

                # Wait for ACK from server to confirm fds were received.
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout:
                    raise RuntimeError("Timed out waiting for ACK from FlexKV worker")
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV worker NACK'd eventfd transfer (ack={ack!r})"
                    )

                self._worker_connected = True
                logger.info(
                    f"[FlexKV] Eventfd setup complete{self._rank_label}: "
                    f"socket={self.layerwise_eventfd_socket}, "
                    f"counters={num_counters}, layers={self.flexkv_config.model_config.num_layers_per_pp_stage}"
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[FlexKV] Failed to send eventfds{self._rank_label} "
                    f"(send_attempt {send_attempt + 1}/{max_send_retries}): "
                    f"socket={self.layerwise_eventfd_socket}, error={e}. "
                    f"Will reconnect and retry..."
                )
            finally:
                if sock is not None:
                    sock.close()
                # Brief pause before reconnecting.
                time.sleep(retry_interval)

        # All send retries exhausted.
        logger.error(
            f"[FlexKV] Failed to send eventfds{self._rank_label} after "
            f"{max_send_retries} attempts: "
            f"socket={self.layerwise_eventfd_socket}, last_error={last_error}",
            exc_info=True,
        )
        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self.layerwise_eventfd_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
