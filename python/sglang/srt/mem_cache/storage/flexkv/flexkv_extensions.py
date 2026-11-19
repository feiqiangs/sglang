"""SGLang-side hooks for FlexKV Indexer pruning and early P->D drain.

* :func:`active_indexer_layer_ids` -- the active (non ``skip_topk``) Indexer
  layers of the DSA pool, passed to the connector as ``indexer_layer_ids``.
  Whether FlexKV actually prunes is decided inside the package by
  ``FLEXKV_DEDUP_INDEXER_GROUP`` (default off); the package rejects IDs
  that disagree with ``skip_topk_layers``.
* :class:`EarlyDisaggDrain` -- reclaims GPU pages held by finished P->D
  transfers before the prefill admission pass
  (``FLEXKV_EARLY_DISAGG_DRAIN``, default on; PD prefill with PP=1 only).

CP layer split needs nothing here: the FlexKV connector derives the holder of
every layer from the physical ``LayerSplitDSATokenToKVPool``
(``layer_shard_enabled`` / ``_get_layer_owner_rank``).
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

EARLY_DRAIN_ENV = "FLEXKV_EARLY_DISAGG_DRAIN"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


# ----------------------------------------------------------------------
# Indexer pruning
# ----------------------------------------------------------------------


def active_indexer_layer_ids(kvcache: Any) -> Optional[List[int]]:
    """Pool-local IDs of Indexer layers that own Index-K (``skip_topk`` False).

    ``None`` means "no Indexer" or "no usable metadata"; the connector then
    derives the set itself. A malformed table is not guessed around: FlexKV's
    preflight rejects it collectively.
    """
    target = getattr(kvcache, "full_kv_pool", kvcache)
    if not getattr(target, "index_k_with_scale_buffer", None):
        return None
    skip = getattr(target, "skip_topk_layers", None)
    if not isinstance(skip, (list, tuple)) or not all(type(s) is bool for s in skip):
        return None
    return [i for i, shared in enumerate(skip) if not shared]


# ----------------------------------------------------------------------
# Early P->D drain
# ----------------------------------------------------------------------


def _find_scheduler() -> Any:
    """Return the ``Scheduler`` on the current call stack, if any."""
    from sglang.srt.managers.scheduler import Scheduler

    frame = sys._getframe(1)
    try:
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if isinstance(candidate, Scheduler):
                return candidate
            frame = frame.f_back
    finally:
        del frame
    return None


def _pp_size() -> Optional[int]:
    # ``Scheduler`` has no ``pp_size`` attribute; the runtime context is the
    # source of truth. ``None`` means it could not be determined.
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().pp_size)
    except Exception:  # pragma: no cover - defensive
        logger.warning("[FlexKV] early P->D drain: cannot read pp_size", exc_info=True)
        return None


class EarlyDisaggDrain:
    """Retire finished P->D transfers before the prefill admission pass.

    Two drain points, both reached by every attention CP/TP rank in the same
    order (``process_disagg_prefill_transfer_queue`` all-reduces over those
    groups, so asymmetric calls would hang):

    1. ``check_hicache_events`` -- invoked by the scheduler on all ranks at
       the start of every batch selection.
    2. after ``policy.calc_priority`` -- the LPM sort ends right before the
       admission loop allocates GPU pages. The policy object is wrapped per
       instance; no SGLang source is modified.

    Only active on PD prefill workers with PP=1 (PP has its own consensus
    path for transfer completion).
    """

    def __init__(self, cache: Any) -> None:
        self._cache = cache
        self.enabled = _env_flag(EARLY_DRAIN_ENV, True)
        self._scheduler = None
        self._resolved = False
        self._draining = False

    def on_scheduler_tick(self) -> None:
        if not self.enabled:
            return
        if not self._resolved:
            self._resolved = True
            self._attach(_find_scheduler())
        if self._scheduler is not None:
            self.drain()

    def _disable(self, reason: str) -> None:
        self.enabled = False
        logger.info("[FlexKV] early P->D drain disabled: %s", reason)

    def _attach(self, scheduler: Any) -> None:
        if scheduler is None:
            self._disable("no Scheduler on the call stack")
            return
        from sglang.srt.disaggregation.mode import DisaggregationMode

        mode = getattr(scheduler, "disaggregation_mode", None)
        if mode != DisaggregationMode.PREFILL:
            self._disable(f"disaggregation_mode={mode}")
            return
        pp_size = _pp_size()
        if pp_size != 1:
            self._disable(f"pp_size={pp_size}")
            return
        self._scheduler = scheduler
        policy = getattr(scheduler, "policy", None)
        calc_priority = getattr(policy, "calc_priority", None)
        if callable(calc_priority) and not getattr(
            policy, "_flexkv_early_drain_hooked", False
        ):
            drain = self.drain

            @functools.wraps(calc_priority)
            def calc_priority_then_drain(*args, **kwargs):
                result = calc_priority(*args, **kwargs)
                # The sorted order is applied to the waiting queue right after
                # this returns; a parked request could be requeued into it.
                drain(allow_requeue=False)
                return result

            policy.calc_priority = calc_priority_then_drain
            policy._flexkv_early_drain_hooked = True
        logger.info("[FlexKV] early P->D drain enabled (%s=1)", EARLY_DRAIN_ENV)

    def drain(self, *, allow_requeue: bool = True) -> None:
        scheduler = self._scheduler
        if scheduler is None or self._draining:
            return
        queue = getattr(scheduler, "disagg_prefill_transfer_queue", None)
        if not queue:
            return
        # A parked (pending_bootstrap) request may be yielded back into the
        # waiting queue. Inside admission that would change the queue the
        # caller already ordered, so skip the drain there. The predicate is
        # derived from the shared queue, so all ranks take the same branch.
        if not allow_requeue and any(
            getattr(req, "pending_bootstrap", False) for req in queue
        ):
            return
        self._draining = True
        try:
            scheduler.process_disagg_prefill_transfer_queue()
        finally:
            self._draining = False
