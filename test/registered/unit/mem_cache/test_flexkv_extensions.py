import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.mem_cache.storage.flexkv import flexkv_extensions as ext
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_flexkv_radix_cache_class():
    """Load the cache without requiring the optional FlexKV package."""
    module_name = "_flexkv_radix_cache_extensions_under_test"
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(sys.modules, {connector_name: connector_stub}):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.FlexKVRadixCache


# ----------------------------------------------------------------------
# _env_flag
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("", True), ("1", True), ("on", True), ("0", False), ("off", False)],
)
def test_env_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("FLEXKV_TEST_FLAG", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_TEST_FLAG", value)
    assert ext._env_flag("FLEXKV_TEST_FLAG", True) is expected


def test_env_flag_rejects_garbage(monkeypatch):
    monkeypatch.setenv("FLEXKV_TEST_FLAG", "maybe")
    with pytest.raises(ValueError):
        ext._env_flag("FLEXKV_TEST_FLAG", True)


# ----------------------------------------------------------------------
# active_indexer_layer_ids
# ----------------------------------------------------------------------


def test_active_indexer_layer_ids_without_indexer():
    assert ext.active_indexer_layer_ids(SimpleNamespace()) is None
    assert (
        ext.active_indexer_layer_ids(SimpleNamespace(index_k_with_scale_buffer=[]))
        is None
    )


def test_active_indexer_layer_ids_returns_non_shared_layers():
    pool = SimpleNamespace(
        index_k_with_scale_buffer=[object()] * 4,
        skip_topk_layers=[False, True, False, True],
    )
    assert ext.active_indexer_layer_ids(pool) == [0, 2]


def test_active_indexer_layer_ids_follows_full_kv_pool():
    inner = SimpleNamespace(
        index_k_with_scale_buffer=[object()] * 3,
        skip_topk_layers=[False, False, True],
    )
    assert ext.active_indexer_layer_ids(SimpleNamespace(full_kv_pool=inner)) == [0, 1]


@pytest.mark.parametrize("skip", [None, [0, 1], [False, 1], "FT"])
def test_active_indexer_layer_ids_rejects_malformed_table(skip):
    pool = SimpleNamespace(index_k_with_scale_buffer=[object()], skip_topk_layers=skip)
    assert ext.active_indexer_layer_ids(pool) is None


# ----------------------------------------------------------------------
# EarlyDisaggDrain
# ----------------------------------------------------------------------


class _Policy:
    def __init__(self):
        self.calls = []

    def calc_priority(self, queue, bootstrap_queue=None):
        self.calls.append((queue, bootstrap_queue))
        return "sorted"


def _scheduler(mode=DisaggregationMode.PREFILL, queue=None):
    return SimpleNamespace(
        disaggregation_mode=mode,
        policy=_Policy(),
        disagg_prefill_transfer_queue=[] if queue is None else queue,
        process_disagg_prefill_transfer_queue=MagicMock(return_value=[]),
    )


def _drain(monkeypatch, scheduler, *, pp_size=1, env=None):
    if env is None:
        monkeypatch.delenv(ext.EARLY_DRAIN_ENV, raising=False)
    else:
        monkeypatch.setenv(ext.EARLY_DRAIN_ENV, env)
    monkeypatch.setattr(ext, "_find_scheduler", lambda: scheduler)
    monkeypatch.setattr(ext, "_pp_size", lambda: pp_size)
    return ext.EarlyDisaggDrain(cache=object())


def test_drain_disabled_by_env(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace()])
    find = MagicMock(return_value=sched)
    drain = _drain(monkeypatch, sched, env="0")
    monkeypatch.setattr(ext, "_find_scheduler", find)
    drain.on_scheduler_tick()
    find.assert_not_called()
    sched.process_disagg_prefill_transfer_queue.assert_not_called()
    assert not getattr(sched.policy, "_flexkv_early_drain_hooked", False)


def test_drain_disabled_without_scheduler(monkeypatch):
    drain = _drain(monkeypatch, None)
    drain.on_scheduler_tick()
    assert drain.enabled is False


@pytest.mark.parametrize("mode", [DisaggregationMode.NULL, DisaggregationMode.DECODE])
def test_drain_disabled_outside_prefill(monkeypatch, mode):
    sched = _scheduler(mode=mode, queue=[SimpleNamespace()])
    drain = _drain(monkeypatch, sched)
    drain.on_scheduler_tick()
    assert drain.enabled is False
    sched.process_disagg_prefill_transfer_queue.assert_not_called()


@pytest.mark.parametrize("pp_size", [2, None])
def test_drain_disabled_unless_pp_is_one(monkeypatch, pp_size):
    sched = _scheduler(queue=[SimpleNamespace()])
    # A stray scheduler attribute must not override the runtime context.
    sched.pp_size = 1
    drain = _drain(monkeypatch, sched, pp_size=pp_size)
    drain.on_scheduler_tick()
    assert drain.enabled is False
    sched.process_disagg_prefill_transfer_queue.assert_not_called()


def test_pp_size_reads_runtime_context():
    with patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(pp_size=4),
    ):
        assert ext._pp_size() == 4


def test_drain_hooks_calc_priority_once(monkeypatch):
    sched = _scheduler()
    original = sched.policy.calc_priority
    _drain(monkeypatch, sched).on_scheduler_tick()
    wrapped = sched.policy.calc_priority
    assert wrapped is not original
    assert sched.policy._flexkv_early_drain_hooked is True
    _drain(monkeypatch, sched).on_scheduler_tick()
    assert sched.policy.calc_priority is wrapped


def test_drain_skips_empty_queue(monkeypatch):
    sched = _scheduler()
    drain = _drain(monkeypatch, sched)
    drain.on_scheduler_tick()
    assert drain.enabled is True
    sched.process_disagg_prefill_transfer_queue.assert_not_called()


def test_tick_drains_transfer_queue(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace(pending_bootstrap=True)])
    drain = _drain(monkeypatch, sched)
    drain.on_scheduler_tick()
    drain.on_scheduler_tick()
    assert sched.process_disagg_prefill_transfer_queue.call_count == 2
    sched.process_disagg_prefill_transfer_queue.assert_called_with()


def test_calc_priority_wrapper_drains_after_sort(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace(pending_bootstrap=False)])
    _drain(monkeypatch, sched).on_scheduler_tick()
    sched.process_disagg_prefill_transfer_queue.reset_mock()
    sorts_seen_by_drain = []
    sched.process_disagg_prefill_transfer_queue.side_effect = (
        lambda: sorts_seen_by_drain.append(len(sched.policy.calls))
    )
    assert sched.policy.calc_priority("q", bootstrap_queue="b") == "sorted"
    assert sched.policy.calls == [("q", "b")]
    assert sorts_seen_by_drain == [1]
    sched.process_disagg_prefill_transfer_queue.assert_called_once_with()


def test_calc_priority_wrapper_skips_parked_requests(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace(pending_bootstrap=True)])
    _drain(monkeypatch, sched).on_scheduler_tick()
    sched.process_disagg_prefill_transfer_queue.reset_mock()
    assert sched.policy.calc_priority("q") == "sorted"
    sched.process_disagg_prefill_transfer_queue.assert_not_called()


def test_drain_is_not_reentrant(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace()])
    drain = _drain(monkeypatch, sched)
    sched.process_disagg_prefill_transfer_queue.side_effect = lambda: drain.drain()
    drain.on_scheduler_tick()
    sched.process_disagg_prefill_transfer_queue.assert_called_once_with()
    assert drain._draining is False


def test_drain_resets_flag_on_error(monkeypatch):
    sched = _scheduler(queue=[SimpleNamespace()])
    drain = _drain(monkeypatch, sched)
    sched.process_disagg_prefill_transfer_queue.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        drain.on_scheduler_tick()
    assert drain._draining is False


# ----------------------------------------------------------------------
# FlexKVRadixCache.check_hicache_events wiring
# ----------------------------------------------------------------------


def _bare_cache():
    cls = _load_flexkv_radix_cache_class()
    cache = cls.__new__(cls)
    cache.flexkv_connector = MagicMock()
    cache._drain_completed_stores = MagicMock()
    cache._launch_pending_stores = MagicMock()
    return cache


def test_check_hicache_events_without_early_drain():
    cache = _bare_cache()
    cache.check_hicache_events()
    cache._drain_completed_stores.assert_called_once_with()
    cache.flexkv_connector.drain_launched_loads.assert_called_once_with()
    cache._launch_pending_stores.assert_called_once_with()


def test_check_hicache_events_drains_p2d_first():
    cache = _bare_cache()
    manager = MagicMock()
    cache._early_drain = manager.early
    cache._drain_completed_stores = manager.stores
    cache.flexkv_connector = manager.connector
    cache._launch_pending_stores = manager.launch
    cache.check_hicache_events()
    assert [c[0] for c in manager.mock_calls] == [
        "early.on_scheduler_tick",
        "stores",
        "connector.drain_launched_loads",
        "launch",
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
