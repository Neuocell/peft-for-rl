import pytest

from verl.single_controller.ray import base as ray_base
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV


def _worker_group_with_refs(refs):
    worker_group = object.__new__(ray_base.RayWorkerGroup)
    worker_group.execute_all_async = lambda *_args, **_kwargs: list(refs)
    return worker_group


def test_execute_all_sync_resolves_ready_refs_and_preserves_rank_order(monkeypatch):
    refs = ["rank-0", "rank-1", "rank-2"]
    completion_order = iter(["rank-2", "rank-0", "rank-1"])
    results = {ref: f"result-{ref}" for ref in refs}

    def fake_wait(pending, num_returns):
        assert num_returns == 1
        ready = next(completion_order)
        assert ready in pending
        return [ready], [ref for ref in pending if ref != ready]

    monkeypatch.setattr(ray_base.ray, "wait", fake_wait)
    monkeypatch.setattr(ray_base.ray, "get", lambda ref: results[ref])

    worker_group = _worker_group_with_refs(refs)

    assert worker_group.execute_all_sync("train") == [results[ref] for ref in refs]


def test_execute_all_sync_raises_as_soon_as_a_ready_worker_fails(monkeypatch):
    refs = ["rank-0", "rank-1", "rank-2"]
    wait_calls = []
    get_calls = []

    def fake_wait(pending, num_returns):
        wait_calls.append(list(pending))
        return ["rank-0"], ["rank-1", "rank-2"]

    def fake_get(ref):
        get_calls.append(ref)
        raise RuntimeError("rank 0 exited")

    monkeypatch.setattr(ray_base.ray, "wait", fake_wait)
    monkeypatch.setattr(ray_base.ray, "get", fake_get)

    worker_group = _worker_group_with_refs(refs)

    with pytest.raises(RuntimeError, match="rank 0 exited"):
        worker_group.execute_all_sync("train")

    assert wait_calls == [refs]
    assert get_calls == ["rank-0"]


def test_blocking_bound_worker_method_uses_fail_fast_get(monkeypatch):
    refs = ["rank-0", "rank-1"]
    calls = []

    def fake_get(object_refs):
        calls.append(object_refs)
        return ["result-0", "result-1"]

    monkeypatch.setattr(ray_base, "_ray_get_fail_fast", fake_get)
    functor = ray_base.func_generator(
        object(),
        "train",
        dispatch_fn=lambda _group, *args, **kwargs: (args, kwargs),
        collect_fn=lambda _group, output: output,
        execute_fn=lambda _method_name, *_args, **_kwargs: refs,
        blocking=True,
    )

    assert functor() == ["result-0", "result-1"]
    assert calls == [refs]


def test_ppo_runtime_enables_nccl_failure_monitoring():
    env_vars = PPO_RAY_RUNTIME_ENV["env_vars"]

    assert env_vars["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env_vars["TORCH_NCCL_ENABLE_MONITORING"] == "1"
    assert int(env_vars["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"]) > 0
    assert env_vars["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert int(env_vars["TORCH_FR_BUFFER_SIZE"]) > 0
