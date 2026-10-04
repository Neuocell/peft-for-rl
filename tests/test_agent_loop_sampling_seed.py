from __future__ import annotations

from verl.experimental.agent_loop.agent_loop import trajectory_sampling_seed
from verl.workers.config.rollout import RolloutConfig


def test_rollout_config_accepts_optional_sampling_seed() -> None:
    assert RolloutConfig(name="vllm").seed is None
    assert RolloutConfig(name="vllm", seed=42).seed == 42


def test_trajectory_sampling_seed_is_stable_and_keyed() -> None:
    trajectory = {
        "step": 7,
        "sample_index": "dataset-row-19",
        "rollout_n": 3,
        "validate": False,
    }
    first = trajectory_sampling_seed(42, trajectory)
    assert first == trajectory_sampling_seed(42, dict(trajectory))
    assert 0 <= first < 2**31 - 1

    variants = []
    for key, value in (
        ("step", 8),
        ("sample_index", "dataset-row-20"),
        ("rollout_n", 4),
        ("validate", True),
    ):
        changed = dict(trajectory)
        changed[key] = value
        variants.append(trajectory_sampling_seed(42, changed))
    variants.append(trajectory_sampling_seed(43, trajectory))

    assert len({first, *variants}) == 1 + len(variants)
