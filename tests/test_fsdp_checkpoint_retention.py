from verl.utils.checkpoint.fsdp_checkpoint_manager import _discover_previous_actor_checkpoints


def test_discovers_numeric_actor_checkpoints_across_resumed_processes(tmp_path):
    for step in (100, 50, 10):
        (tmp_path / f"global_step_{step}" / "actor").mkdir(parents=True)
    (tmp_path / "global_step_bad" / "actor").mkdir(parents=True)
    (tmp_path / "global_step_75").mkdir()

    current = tmp_path / "global_step_100" / "actor"
    discovered = _discover_previous_actor_checkpoints(str(current))

    assert discovered == [
        str(tmp_path / "global_step_10" / "actor"),
        str(tmp_path / "global_step_50" / "actor"),
    ]
