import math

from verl.utils.reward_score.judge_ensemble import flatten_metric_tree, iter_reward_extra_items


def test_flatten_metric_tree_keeps_only_scalar_metrics():
    metrics = flatten_metric_tree(
        {
            "main": {"score": 1, "passed": True},
            "aux": {"score": 0.5, "payload": [1, 2]},
        }
    )

    assert metrics == {
        "main/score": 1.0,
        "main/passed": 1.0,
        "aux/score": 0.5,
    }


def test_iter_reward_extra_items_matches_dapo_metric_contract():
    result = {
        "score": 1.0,
        "acc": True,
        "parse_success": False,
        "explanation": "ok",
        "missing": None,
        "judge_results": {"raw": "not aggregatable"},
        "reward_metrics": {"judges": {"main": {"score": 1.0, "missing": None}}},
    }

    extra = iter_reward_extra_items(result)

    assert extra["score"] == 1.0
    assert extra["acc"] is True
    assert extra["parse_success"] is False
    assert extra["explanation"] == "ok"
    assert math.isnan(extra["reward_metrics/judges/main/missing"])
    assert extra["reward_metrics/judges/main/score"] == 1.0
    assert "missing" not in extra
    assert "judge_results" not in extra
