import logging
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)


def _coerce_scalar_metric(value: Any) -> float | None:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    return None


def flatten_metric_tree(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested scalar metric dictionaries to slash-separated keys."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        key = str(key)
        next_prefix = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_metric_tree(value, prefix=next_prefix))
            continue

        scalar = _coerce_scalar_metric(value)
        if scalar is None:
            logger.debug("Skip non-scalar reward metric %s=%r", next_prefix, value)
            continue
        flat[next_prefix] = scalar
    return flat


def iter_reward_extra_items(result: dict[str, Any]) -> dict[str, Any]:
    """Return reward metrics that verl can aggregate across a training batch."""
    reward_extra: dict[str, Any] = {}
    reward_metrics = result.get("reward_metrics")
    if isinstance(reward_metrics, Mapping):
        reward_extra.update(flatten_metric_tree(reward_metrics, prefix="reward_metrics"))

    for key, value in result.items():
        if key == "reward_metrics":
            continue
        if value is None:
            logger.debug("Skip None reward extra %s", key)
            continue
        if isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        ):
            logger.debug("Skip non-scalar reward extra %s=%r", key, value)
            continue
        reward_extra[key] = value
    return reward_extra
