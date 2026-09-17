"""Name-safe robot-neutral loading and Wuji replay integration helpers."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Mapping, Sequence

import numpy as np
import yaml


def load_neutral_pose(path: Path, hand_side: str) -> dict[str, float]:
    """Load one finite joint-name keyed neutral pose."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    configured_hand = str(document.get("hand", "")).lower()
    if configured_hand != hand_side.lower():
        raise ValueError(
            f"neutral pose is for {configured_hand!r}, requested {hand_side!r}: {path}"
        )
    joints = document.get("joints")
    if not isinstance(joints, Mapping) or not joints:
        raise ValueError(f"neutral pose must contain a non-empty 'joints' mapping: {path}")
    result = {str(name): float(value) for name, value in joints.items()}
    if not np.isfinite(np.fromiter(result.values(), dtype=np.float64)).all():
        raise ValueError(f"neutral pose contains non-finite values: {path}")
    return result


def vector_by_joint_name(
    neutral_by_name: Mapping[str, float], joint_names: Sequence[str]
) -> np.ndarray:
    """Return neutral values in ``joint_names`` order, requiring an exact name set."""
    names = [str(name) for name in joint_names]
    if len(names) != len(set(names)):
        raise ValueError("destination joint names contain duplicates")
    expected, actual = set(names), set(neutral_by_name)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"neutral joint-name mismatch; missing={missing}, extra={extra}")
    return np.asarray([neutral_by_name[name] for name in names], dtype=np.float64)


def prime_optimizer_neutral(optimizer, neutral_by_name: Mapping[str, float]) -> np.ndarray:
    """Use neutral only as the no-history initial guess, not as loss regularization."""
    neutral = vector_by_joint_name(neutral_by_name, optimizer.robot.dof_joint_names)
    limits = np.asarray(optimizer.robot.joint_limits, dtype=np.float64)
    if np.any(neutral < limits[:, 0]) or np.any(neutral > limits[:, 1]):
        bad = np.flatnonzero((neutral < limits[:, 0]) | (neutral > limits[:, 1]))
        names = [optimizer.robot.dof_joint_names[index] for index in bad]
        raise ValueError(f"neutral pose violates optimizer joint limits: {names}")

    original_get_init = optimizer._get_init_qpos

    def get_init_with_neutral(self, last_qpos):
        if last_qpos is None and self.last_qpos is None:
            return neutral.copy()
        return original_get_init(last_qpos)

    optimizer._get_init_qpos = MethodType(get_init_with_neutral, optimizer)
    return neutral
