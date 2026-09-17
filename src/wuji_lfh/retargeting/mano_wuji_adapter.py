"""MANO-native input and morphology calibration for Wuji retargeting.

The upstream optimizer accepts a MediaPipe-shaped ``[21, 3]`` array.  This
module intentionally treats that as a topology contract rather than as a
license to apply MediaPipe camera corrections.  Every point here is metric
MANO geometry in the fixed-beta DexterHand convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PIP_INDICES = np.asarray((2, 6, 10, 14, 18), dtype=np.int64)
DIP_INDICES = np.asarray((3, 7, 11, 15, 19), dtype=np.int64)
TIP_INDICES = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)


@dataclass(frozen=True)
class ManoMorphology:
    """Resolved, single-application target scales for an adaptive solve."""

    tip_scaling: float
    segment_scaling: np.ndarray  # [finger, PIP/DIP/TIP]
    mano_radial_m: np.ndarray
    robot_radial_m: np.ndarray

    def as_dict(self) -> dict:
        return {
            "tip_scaling": float(self.tip_scaling),
            "segment_scaling": self.segment_scaling.tolist(),
            "mano_radial_m": self.mano_radial_m.tolist(),
            "robot_radial_m": self.robot_radial_m.tolist(),
            "finger_order": list(FINGER_NAMES),
            "target_order": ["PIP", "DIP", "TIP"],
        }


def validate_mano_keypoints(keypoints: np.ndarray, *, label: str = "MANO keypoints") -> np.ndarray:
    """Validate a metric native MANO 21-point sequence without modifying it."""
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim not in (2, 3) or points.shape[-2:] != (21, 3):
        raise ValueError(f"{label} must have shape [21,3] or [T,21,3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return points


def load_mano_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load fixed-beta neutral geometry emitted by the extraction stage."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MANO calibration not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"neutral_keypoints_m", "betas"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"MANO calibration missing fields: {sorted(missing)}")
        neutral = np.asarray(archive["neutral_keypoints_m"], dtype=np.float64)
        betas = np.asarray(archive["betas"], dtype=np.float64)
    validate_mano_keypoints(neutral, label="neutral MANO keypoints")
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError(f"MANO calibration betas must have shape [10], got {betas.shape}")
    return neutral, betas


def load_position_tips(path: Path, frame_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load separate position-only targets; old surface-tip sidecars fail closed."""
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        required = {"position_tips_m", "neutral_position_tips_m"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"MANO calibration missing axis-tip fields: {sorted(missing)}")
        frames = np.asarray(archive["position_tips_m"], dtype=np.float64)
        neutral = np.asarray(archive["neutral_position_tips_m"], dtype=np.float64)
    if (frames.ndim != 3 or frames.shape[1:] != (5, 3) or not len(frames)
            or (frame_count is not None and len(frames) != frame_count)
            or not np.isfinite(frames).all()):
        raise ValueError(f"position_tips_m must be finite [{frame_count or 'T'},5,3]")
    if neutral.shape != (5, 3) or not np.isfinite(neutral).all():
        raise ValueError("neutral_position_tips_m must be finite [5,3]")
    return frames, neutral


def _radial_lengths(points: np.ndarray) -> np.ndarray:
    points = validate_mano_keypoints(points)
    if points.ndim != 2:
        raise ValueError("neutral geometry must have shape [21,3]")
    wrist = points[0]
    targets = np.stack((points[PIP_INDICES], points[DIP_INDICES], points[TIP_INDICES]), axis=1)
    lengths = np.linalg.norm(targets - wrist, axis=-1)
    if np.any(lengths < 1e-7):
        raise ValueError("neutral morphology contains a degenerate wrist radial vector")
    return lengths


def resolve_morphology(
    mano_neutral_wrist_frame_m: np.ndarray,
    robot_radial_m: np.ndarray,
    *, mode: str,
) -> ManoMorphology:
    """Resolve morphology once for each official target family.

    ``AdaptiveOptimizerAnalytical.scaling`` is used only by its pinch tip
    position target.  ``segment_scaling`` is used only by its non-pinch
    full-hand radial targets.  Keeping their ledger separate prevents the
    legacy 0.09-m preprocessing from being accidentally applied a second time.
    """
    mano_radial = _radial_lengths(mano_neutral_wrist_frame_m)
    robot_radial = np.asarray(robot_radial_m, dtype=np.float64)
    if robot_radial.shape != (5, 3) or not np.isfinite(robot_radial).all():
        raise ValueError(f"robot radial lengths must have shape [5,3], got {robot_radial.shape}")
    if np.any(robot_radial < 1e-7):
        raise ValueError("robot neutral morphology contains a degenerate wrist radial vector")
    if mode == "identity":
        scales = np.ones((5, 3), dtype=np.float64)
    elif mode == "neutral_radial":
        scales = robot_radial / mano_radial
    else:
        raise ValueError("retarget.morphology_mode must be 'identity' or 'neutral_radial'")
    # Position target has one scalar whereas FullHand has one radial scale per
    # finger/target.  The geometric median is robust to thumb/pinky extremes.
    tip_scaling = float(np.median(scales[:, 2]))
    return ManoMorphology(tip_scaling, scales, mano_radial, robot_radial)
