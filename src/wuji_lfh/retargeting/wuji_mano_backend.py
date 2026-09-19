"""Thin MANO adapter around Wuji's pinned adaptive analytical optimizer.

No Wuji optimizer math is copied here.  The only subclass is the documented
Pinocchio 2/3 frame-parent compatibility shim; all target construction, Huber
losses, FK/Jacobians, warm start, limits, and NLopt use the pinned upstream
implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from .mano_wuji_adapter import (
    DIP_INDICES,
    TIP_INDICES,
    load_mano_calibration,
    load_position_tips,
    resolve_morphology,
    validate_mano_keypoints,
)
from .neutral_pose import load_neutral_pose, prime_optimizer_neutral


class PinCompatibleAdaptiveOptimizer:
    """Factory for an official optimizer with a Pinocchio 2.x index shim."""

    def __new__(cls, config: dict):
        from wuji_lfh._vendor.wuji_retargeting.opt.adaptive_analytical import (
            AdaptiveOptimizerAnalytical,
        )

        class _Implementation(AdaptiveOptimizerAnalytical):
            position_tip_keypoints = None

            def set_position_tip_keypoints(self, keypoints: np.ndarray) -> None:
                points = np.asarray(keypoints, dtype=np.float64)
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError("position target keypoints must be finite [21,3]")
                self.position_tip_keypoints = points.copy()

            def _compute_tip_vectors(self, keypoints, scaling=1.0):
                if self.position_tip_keypoints is None:
                    raise RuntimeError("MANO axis-tip position targets were not set")
                return super()._compute_tip_vectors(self.position_tip_keypoints, scaling)

            def _compute_full_hand_vectors(self, keypoints, scaling):
                if self.position_tip_keypoints is None:
                    raise RuntimeError("MANO axis-tip position targets were not set")
                position_points = np.asarray(keypoints, dtype=np.float64).copy()
                position_points[self.MP_TIP_INDICES] = self.position_tip_keypoints[self.MP_TIP_INDICES]
                return super()._compute_full_hand_vectors(position_points, scaling)

            def _resolve_flex_indices(self):
                def q_index(link_name: str) -> int:
                    frame_id = self.robot.get_link_index(link_name)
                    frame = self.robot.model.frames[frame_id]
                    joint_id = getattr(frame, "parentJoint", frame.parent)
                    if joint_id <= 0 or self.robot.model.nqs[joint_id] != 1:
                        raise RuntimeError(
                            f"expected one-DoF parent joint for {link_name!r}"
                        )
                    return int(self.robot.model.idx_qs[joint_id])

                pip_idx = [q_index(name) for name in self.link3_names]
                dip_idx = [q_index(name) for name in self.link4_names]
                combined = pip_idx + dip_idx
                if len(set(combined)) != len(combined):
                    raise RuntimeError(
                        "Resolved PIP/DIP qpos indices contain duplicates "
                        f"(pip={pip_idx}, dip={dip_idx})"
                    )
                self._pip_idx = np.asarray(pip_idx, dtype=np.int64)
                self._dip_idx = np.asarray(dip_idx, dtype=np.int64)
                self._flex_idx = np.asarray(sorted(combined), dtype=np.int64)

        return _Implementation(config)


def _rotation_xyz(points: np.ndarray, rotation: dict) -> np.ndarray:
    """Apply the same optional Euler adjustment as upstream ``Retargeter``."""
    x, y, z = (float(rotation.get(axis, 0.0)) for axis in ("x", "y", "z"))
    if x == 0.0 and y == 0.0 and z == 0.0:
        return points
    from scipy.spatial.transform import Rotation
    return points @ Rotation.from_euler("xyz", [x, y, z], degrees=True).as_matrix().T


class WujiManoBackend:
    """Sequential, unfiltered MANO-to-Wuji retargeting backend."""

    def __init__(self, config_path: Path, hand_side: str, calibration_path: Path):
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        self.config["__yaml_dir"] = str(config_path.parent)
        self.config.setdefault("optimizer", {})["hand_side"] = hand_side
        retarget = self.config.setdefault("retarget", {})
        if retarget.get("backend") != "wuji_mano":
            raise ValueError("WujiManoBackend requires retarget.backend: wuji_mano")
        if self.config["optimizer"].get("type") != "AdaptiveOptimizerAnalytical":
            raise ValueError("wuji_mano requires AdaptiveOptimizerAnalytical")
        for forbidden in ("video_z_scale", "wrist_middle_normalization", "correct_segments"):
            if retarget.get(forbidden, False):
                raise ValueError(f"wuji_mano forbids source-specific {forbidden}")

        self.hand_side = hand_side
        self.optimizer = PinCompatibleAdaptiveOptimizer(self.config)
        neutral_pose = retarget.get("neutral_pose")
        if not neutral_pose:
            raise ValueError("wuji_mano requires retarget.neutral_pose")
        neutral_path = Path(neutral_pose)
        if not neutral_path.is_absolute():
            neutral_path = config_path.parent / neutral_path
        self.neutral_qpos = prime_optimizer_neutral(
            self.optimizer, load_neutral_pose(neutral_path, hand_side)
        )

        self.rotation = dict(retarget.get("mediapipe_rotation") or {})
        self.wrist_offset_m = np.asarray(retarget.get("wrist_offset_cm", (0, 0, 0)), dtype=float) / 100
        self.thumb_offset_m = np.asarray(retarget.get("thumb_offset_cm", (0, 0, 0)), dtype=float) / 100
        if self.wrist_offset_m.shape != (3,) or self.thumb_offset_m.shape != (3,):
            raise ValueError("wrist_offset_cm and thumb_offset_cm must have three values")

        neutral_raw, self.betas = load_mano_calibration(calibration_path)
        self.position_tips_raw_m, neutral_position_tips = load_position_tips(calibration_path)
        neutral = self.transform_position_tips(neutral_raw, neutral_position_tips)
        morphology_mode = str(retarget.get("morphology_mode", "neutral_radial"))
        self.morphology = resolve_morphology(
            neutral, self._robot_neutral_radial_lengths(), mode=morphology_mode
        )
        # These are the only scale insertion points used by the official loss.
        self.optimizer.scaling = self.morphology.tip_scaling
        self.optimizer.segment_scaling = self.morphology.segment_scaling.copy()

    def transform(self, raw_keypoints_m: np.ndarray) -> np.ndarray:
        """Run only upstream wrist-frame construction plus explicit zero-safe offsets."""
        from wuji_lfh._vendor.wuji_retargeting.mediapipe import (
            apply_mediapipe_transformations,
        )

        points = validate_mano_keypoints(raw_keypoints_m, label="native MANO keypoints")
        if points.ndim != 2:
            raise ValueError("one frame [21,3] is required")
        transformed = apply_mediapipe_transformations(points, self.hand_side)
        transformed = _rotation_xyz(transformed, self.rotation)
        transformed = transformed.copy()
        transformed[5:] += self.wrist_offset_m
        transformed[1:5] += self.thumb_offset_m
        return transformed

    def transform_position_tips(self, raw_keypoints_m: np.ndarray,
                                position_tips_m: np.ndarray) -> np.ndarray:
        """Apply the same wrist frame to position-only tips, preserving 21-point input."""
        raw = validate_mano_keypoints(raw_keypoints_m)
        tips = np.asarray(position_tips_m, dtype=np.float64)
        if raw.shape != (21, 3) or tips.shape != (5, 3) or not np.isfinite(tips).all():
            raise ValueError("position tips require one finite [21,3] frame and [5,3] tips")
        position_points = raw.copy()
        position_points[TIP_INDICES] = tips
        return self.transform(position_points)

    def _robot_neutral_radial_lengths(self) -> np.ndarray:
        opt = self.optimizer
        opt.robot.compute_forward_kinematics(self.neutral_qpos)
        positions = np.asarray(
            [opt.robot.get_link_pose(index)[:3, 3] for index in opt.computed_link_indices],
            dtype=np.float64,
        )
        wrist = positions[opt.origin_indices[0]]
        return np.stack((
            np.linalg.norm(positions[opt.link3_indices] - wrist, axis=1),
            np.linalg.norm(positions[opt.link4_indices] - wrist, axis=1),
            np.linalg.norm(positions[opt.task_indices] - wrist, axis=1),
        ), axis=1)

    def _robot_targets(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        opt = self.optimizer
        opt.robot.compute_forward_kinematics(qpos)
        positions = np.asarray(
            [opt.robot.get_link_pose(index)[:3, 3] for index in opt.computed_link_indices],
            dtype=np.float64,
        )
        wrist = positions[opt.origin_indices[0]]
        tips = positions[opt.task_indices] - wrist
        distal = positions[opt.task_indices] - positions[opt.link4_indices]
        distal /= np.linalg.norm(distal, axis=1, keepdims=True).clip(min=1e-9)
        return tips, distal

    def solve_sequence(
        self, raw_keypoints_m: np.ndarray, *, segment_starts: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        raw = validate_mano_keypoints(raw_keypoints_m)
        if raw.ndim != 3 or not len(raw):
            raise ValueError("MANO sequence must have non-empty shape [T,21,3]")
        if len(self.position_tips_raw_m) != len(raw):
            raise ValueError("MANO position-tip frame count differs from replay")
        starts = {0} if segment_starts is None else set(segment_starts) | {0}
        q_values, targets, position_targets, full_hand_targets = [], [], [], []
        robot_tips, direction_errors, alphas = [], [], []
        costs, result_codes, evaluations = [], [], []
        started = time.perf_counter()
        for frame_index, frame in enumerate(raw):
            if frame_index in starts:
                self.optimizer.last_qpos = None
            transformed = self.transform(frame)
            position_points = self.transform_position_tips(
                frame, self.position_tips_raw_m[frame_index]
            )
            self.optimizer.set_position_tip_keypoints(position_points)
            qpos = self.optimizer.solve(transformed)
            q_values.append(qpos)
            # AdaptiveOptimizerAnalytical has two position target families.
            # Keep both in evidence and publish an alpha-weighted convenience
            # target for trajectory summaries; neither replaces the official
            # separate robust-loss terms.
            target_tip_position = position_points[TIP_INDICES] * self.morphology.tip_scaling
            target_tip_full_hand = position_points[TIP_INDICES] * self.morphology.segment_scaling[:, 2:3]
            robot_tip, robot_direction = self._robot_targets(qpos)
            target_direction = transformed[TIP_INDICES] - transformed[DIP_INDICES]
            target_direction /= np.linalg.norm(target_direction, axis=1, keepdims=True).clip(min=1e-9)
            cosine = np.clip(np.sum(robot_direction * target_direction, axis=1), -1.0, 1.0)
            alpha = self.optimizer._compute_pinch_alpha(transformed)
            targets.append(alpha[:, None] * target_tip_position +
                           (1.0 - alpha[:, None]) * target_tip_full_hand)
            position_targets.append(target_tip_position)
            full_hand_targets.append(target_tip_full_hand)
            robot_tips.append(robot_tip)
            direction_errors.append(np.degrees(np.arccos(cosine)))
            alphas.append(alpha)
            costs.append(self.optimizer.compute_cost(qpos, transformed))
            result_codes.append(int(self.optimizer.opt.last_optimize_result()))
            evaluations.append(int(self.optimizer.opt.get_numevals()))

        q_ik = np.asarray(q_values, dtype=np.float32)
        diagnostics = {
            "target_tip_wrist_m": np.asarray(targets, dtype=np.float32),
            "target_tip_position_wrist_m": np.asarray(position_targets, dtype=np.float32),
            "target_full_hand_tip_wrist_m": np.asarray(full_hand_targets, dtype=np.float32),
            "robot_tip_wrist_m": np.asarray(robot_tips, dtype=np.float32),
            "direction_error_deg": np.asarray(direction_errors, dtype=np.float32),
            "pinch_alphas": np.asarray(alphas, dtype=np.float32),
            "cost": np.asarray(costs, dtype=np.float64),
            "solver_result": np.asarray(result_codes, dtype=np.int32),
            "solver_evaluations": np.asarray(evaluations, dtype=np.int32),
            "elapsed_seconds": np.asarray(time.perf_counter() - started, dtype=np.float64),
        }
        return q_ik, np.asarray(self.optimizer.robot.dof_joint_names, dtype=str), diagnostics

    def write_evidence(self, output_dir: Path, raw_keypoints_m: np.ndarray,
                       diagnostics: dict[str, np.ndarray]) -> None:
        """Publish side evidence without changing the formal q_ik NPZ schema."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        transformed = np.asarray([self.transform(frame) for frame in raw_keypoints_m], dtype=np.float32)
        np.savez(
            output_dir / "adapter_input.npz",
            keypoints_mano_m=np.asarray(raw_keypoints_m, dtype=np.float32),
            keypoints_wuji_m=transformed,
            position_tips_mano_m=self.position_tips_raw_m.astype(np.float32),
            position_tips_wuji_m=np.asarray([
                self.transform_position_tips(frame, tips)[TIP_INDICES]
                for frame, tips in zip(raw_keypoints_m, self.position_tips_raw_m)
            ], dtype=np.float32),
        )
        np.savez(output_dir / "diagnostics.npz", **diagnostics)
        calibration = {
            "schema": "wrench-retarget-wuji-mano-v2",
            "frame_mode": "wuji_point_estimate_v1",
            "source": "MANO native 21 points; separate 90% distal-axis position tips",
            "tip_position_definition": "DIP/IP + 0.90 * distance to first mesh exit along preceding-joint axis",
            "betas": self.betas.tolist(),
            "morphology": self.morphology.as_dict(),
            "scale_ledger": {
                "video_z_scale": False,
                "wrist_middle_normalization": False,
                "correct_segments": False,
                "tip_position_scale": "optimizer.scaling only",
                "full_hand_scale": "optimizer.segment_scaling only",
            },
        }
        (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
