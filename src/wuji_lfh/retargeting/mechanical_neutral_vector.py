"""Minimal kinematic IK built on Wuji's official vector optimizer plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .neutral_pose import load_neutral_pose, vector_by_joint_name


def create_retargeter(config_path: Path, hand_side: str):
    """Create an official Retargeter, with the local extension when requested."""
    from wuji_lfh._vendor.wuji_retargeting import Retargeter

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    optimizer_type = (config.get("optimizer") or {}).get("type")
    if optimizer_type not in {
        "MechanicalNeutralVectorOptimizer"
    }:
        return Retargeter.from_yaml(str(config_path), hand_side)

    config["__yaml_dir"] = str(config_path.parent)
    config.setdefault("optimizer", {})["hand_side"] = hand_side

    # Retargeter owns coordinate transforms and filtering. Construct it with a
    # registered upstream type, then replace only its optimizer with our local
    # subclass of the official VectorOptimizer.
    upstream_config = dict(config)
    upstream_config["optimizer"] = dict(config["optimizer"])
    upstream_config["optimizer"]["type"] = "VectorOptimizer"
    retargeter = Retargeter.from_config(upstream_config, hand_side)
    retargeter.optimizer = MechanicalNeutralVectorOptimizer(config)
    return retargeter


def _q_index_for_link(robot, link_name: str) -> int:
    frame_id = robot.get_link_index(link_name)
    frame = robot.model.frames[frame_id]
    # Pinocchio 2.x exposes ``parent``; 3.x renamed it to ``parentJoint``.
    joint_id = getattr(frame, "parentJoint", frame.parent)
    if joint_id <= 0 or robot.model.nqs[joint_id] != 1:
        raise ValueError(f"expected one-DoF parent joint for {link_name!r}")
    return int(robot.model.idx_qs[joint_id])


class MechanicalNeutralVectorOptimizer:
    """Factory-style placeholder replaced by the real subclass on construction."""

    def __new__(cls, config: dict):
        # Import lazily so extraction/test environments do not require Pinocchio.
        from wuji_lfh._vendor.wuji_retargeting.opt.vector import VectorOptimizer

        class _Implementation(VectorOptimizer):
            def __init__(self, implementation_config: dict):
                super().__init__(implementation_config)
                # Local key vectors do not mention the palm, but the auxiliary
                # wrist-to-tip objective needs it in the same FK/Jacobian batch.
                if self.origin_link_name not in self._kv_computed_link_names:
                    self._kv_computed_link_names.append(self.origin_link_name)
                    self._kv_computed_link_indices.append(
                        self.robot.get_link_index(self.origin_link_name)
                    )
                retarget = implementation_config.get("retarget", {})
                neutral_file = retarget.get("neutral_pose")
                if not neutral_file:
                    raise ValueError("retarget.neutral_pose is required")
                neutral_path = Path(neutral_file)
                if not neutral_path.is_absolute():
                    neutral_path = Path(implementation_config["__yaml_dir"]) / neutral_path
                neutral_by_name = load_neutral_pose(
                    neutral_path, implementation_config["optimizer"]["hand_side"]
                )
                self.neutral_qpos = vector_by_joint_name(
                    neutral_by_name, self.robot.dof_joint_names
                )
                self.lambda_direction = float(retarget.get("lambda_direction", 1.0))
                self.lambda_tip = float(retarget.get("lambda_tip", 0.02))
                self.lambda_continuity = float(retarget.get("lambda_continuity", 0.01))
                self.tip_reference_cm = 100.0 * float(
                    retarget.get("tip_reference_length_m", 0.09)
                )
                self.continuity_reference_rad = float(
                    retarget.get("continuity_reference_rad", 0.1)
                )
                self.tip_scalings = np.asarray(
                    retarget.get("tip_scalings", [1.0] * 5), dtype=np.float64
                )
                self.segment_weights = np.asarray(
                    retarget.get("segment_weights", [1.0] * self.num_vectors),
                    dtype=np.float64,
                )
                if min(self.lambda_direction, self.lambda_tip, self.lambda_continuity) < 0:
                    raise ValueError("minimal kinematic objective weights must be non-negative")
                if self.tip_reference_cm <= 0 or self.continuity_reference_rad <= 0:
                    raise ValueError("objective reference scales must be positive")
                if self.tip_scalings.shape != (5,) or np.any(self.tip_scalings <= 0):
                    raise ValueError("retarget.tip_scalings must contain five positive values")
                if self.segment_weights.shape != (self.num_vectors,) or np.any(
                    self.segment_weights <= 0
                ):
                    raise ValueError(
                        f"retarget.segment_weights must contain {self.num_vectors} positive values"
                    )

                # Exclude hyper-extension from the feasible space. Resolve every
                # flexion DoF through the parent joint of a semantic robot link;
                # spread/opposition DoFs retain their physical URDF ranges.
                lower = self.robot.joint_limits[:, 0].copy()
                flexion_links = list(self.link3_names) + list(self.link4_names)
                flexion_links += list(self.link1_names[1:])
                self.flexion_q_indices = sorted({
                    _q_index_for_link(self.robot, link_name)
                    for link_name in flexion_links
                })
                for q_index in self.flexion_q_indices:
                    lower[q_index] = max(lower[q_index], self.neutral_qpos[q_index])
                self.opt.set_lower_bounds(lower.tolist())
                self.optimization_lower_limits = lower
                self.optimization_upper_limits = self.robot.joint_limits[:, 1].copy()

            def _get_init_qpos(self, last_qpos):
                if last_qpos is None and self.last_qpos is None:
                    return self.neutral_qpos.copy()
                return super()._get_init_qpos(last_qpos)

            @staticmethod
            def _normalize_vectors(vectors):
                lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(lengths < 1e-8):
                    raise ValueError("degenerate finger segment in retarget target")
                return vectors / lengths

            def _compute_targets(self, keypoints):
                points = np.asarray(keypoints, dtype=np.float64)
                local_vectors = (
                    points[self._task_kp_indices] - points[self._origin_kp_indices]
                )
                directions = self._normalize_vectors(local_vectors)
                tips = self._compute_tip_vectors(points)
                tips *= self.tip_scalings[:, None]
                return directions, tips

            def _robot_geometry(self, qpos, with_jacobians=True):
                self.robot.compute_forward_kinematics(qpos)
                positions = np.asarray([
                    self.robot.get_link_pose(index)[:3, 3]
                    for index in self._kv_computed_link_indices
                ]) * 100.0
                jacobians = None
                if with_jacobians:
                    jacobians = self.robot.compute_all_jacobians_batch(
                        qpos, self._kv_computed_link_indices
                    ) * 100.0
                vectors = positions[self._kv_task_indices] - positions[self._kv_origin_indices]
                directions = self._normalize_vectors(vectors)
                palm_index = self._kv_computed_link_names.index(self.origin_link_name)
                tip_indices = np.asarray([
                    self._kv_computed_link_names.index(name) for name in self.task_link_names
                ])
                tip_vectors = positions[tip_indices] - positions[palm_index]
                return vectors, directions, tip_vectors, jacobians, palm_index, tip_indices

            def _minimal_loss_and_grad(self, qpos, targets, last_qpos):
                qpos = np.asarray(qpos, dtype=np.float64)
                target_dirs, target_tips = targets
                vectors, robot_dirs, robot_tips, jacobians, palm_i, tip_is = (
                    self._robot_geometry(qpos, with_jacobians=True)
                )
                n_segments = float(self.num_vectors)
                dir_diff = robot_dirs - target_dirs
                tip_diff = robot_tips - target_tips
                direction_raw = np.sum(
                    self.segment_weights[:, None] * dir_diff ** 2
                ) / n_segments
                tip_scale = self.tip_reference_cm
                continuity_scale = self.continuity_reference_rad * np.sqrt(self.num_joints)
                tip_raw = np.mean(np.sum((tip_diff / tip_scale) ** 2, axis=1))
                continuity_raw = 0.0
                if last_qpos is not None:
                    continuity_raw = float(np.sum(
                        ((qpos - last_qpos) / continuity_scale) ** 2
                    ))
                loss = self.lambda_direction * direction_raw
                loss += self.lambda_tip * tip_raw
                loss += self.lambda_continuity * continuity_raw
                grad = np.zeros(self.num_joints, dtype=np.float64)
                identity = np.eye(3)
                for i in range(self.num_vectors):
                    vector_jacobian = (
                        jacobians[self._kv_task_indices[i]]
                        - jacobians[self._kv_origin_indices[i]]
                    )
                    direction_jacobian = (
                        (identity - np.outer(robot_dirs[i], robot_dirs[i]))
                        / np.linalg.norm(vectors[i])
                    ) @ vector_jacobian
                    grad += (
                        2.0 * self.lambda_direction * self.segment_weights[i]
                        / n_segments
                    ) * (dir_diff[i] @ direction_jacobian)
                for i, tip_i in enumerate(tip_is):
                    tip_jacobian = jacobians[tip_i] - jacobians[palm_i]
                    grad += (
                        2.0 * self.lambda_tip / (5.0 * tip_scale ** 2)
                    ) * (tip_diff[i] @ tip_jacobian)
                if last_qpos is not None:
                    delta = qpos - last_qpos
                    grad += (
                        2.0 * self.lambda_continuity / continuity_scale ** 2
                    ) * delta
                return float(loss), grad

            def objective_terms(self, qpos, mediapipe_keypoints, last_qpos=None):
                """Return exactly the raw and weighted terms used by the solver."""
                qpos = np.asarray(qpos, dtype=np.float64)
                target_dirs, target_tips = self._compute_targets(mediapipe_keypoints)
                _, robot_dirs, robot_tips, _, _, _ = self._robot_geometry(
                    qpos, with_jacobians=False
                )
                direction_raw = float(np.sum(
                    self.segment_weights[:, None] * (robot_dirs - target_dirs) ** 2
                ) / self.num_vectors)
                tip_scale = self.tip_reference_cm
                tip_raw = float(np.mean(np.sum(
                    ((robot_tips - target_tips) / tip_scale) ** 2, axis=1
                )))
                continuity_raw = 0.0
                if last_qpos is not None:
                    continuity_scale = (
                    self.continuity_reference_rad * np.sqrt(self.num_joints)
                    )
                    continuity_raw = float(np.sum(
                        ((qpos - np.asarray(last_qpos)) / continuity_scale) ** 2
                    ))
                weighted = {
                    "direction": self.lambda_direction * direction_raw,
                    "tip": self.lambda_tip * tip_raw,
                    "continuity": self.lambda_continuity * continuity_raw,
                }
                return {
                    "raw": {
                        "direction": direction_raw,
                        "tip": tip_raw,
                        "continuity": continuity_raw,
                    },
                    "weighted": weighted,
                    "total": float(sum(weighted.values())),
                }

            def _get_objective(self, targets, last_qpos):
                def objective(x, grad_out):
                    loss, gradient = self._minimal_loss_and_grad(x, targets, last_qpos)
                    if grad_out.size:
                        grad_out[:] = gradient
                    return loss
                return objective

            def solve(self, mediapipe_keypoints, last_qpos=None):
                points = np.asarray(mediapipe_keypoints, dtype=np.float64)
                if points.shape != (21, 3):
                    raise ValueError(f"Expected shape (21, 3), got {points.shape}")
                targets = self._compute_targets(points)
                reg_qpos = self._get_reg_qpos(last_qpos)
                init_qpos = self._get_init_qpos(last_qpos)
                return self._run_optimization(self._get_objective(targets, reg_qpos), init_qpos)

            def compute_cost(self, qpos, mediapipe_keypoints):
                loss, _ = self._minimal_loss_and_grad(
                    qpos, self._compute_targets(mediapipe_keypoints), None
                )
                return loss

            def kinematic_diagnostics(self, qpos, mediapipe_keypoints):
                target_dirs, target_tips = self._compute_targets(mediapipe_keypoints)
                _, robot_dirs, robot_tips, _, _, _ = self._robot_geometry(
                    np.asarray(qpos, dtype=np.float64), with_jacobians=False
                )
                dots = np.sum(target_dirs * robot_dirs, axis=1)
                return {
                    "human_directions": target_dirs,
                    "robot_directions": robot_dirs,
                    "direction_error": np.linalg.norm(robot_dirs - target_dirs, axis=1),
                    "direction_error_deg": np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0))),
                    "human_tip_vectors_cm": target_tips,
                    "robot_tip_vectors_cm": robot_tips,
                    "tip_error_cm": np.linalg.norm(robot_tips - target_tips, axis=1),
                }

        return _Implementation(config)
