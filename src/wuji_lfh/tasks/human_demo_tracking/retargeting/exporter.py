"""Convert a WrenchRetarget q_ik run into the DemoTrack reference contract.

This module intentionally owns only the IK-output boundary.  Contact, Pre-touch,
Upper and wrench optimization are not dependencies of DemoTrack.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from wuji_lfh.calibration import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ

from ..reference import JOINT_NAMES_20, SCHEMA_VERSION, sha256_file


def _load_wrench_run(run_dir: Path):
  manifest_path = run_dir / "run.json"
  q_path = run_dir / "q_ik.npz"
  if not manifest_path.is_file() or not q_path.is_file():
    raise FileNotFoundError("run must contain run.json and q_ik.npz")
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("stages", {}).get("retarget") != "complete":
    raise ValueError("DemoTrack requires a completed WrenchRetarget retarget stage")
  with np.load(q_path, allow_pickle=False) as archive:
    q = np.asarray(archive["q_ik"], dtype=np.float64)
    fps = float(archive["fps"])
    names = tuple(str(v) for v in archive["joint_names"].tolist())
  if names != JOINT_NAMES_20:
    raise ValueError("q_ik joint order differs from Wuji encoder order")
  start = int(manifest["frames"]["start"])
  stop = int(manifest["frames"]["stop"])
  if q.shape != (stop - start, 20):
    raise ValueError("q_ik length and manifest source interval disagree")
  source_path = Path(manifest["source"]["path"]).expanduser().resolve()
  if not source_path.is_file():
    source_path = run_dir / "contact" / "source.npz"
  raw_path = run_dir / manifest["stage_outputs"]["retarget"]["raw"]["path"]
  if not source_path.is_file() or not raw_path.is_file():
    raise FileNotFoundError("source.npz or retarget/raw.pkl is unavailable")
  return manifest, q_path, q, fps, source_path, raw_path, start, stop


def _estimate_mano_to_robot(points: np.ndarray) -> np.ndarray:
  centered = points - points[0]
  selected = centered[[0, 5, 9]]
  x_vector = selected[0] - selected[2]
  demeaned = selected - selected.mean(axis=0, keepdims=True)
  _, _, vh = np.linalg.svd(demeaned)
  normal = vh[2]
  x_axis = x_vector - np.dot(x_vector, normal) * normal
  x_axis /= np.linalg.norm(x_axis)
  z_axis = np.cross(x_axis, normal)
  if np.dot(z_axis, demeaned[1] - demeaned[2]) < 0:
    normal *= -1
    z_axis *= -1
  wrist_frame = np.stack([x_axis, normal, z_axis], axis=1)
  operator_to_mano_right = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  )
  return wrist_frame @ operator_to_mano_right


def _object_in_robot_frame(
  points: np.ndarray, hand_rotation: np.ndarray, hand_translation: np.ndarray,
  object_rotation: np.ndarray, object_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  r_demo_robot = hand_rotation @ _estimate_mano_to_robot(points)
  x_demo_robot = hand_rotation @ points[0] + hand_translation
  r_robot_object = r_demo_robot.T @ object_rotation
  x_robot_object = r_demo_robot.T @ (object_translation - x_demo_robot)
  return r_robot_object, x_robot_object


def _resample(
  q: np.ndarray, positions: np.ndarray, rotations: Rotation, source_dt: float,
  target_dt: float,
):
  source_time = np.arange(len(q), dtype=np.float64) * source_dt
  target_time = np.arange(
    0.0, source_time[-1] + target_dt * 0.25, target_dt, dtype=np.float64
  )
  target_time = target_time[target_time <= source_time[-1] + 1e-9]
  q_out = np.stack([
    np.interp(target_time, source_time, q[:, joint]) for joint in range(q.shape[1])
  ], axis=1)
  pos_out = np.stack([
    np.interp(target_time, source_time, positions[:, axis]) for axis in range(3)
  ], axis=1)
  rot_out = Slerp(source_time, rotations)(target_time)
  return target_time, q_out, pos_out, rot_out


def _lowpass(
  q: np.ndarray, position: np.ndarray, rotation: Rotation, source_fps: float,
  cutoff_hz: float,
):
  """Zero-phase anti-alias filter before reducing the demonstration rate."""
  if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * source_fps:
    raise ValueError("lowpass_hz must be in (0, source_fps / 2)")
  if len(q) < 16:
    raise ValueError("anti-alias filtering requires at least 16 source frames")
  q_min, q_max = q.min(axis=0), q.max(axis=0)
  sos = butter(4, cutoff_hz, btype="lowpass", fs=source_fps, output="sos")
  q_filtered = sosfiltfilt(sos, q, axis=0)
  q_filtered = np.clip(q_filtered, q_min, q_max)
  position_filtered = sosfiltfilt(sos, position, axis=0)
  quaternion = rotation.as_quat()
  for index in range(1, len(quaternion)):
    if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
      quaternion[index] *= -1.0
  quaternion = sosfiltfilt(sos, quaternion, axis=0)
  quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
  return q_filtered, position_filtered, Rotation.from_quat(quaternion)


def _velocities(q, position, rotation: Rotation, dt: float):
  qd = np.gradient(q, dt, axis=0)
  linear = np.gradient(position, dt, axis=0)
  # Left relative rotation gives angular velocity in the fixed tag frame.
  relative = rotation[1:] * rotation[:-1].inv()
  angular_step = relative.as_rotvec() / dt
  angular = np.empty((len(q), 3), dtype=np.float64)
  angular[:-1] = angular_step
  angular[-1] = angular_step[-1]
  return qd, linear, angular


def export_wrench_run(
  run_dir: str | Path,
  output: str | Path,
  *,
  control_dt: float = 0.05,
  lowpass_hz: float = 8.0,
  robot_to_tag_position_m=TAG_IN_PALM_POS,
  robot_to_tag_quaternion_wxyz=TAG_IN_PALM_QUAT_WXYZ,
) -> Path:
  """Export q_ik and synchronized object motion in the calibrated tag frame."""
  run_dir = Path(run_dir).expanduser().resolve()
  output = Path(output).expanduser().resolve()
  manifest, q_path, q, fps, source_path, raw_path, start, stop = _load_wrench_run(run_dir)
  if not np.isfinite(control_dt) or control_dt <= 0.0:
    raise ValueError("control_dt must be positive")
  with np.load(source_path, allow_pickle=True) as archive:
    metadata = archive["metadata"].item()
    hand_r = Rotation.from_rotvec(archive["hand_orientations_axis_angle"]).as_matrix()
    hand_t = np.asarray(archive["hand_translations"], dtype=np.float64)
    object_r = Rotation.from_quat(archive["object_orientations_quat_xyzw"]).as_matrix()
    object_t = np.asarray(archive["object_translations"], dtype=np.float64)
  with raw_path.open("rb") as stream:
    raw = pickle.load(stream)
  if len(raw) != len(q):
    raise ValueError("retarget/raw.pkl is not aligned with q_ik")

  r_robot_object, x_robot_object = [], []
  for local, source_index in enumerate(range(start, stop)):
    points = np.asarray(raw[local]["right_fingers"], dtype=np.float64)
    rotation, position = _object_in_robot_frame(
      points, hand_r[source_index], hand_t[source_index],
      object_r[source_index], object_t[source_index],
    )
    r_robot_object.append(rotation)
    x_robot_object.append(position)
  r_robot_object = np.asarray(r_robot_object)
  x_robot_object = np.asarray(x_robot_object)

  q_rt = np.asarray(robot_to_tag_quaternion_wxyz, dtype=np.float64)
  q_rt /= np.linalg.norm(q_rt)
  r_robot_tag = Rotation.from_quat([q_rt[1], q_rt[2], q_rt[3], q_rt[0]]).as_matrix()
  x_robot_tag = np.asarray(robot_to_tag_position_m, dtype=np.float64)
  # T_tag_object = inv(T_robot_tag) * T_robot_object.
  r_tag_object = np.einsum("ij,njk->nik", r_robot_tag.T, r_robot_object)
  x_tag_object = np.einsum("ij,nj->ni", r_robot_tag.T, x_robot_object - x_robot_tag)
  source_rotations = Rotation.from_matrix(r_tag_object)
  q, x_tag_object, source_rotations = _lowpass(
    q, x_tag_object, source_rotations, fps, lowpass_hz
  )
  target_time, q, position, rotations = _resample(
    q, x_tag_object, source_rotations, 1.0 / fps, control_dt
  )
  qd, linear, angular = _velocities(q, position, rotations, control_dt)
  xyzw = rotations.as_quat()
  wxyz = xyzw[:, [3, 0, 1, 2]]
  wxyz[wxyz[:, 0] < 0] *= -1
  valid = np.ones(len(q), dtype=bool)
  # Random resets need enough remaining trajectory to produce a useful rollout.
  valid[-min(40, max(len(q) - 1, 1)):] = False
  hashes = {
    "q_ik": sha256_file(q_path), "source": sha256_file(source_path),
    "run_manifest": sha256_file(run_dir / "run.json"),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    schema_version=np.asarray(SCHEMA_VERSION), demo_id=np.asarray(run_dir.name),
    q_ref=q.astype(np.float32), q_ref_velocity=qd.astype(np.float32),
    joint_names=np.asarray(JOINT_NAMES_20),
    object_position=position.astype(np.float32),
    object_quaternion_wxyz=wxyz.astype(np.float32),
    object_linear_velocity=linear.astype(np.float32),
    object_angular_velocity=angular.astype(np.float32),
    valid_reset_mask=valid, dt=np.asarray(control_dt, dtype=np.float64),
    lowpass_hz=np.asarray(lowpass_hz, dtype=np.float64),
    source_fps=np.asarray(fps, dtype=np.float64), source_time_s=target_time,
    object_frame=np.asarray("wuji_wrist_tag"), quaternion_convention=np.asarray("wxyz"),
    object_class=np.asarray(str(metadata["object_class"])),
    object_size_m=np.asarray(metadata["object_size"], dtype=np.float32),
    robot_to_tag_position_m=x_robot_tag.astype(np.float32),
    robot_to_tag_quaternion_wxyz=q_rt.astype(np.float32),
    source_hashes_json=np.asarray(json.dumps(hashes, sort_keys=True)),
  )
  return output

# Stable alias for the standalone IK-only workflow.
export_ik_run = export_wrench_run
