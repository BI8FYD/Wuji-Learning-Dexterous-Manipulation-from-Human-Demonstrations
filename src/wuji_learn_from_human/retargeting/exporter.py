"""Build a simulator- and deployment-ready reference from an IK run.

The exporter owns only the boundary after IK retargeting: it consumes joint IK
output plus synchronised human/object motion and emits an immutable reference.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from ..calibration import (
  DEFAULT_TAG_IN_PALM_POSITION_M,
  DEFAULT_TAG_IN_PALM_QUATERNION_WXYZ,
)
from ..reference import JOINT_NAMES_20, SCHEMA_VERSION, sha256_file


def _load_ik_run(run_dir: Path):
  """Read the stable interchange files emitted by an IK retargeting run."""
  manifest_path = run_dir / "run.json"
  q_path = run_dir / "q_ik.npz"
  if not manifest_path.is_file() or not q_path.is_file():
    raise FileNotFoundError("ik run must contain run.json and q_ik.npz")
  manifest = json.loads(manifest_path.read_text())
  with np.load(q_path, allow_pickle=False) as archive:
    q = np.asarray(archive["q_ik"], dtype=np.float64)
    fps = float(archive["fps"])
    names = tuple(str(value) for value in archive["joint_names"].tolist())
  if names != JOINT_NAMES_20:
    raise ValueError("q_ik joint order differs from the WujiHand encoder order")
  start = int(manifest["frames"]["start"])
  stop = int(manifest["frames"]["stop"])
  if q.shape != (stop - start, 20):
    raise ValueError("q_ik length and manifest source interval disagree")
  source_path = Path(manifest["source"]["path"]).expanduser().resolve()
  if not source_path.is_file():
    source_path = run_dir / "source.npz"
  raw_path = run_dir / manifest["stage_outputs"]["retarget"]["raw"]["path"]
  if not source_path.is_file() or not raw_path.is_file():
    raise FileNotFoundError("source.npz or retarget/raw.pkl is unavailable")
  return manifest, q_path, q, fps, source_path, raw_path, start, stop


def _estimate_demo_to_robot(points: np.ndarray) -> np.ndarray:
  centered = points - points[0]
  selected = centered[[0, 5, 9]]
  x_vector = selected[0] - selected[2]
  _, _, vh = np.linalg.svd(selected - selected.mean(axis=0, keepdims=True))
  normal = vh[2]
  x_axis = x_vector - np.dot(x_vector, normal) * normal
  x_axis /= np.linalg.norm(x_axis)
  z_axis = np.cross(x_axis, normal)
  if np.dot(z_axis, selected[1] - selected[2]) < 0:
    normal *= -1
    z_axis *= -1
  demo_wrist_frame = np.stack([x_axis, normal, z_axis], axis=1)
  operator_to_mano_right = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  )
  return demo_wrist_frame @ operator_to_mano_right


def _object_in_robot_frame(
  points: np.ndarray, hand_rotation: np.ndarray, hand_translation: np.ndarray,
  object_rotation: np.ndarray, object_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  demo_to_robot = hand_rotation @ _estimate_demo_to_robot(points)
  wrist_origin = hand_rotation @ points[0] + hand_translation
  return (
    demo_to_robot.T @ object_rotation,
    demo_to_robot.T @ (object_translation - wrist_origin),
  )


def _resample(
  q: np.ndarray, position: np.ndarray, rotation: Rotation, source_dt: float,
  target_dt: float,
):
  source_time = np.arange(len(q), dtype=np.float64) * source_dt
  target_time = np.arange(0.0, source_time[-1] + target_dt * 0.25, target_dt)
  target_time = target_time[target_time <= source_time[-1] + 1e-9]
  q_out = np.stack([
    np.interp(target_time, source_time, q[:, joint]) for joint in range(q.shape[1])
  ], axis=1)
  position_out = np.stack([
    np.interp(target_time, source_time, position[:, axis]) for axis in range(3)
  ], axis=1)
  return target_time, q_out, position_out, Slerp(source_time, rotation)(target_time)


def _lowpass(
  q: np.ndarray, position: np.ndarray, rotation: Rotation, source_fps: float,
  cutoff_hz: float,
):
  if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * source_fps:
    raise ValueError("lowpass_hz must be in (0, source_fps / 2)")
  if len(q) < 16:
    raise ValueError("anti-alias filtering requires at least 16 source frames")
  q_min, q_max = q.min(axis=0), q.max(axis=0)
  sos = butter(4, cutoff_hz, btype="lowpass", fs=source_fps, output="sos")
  q_filtered = np.clip(sosfiltfilt(sos, q, axis=0), q_min, q_max)
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
  relative = rotation[1:] * rotation[:-1].inv()
  angular = np.empty((len(q), 3), dtype=np.float64)
  angular[:-1] = relative.as_rotvec() / dt
  angular[-1] = angular[-2]
  return qd, linear, angular


def export_ik_run(
  run_dir: str | Path,
  output: str | Path,
  *,
  control_dt: float = 0.05,
  lowpass_hz: float = 8.0,
  robot_to_tag_position_m=DEFAULT_TAG_IN_PALM_POSITION_M,
  robot_to_tag_quaternion_wxyz=DEFAULT_TAG_IN_PALM_QUATERNION_WXYZ,
) -> Path:
  """Export IK and object motion in the calibrated WujiHand wrist-tag frame."""
  run_dir = Path(run_dir).expanduser().resolve()
  output = Path(output).expanduser().resolve()
  manifest, q_path, q, fps, source_path, raw_path, start, stop = _load_ik_run(run_dir)
  if not np.isfinite(control_dt) or control_dt <= 0.0:
    raise ValueError("control_dt must be positive")
  with np.load(source_path, allow_pickle=True) as archive:
    metadata = archive["metadata"].item()
    hand_rotation = Rotation.from_rotvec(archive["hand_orientations_axis_angle"]).as_matrix()
    hand_translation = np.asarray(archive["hand_translations"], dtype=np.float64)
    object_rotation = Rotation.from_quat(archive["object_orientations_quat_xyzw"]).as_matrix()
    object_translation = np.asarray(archive["object_translations"], dtype=np.float64)
  with raw_path.open("rb") as stream:
    raw = pickle.load(stream)
  if len(raw) != len(q):
    raise ValueError("retarget/raw.pkl is not aligned with q_ik")

  object_rotations, object_positions = [], []
  for local, source_index in enumerate(range(start, stop)):
    rotation, position = _object_in_robot_frame(
      np.asarray(raw[local]["right_fingers"], dtype=np.float64),
      hand_rotation[source_index], hand_translation[source_index],
      object_rotation[source_index], object_translation[source_index],
    )
    object_rotations.append(rotation)
    object_positions.append(position)
  tag_in_robot_quaternion = np.asarray(robot_to_tag_quaternion_wxyz, dtype=np.float64)
  tag_in_robot_quaternion /= np.linalg.norm(tag_in_robot_quaternion)
  robot_to_tag_rotation = Rotation.from_quat(
    tag_in_robot_quaternion[[1, 2, 3, 0]]
  ).as_matrix()
  robot_to_tag_position = np.asarray(robot_to_tag_position_m, dtype=np.float64)
  robot_object_rotation = np.asarray(object_rotations)
  robot_object_position = np.asarray(object_positions)
  tag_object_rotation = np.einsum(
    "ij,njk->nik", robot_to_tag_rotation.T, robot_object_rotation
  )
  tag_object_position = np.einsum(
    "ij,nj->ni", robot_to_tag_rotation.T,
    robot_object_position - robot_to_tag_position,
  )
  q, tag_object_position, rotations = _lowpass(
    q, tag_object_position, Rotation.from_matrix(tag_object_rotation), fps, lowpass_hz
  )
  target_time, q, position, rotations = _resample(
    q, tag_object_position, rotations, 1.0 / fps, control_dt
  )
  qd, linear, angular = _velocities(q, position, rotations, control_dt)
  wxyz = rotations.as_quat()[:, [3, 0, 1, 2]]
  wxyz[wxyz[:, 0] < 0] *= -1
  valid_reset = np.ones(len(q), dtype=bool)
  valid_reset[-min(40, max(len(q) - 1, 1)):] = False
  hashes = {
    "q_ik": sha256_file(q_path),
    "source": sha256_file(source_path),
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
    valid_reset_mask=valid_reset, dt=np.asarray(control_dt, dtype=np.float64),
    lowpass_hz=np.asarray(lowpass_hz, dtype=np.float64),
    source_fps=np.asarray(fps, dtype=np.float64), source_time_s=target_time,
    object_frame=np.asarray("wuji_wrist_tag"),
    quaternion_convention=np.asarray("wxyz"),
    object_class=np.asarray(str(metadata["object_class"])),
    object_size_m=np.asarray(metadata["object_size"], dtype=np.float32),
    robot_to_tag_position_m=robot_to_tag_position.astype(np.float32),
    robot_to_tag_quaternion_wxyz=tag_in_robot_quaternion.astype(np.float32),
    source_hashes_json=np.asarray(json.dumps(hashes, sort_keys=True)),
  )
  return output
