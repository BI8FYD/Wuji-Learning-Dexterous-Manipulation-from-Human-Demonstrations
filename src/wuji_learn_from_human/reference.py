"""Versioned, simulator-independent reference trajectory contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "wuji-human-demo-v1"
JOINT_NAMES_20 = tuple(
  f"right_finger{finger}_joint{joint}"
  for finger in range(1, 6)
  for joint in range(1, 5)
)


@dataclass(frozen=True)
class ReferenceTrajectory:
  """An immutable, time-synchronised IK and object-motion trajectory."""

  q: np.ndarray
  qd: np.ndarray
  object_position: np.ndarray
  object_quaternion_wxyz: np.ndarray
  object_linear_velocity: np.ndarray
  object_angular_velocity: np.ndarray
  valid_reset_mask: np.ndarray
  dt: float
  joint_names: tuple[str, ...]
  object_size_m: np.ndarray
  demo_id: str
  object_frame: str
  quaternion_convention: str
  robot_to_tag_position_m: np.ndarray
  robot_to_tag_quaternion_wxyz: np.ndarray
  source_hashes: dict[str, str]
  path: Path

  @property
  def length(self) -> int:
    return int(self.q.shape[0])


def sha256_file(path: str | Path) -> str:
  """Return the SHA-256 of a file without loading it entirely into memory."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, key: str):
  value = archive[key]
  if value.shape != ():
    raise ValueError(f"{key} must be a scalar, got {value.shape}")
  return value.item()


def load_reference(path: str | Path) -> ReferenceTrajectory:
  """Load and fail-closed validate one immutable human-demonstration archive."""
  resolved = Path(path).expanduser().resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f"reference trajectory not found: {resolved}")
  with np.load(resolved, allow_pickle=False) as archive:
    required = {
      "schema_version", "demo_id", "q_ref", "q_ref_velocity", "joint_names",
      "object_position", "object_quaternion_wxyz", "object_linear_velocity",
      "object_angular_velocity", "valid_reset_mask", "dt", "object_frame",
      "object_size_m", "source_hashes_json", "quaternion_convention",
      "robot_to_tag_position_m", "robot_to_tag_quaternion_wxyz",
    }
    missing = required.difference(archive.files)
    if missing:
      raise ValueError(f"reference is missing fields: {sorted(missing)}")
    schema = str(_scalar(archive, "schema_version"))
    if schema != SCHEMA_VERSION:
      raise ValueError(f"unsupported reference schema {schema!r}")
    q = np.asarray(archive["q_ref"], dtype=np.float32)
    qd = np.asarray(archive["q_ref_velocity"], dtype=np.float32)
    position = np.asarray(archive["object_position"], dtype=np.float32)
    quaternion = np.asarray(archive["object_quaternion_wxyz"], dtype=np.float32)
    linear_velocity = np.asarray(archive["object_linear_velocity"], dtype=np.float32)
    angular_velocity = np.asarray(archive["object_angular_velocity"], dtype=np.float32)
    valid_reset = np.asarray(archive["valid_reset_mask"], dtype=bool)
    names = tuple(str(name) for name in archive["joint_names"].tolist())
    size = np.asarray(archive["object_size_m"], dtype=np.float32)
    dt = float(_scalar(archive, "dt"))
    demo_id = str(_scalar(archive, "demo_id"))
    object_frame = str(_scalar(archive, "object_frame"))
    convention = str(_scalar(archive, "quaternion_convention"))
    robot_to_tag_position = np.asarray(
      archive["robot_to_tag_position_m"], dtype=np.float32
    )
    robot_to_tag_quaternion = np.asarray(
      archive["robot_to_tag_quaternion_wxyz"], dtype=np.float32
    )
    source_hashes = json.loads(str(_scalar(archive, "source_hashes_json")))

  if q.ndim != 2 or q.shape[1] != 20 or q.shape[0] < 2:
    raise ValueError(f"q_ref must be [T,20], T>=2; got {q.shape}")
  count = q.shape[0]
  expected = {
    "q_ref_velocity": (qd, (count, 20)),
    "object_position": (position, (count, 3)),
    "object_quaternion_wxyz": (quaternion, (count, 4)),
    "object_linear_velocity": (linear_velocity, (count, 3)),
    "object_angular_velocity": (angular_velocity, (count, 3)),
    "valid_reset_mask": (valid_reset, (count,)),
  }
  for name, (value, shape) in expected.items():
    if value.shape != shape:
      raise ValueError(f"{name} must be {shape}, got {value.shape}")
  if names != JOINT_NAMES_20:
    raise ValueError(f"joint_names must match the WujiHand encoder order; got {names}")
  if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError("dt must be positive and finite")
  if size.shape != (3,) or np.any(size <= 0.0) or not np.isfinite(size).all():
    raise ValueError("object_size_m must be finite positive [3]")
  arrays = (q, qd, position, quaternion, linear_velocity, angular_velocity)
  if not all(np.isfinite(value).all() for value in arrays):
    raise ValueError("reference contains NaN or infinity")
  if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-4):
    raise ValueError("object quaternions must be normalized wxyz")
  if not valid_reset.any():
    raise ValueError("valid_reset_mask contains no usable frames")
  if object_frame != "wuji_wrist_tag":
    raise ValueError("object_frame must be 'wuji_wrist_tag'")
  if convention != "wxyz":
    raise ValueError("quaternion_convention must be 'wxyz'")
  if robot_to_tag_position.shape != (3,) or not np.isfinite(robot_to_tag_position).all():
    raise ValueError("robot_to_tag_position_m must be finite [3]")
  if robot_to_tag_quaternion.shape != (4,) or not np.allclose(
    np.linalg.norm(robot_to_tag_quaternion), 1.0, atol=1e-4
  ):
    raise ValueError("robot_to_tag_quaternion_wxyz must be normalized [4]")
  for value in (*arrays, valid_reset, size, robot_to_tag_position, robot_to_tag_quaternion):
    value.setflags(write=False)
  return ReferenceTrajectory(
    q=q, qd=qd, object_position=position,
    object_quaternion_wxyz=quaternion,
    object_linear_velocity=linear_velocity,
    object_angular_velocity=angular_velocity, valid_reset_mask=valid_reset,
    dt=dt, joint_names=names, object_size_m=size, demo_id=demo_id,
    object_frame=object_frame, quaternion_convention=convention,
    robot_to_tag_position_m=robot_to_tag_position,
    robot_to_tag_quaternion_wxyz=robot_to_tag_quaternion,
    source_hashes=source_hashes, path=resolved,
  )
