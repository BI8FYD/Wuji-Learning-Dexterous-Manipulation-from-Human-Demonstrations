from __future__ import annotations

import json

import numpy as np
import pytest

from wuji_learn_from_human.deployment import (
  PolicyReferenceMetadata,
  validate_policy_reference,
)
from wuji_learn_from_human.reference import (
  JOINT_NAMES_20,
  SCHEMA_VERSION,
  load_reference,
  sha256_file,
)
from wuji_learn_from_human.residual import ResidualController


def _write(path, *, names=JOINT_NAMES_20, frame="wuji_wrist_tag"):
  count = 4
  quaternion = np.zeros((count, 4), dtype=np.float32)
  quaternion[:, 0] = 1.0
  np.savez(
    path, schema_version=np.asarray(SCHEMA_VERSION), demo_id=np.asarray("demo"),
    q_ref=np.zeros((count, 20), np.float32),
    q_ref_velocity=np.zeros((count, 20), np.float32), joint_names=np.asarray(names),
    object_position=np.zeros((count, 3), np.float32),
    object_quaternion_wxyz=quaternion,
    object_linear_velocity=np.zeros((count, 3), np.float32),
    object_angular_velocity=np.zeros((count, 3), np.float32),
    valid_reset_mask=np.array([True, True, True, False]), dt=np.asarray(0.05),
    object_frame=np.asarray(frame), object_size_m=np.full(3, 0.06, np.float32),
    quaternion_convention=np.asarray("wxyz"),
    robot_to_tag_position_m=np.zeros(3, np.float32),
    robot_to_tag_quaternion_wxyz=np.asarray([1, 0, 0, 0], np.float32),
    source_hashes_json=np.asarray(json.dumps({"q_ik": "abc"})),
  )


def test_reference_contract_round_trip(tmp_path):
  path = tmp_path / "demo.npz"
  _write(path)
  reference = load_reference(path)
  assert reference.q.shape == (4, 20)
  assert reference.dt == pytest.approx(0.05)
  assert reference.q.flags.writeable is False


def test_reference_rejects_joint_order_and_frame(tmp_path):
  path = tmp_path / "bad.npz"
  _write(path, names=tuple(reversed(JOINT_NAMES_20)))
  with pytest.raises(ValueError, match="joint_names"):
    load_reference(path)
  _write(path, frame="world")
  with pytest.raises(ValueError, match="object_frame"):
    load_reference(path)


def test_policy_reference_pair_is_verified(tmp_path):
  path = tmp_path / "demo.npz"
  _write(path)
  reference = load_reference(path)
  metadata = PolicyReferenceMetadata(
    reference_sha256=sha256_file(path), reference_schema_version=SCHEMA_VERSION,
    reference_joint_names=JOINT_NAMES_20, control_dt=0.05,
    residual_scale=0.2, ema_alpha=0.5,
  )
  validate_policy_reference(metadata, reference)


def test_residual_controller_filters_only_the_learned_correction():
  controller = ResidualController(
    np.full(20, -1.0), np.full(20, 1.0), residual_scale=0.2, ema_alpha=0.5
  )
  assert np.allclose(controller.step(np.zeros(20), np.ones(20)), 0.1)
  assert np.allclose(controller.step(np.full(20, 0.5), np.zeros(20)), 0.55)
