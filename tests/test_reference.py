from __future__ import annotations

import json

import numpy as np
import pytest
from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import _build_config

from wuji_lfh.tasks.human_demo_tracking.reference import (
  JOINT_NAMES_20,
  SCHEMA_VERSION,
  load_reference,
)


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


def test_onnx_config_embeds_reference_contract(tmp_path):
  path = tmp_path / "demo.npz"
  _write(path)
  config = _build_config("unused", {
    "actions": {"joint_pos": {"residual_scale": 0.2, "ema_alpha": 0.5}},
    "commands": {"demo_trajectory": {
      "reference_path": str(path), "lookahead_frames": [0, 4],
    }},
    "decimation": 5,
    "sim": {"timestep": 0.01},
  })
  assert config["control_mode"] == "reference_residual"
  assert config["reference_schema_version"] == SCHEMA_VERSION
  assert config["reference_sha256"]
  assert config["reference_joint_names"] == list(JOINT_NAMES_20)
