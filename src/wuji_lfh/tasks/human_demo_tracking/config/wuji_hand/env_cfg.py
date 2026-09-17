"""Wuji Hand binding for DemoTrack."""

from __future__ import annotations

import os

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg

from wuji_lfh.shared import HAND_INIT_STATE, OBJECT_INIT_STATE

from ...demo_track_env_cfg import make_demo_track_env_cfg
from ...reference import load_reference


def wuji_hand_demo_track_env_cfg(
  reference_path: str | None = None, *, play: bool = False,
  num_envs: int = 4096, dr_stage: int = 1,
):
  reference_path = reference_path or os.environ.get("WUJI_DEMO_TRACK_REFERENCE", "")
  cfg = make_demo_track_env_cfg(
    reference_path, play=play, num_envs=num_envs, dr_stage=dr_stage
  )
  edge = None
  if reference_path:
    size = load_reference(reference_path).object_size_m
    if not all(abs(float(value) - float(size[0])) < 1e-6 for value in size):
      raise ValueError("MVP cube asset requires equal object_size_m edges")
    edge = float(size[0])
  robot = get_wuji_hand_cfg()
  # IK solves against the hardware command limits. The base robot's
  # generic 0.9 soft-limit shrink would make valid q_ik boundary samples
  # unreachable and create a permanent imitation error.
  robot.articulation.soft_joint_pos_limit_factor = 1.0
  cfg.scene.entities = {"robot": robot, "object": get_inhand_object_cfg(edge_m=edge)}
  cfg.scene.entities["robot"].init_state = HAND_INIT_STATE
  cfg.scene.entities["object"].init_state = OBJECT_INIT_STATE
  cfg.viewer.body_name = "right_palm_link"
  return cfg
