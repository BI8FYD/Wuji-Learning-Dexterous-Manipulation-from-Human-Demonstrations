"""DemoTrack termination terms."""

from __future__ import annotations

import torch

from .observations import cube_pos_in_tag


def trajectory_end(env):
  return env.command_manager.get_term("demo_trajectory").trajectory_end


def nonfinite_state(env):
  robot = env.scene["robot"]
  obj = env.scene["object"]
  return ~torch.isfinite(robot.data.joint_pos).all(-1) | ~torch.isfinite(
    obj.data.root_link_pose_w
  ).all(-1)


def object_reference_deviation(env, max_distance_m: float = 0.12):
  """Terminate a dropped/lost object without relying on Reorient cage state."""
  reference = env.command_manager.get_term("demo_trajectory").sample()[2]
  return torch.linalg.norm(cube_pos_in_tag(env) - reference, dim=-1) > max_distance_m
