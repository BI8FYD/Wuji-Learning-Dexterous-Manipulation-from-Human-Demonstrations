"""Minimal trajectory-tracking rewards."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from .observations import cube_pos_in_tag, object_quat_in_tag


def object_position_tracking(env, sigma: float = 0.02):
  ref = env.command_manager.get_term("demo_trajectory").sample()[2]
  error_sq = torch.sum(torch.square(cube_pos_in_tag(env) - ref), dim=-1)
  return torch.exp(-error_sq / sigma**2)


def object_orientation_tracking(env, sigma: float = 0.25):
  ref = env.command_manager.get_term("demo_trajectory").sample()[3]
  error = quat_error_magnitude(object_quat_in_tag(env), ref)
  return torch.exp(-torch.square(error) / sigma**2)


def object_position_error_l2(env):
  ref = env.command_manager.get_term("demo_trajectory").sample()[2]
  return torch.linalg.norm(cube_pos_in_tag(env) - ref, dim=-1)


def object_orientation_error(env):
  ref = env.command_manager.get_term("demo_trajectory").sample()[3]
  return quat_error_magnitude(object_quat_in_tag(env), ref)


def joint_reference_tracking(env, sigma: float = 0.30, asset_cfg=SceneEntityCfg("robot")):
  robot: Entity = env.scene[asset_cfg.name]
  ref = env.command_manager.get_term("demo_trajectory").sample()[0]
  error = torch.mean(torch.square(robot.data.joint_pos[:, asset_cfg.joint_ids] - ref), dim=-1)
  return torch.exp(-error / sigma**2)


def residual_action_penalty(env):
  return torch.mean(torch.square(env.action_manager.action), dim=-1)


def residual_action_rate(env):
  return torch.mean(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
  )
