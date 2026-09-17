"""Deployable actor observations for DemoTrack."""

from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

from wuji_lfh.shared import (
  _palm_pose_to_tag_pose,
  cube_pos_in_tag,
)


def _command(env):
  return env.command_manager.get_term("demo_trajectory")


def normalized_q(env, asset_cfg=SceneEntityCfg("robot")):
  robot: Entity = env.scene[asset_cfg.name]
  q = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (limits[..., 0] + limits[..., 1])
  half = 0.5 * (limits[..., 1] - limits[..., 0])
  return ((q - center) / (half + 1e-6)).clamp(-1.0, 1.0)


def normalized_reference(env, offset: int = 0, asset_cfg=SceneEntityCfg("robot")):
  robot: Entity = env.scene[asset_cfg.name]
  q_ref = _command(env).sample(offset)[0]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (limits[..., 0] + limits[..., 1])
  half = 0.5 * (limits[..., 1] - limits[..., 0])
  return ((q_ref - center) / (half + 1e-6)).clamp(-1.0, 1.0)


def normalized_tracking_error(env, asset_cfg=SceneEntityCfg("robot")):
  return normalized_q(env, asset_cfg) - normalized_reference(env, 0, asset_cfg)


def object_quat_in_tag(
  env, object_cfg=SceneEntityCfg("object"),
  robot_cfg=SceneEntityCfg("robot", body_names=(".*_palm_link",)),
):
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_ids = robot_cfg.body_ids
  if isinstance(palm_ids, slice):
    palm_ids, _ = robot.find_bodies(robot_cfg.body_names)
  palm_pose = robot.data.body_link_pose_w[:, int(palm_ids[0])]
  _, tag_quat = _palm_pose_to_tag_pose(palm_pose[:, :3], palm_pose[:, 3:7])
  return quat_mul(quat_inv(tag_quat), obj.data.root_link_quat_w)


def object_position_error(env, offset: int = 0):
  return cube_pos_in_tag(env) - _command(env).sample(offset)[2]


def object_orientation_error_6d(env, offset: int = 0):
  current = object_quat_in_tag(env)
  reference = _command(env).sample(offset)[3]
  error = quat_mul(current, quat_inv(reference))
  matrix = matrix_from_quat(error)
  return matrix.reshape(*matrix.shape[:-2], 9)[..., 3:]


def previous_residual_action(env):
  return env.action_manager.prev_action


def phase_encoding(env):
  command = _command(env)
  phase = command.phase.float() / max(command.reference.length - 1, 1)
  return torch.stack((torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase), phase), -1)


def reference_joint_velocity(env):
  return _command(env).sample()[1]


def object_velocity_in_tag(
  env, object_cfg=SceneEntityCfg("object"),
  robot_cfg=SceneEntityCfg("robot", body_names=(".*_palm_link",)),
):
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_ids = robot_cfg.body_ids
  if isinstance(palm_ids, slice):
    palm_ids, _ = robot.find_bodies(robot_cfg.body_names)
  palm_pose = robot.data.body_link_pose_w[:, int(palm_ids[0])]
  _, tag_quat = _palm_pose_to_tag_pose(palm_pose[:, :3], palm_pose[:, 3:7])
  linear = quat_apply_inverse(tag_quat, obj.data.root_link_lin_vel_w)
  angular = quat_apply_inverse(tag_quat, obj.data.root_link_ang_vel_w)
  return torch.cat((linear, angular), dim=-1)
