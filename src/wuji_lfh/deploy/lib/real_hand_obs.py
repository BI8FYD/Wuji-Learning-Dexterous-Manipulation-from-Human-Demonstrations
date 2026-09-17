"""ZMQ-backed replacements for DemoTrack object observations."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul


def cube_pos_in_tag_from_zmq(env, **_kwargs):
  if getattr(env, "_cube_zmq", None) is None:
    return torch.zeros((env.num_envs, 3), device=env.device)
  position, _ = env._cube_zmq.latest()
  return torch.from_numpy(position).float().unsqueeze(0).to(env.device)


def object_position_error_from_zmq(env, offset: int = 0):
  reference = env.command_manager.get_term("demo_trajectory").sample(offset)[2]
  return cube_pos_in_tag_from_zmq(env) - reference


def object_orientation_error_6d_from_zmq(env, offset: int = 0):
  if getattr(env, "_cube_zmq", None) is None:
    current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
  else:
    _, quaternion = env._cube_zmq.latest()
    current = torch.from_numpy(quaternion).float().unsqueeze(0).to(env.device)
  reference = env.command_manager.get_term("demo_trajectory").sample(offset)[3]
  matrix = matrix_from_quat(quat_mul(current, quat_inv(reference)))
  return matrix.reshape(*matrix.shape[:-2], 9)[..., 3:]
