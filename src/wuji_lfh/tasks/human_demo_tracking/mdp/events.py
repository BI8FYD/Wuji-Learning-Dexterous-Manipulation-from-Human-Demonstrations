"""Reference-state reset events."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from wuji_lfh.shared import _palm_pose_to_tag_pose


def reset_to_reference(
  env, env_ids, joint_noise_rad: float = 0.01,
  object_position_noise_m: float = 0.001,
):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  command = env.command_manager.get_term("demo_trajectory")
  # Reset events run before CommandManager.reset(). Prepare the phase here and
  # let the command term preserve it when its reset hook runs later.
  command.prepare_reset(env_ids)
  robot, obj = env.scene["robot"], env.scene["object"]
  q_ref, qd_ref, pos_tag, quat_tag, lin_ref, ang_ref = command.sample()
  q = q_ref[env_ids].clone()
  if joint_noise_rad > 0:
    q += torch.empty_like(q).uniform_(-joint_noise_rad, joint_noise_rad)
  limits = robot.data.soft_joint_pos_limits[env_ids]
  q = torch.clamp(q, limits[..., 0], limits[..., 1])
  # The kinematic reference is not dynamically identified and its numerical
  # derivative contains IK discontinuities. Real deployment also starts from
  # rest, so zero reset velocity is the consistent, safe contract.
  robot.write_joint_state_to_sim(q, torch.zeros_like(qd_ref[env_ids]), env_ids=env_ids)

  palm_ids, _ = robot.find_bodies((".*_palm_link",))
  palm_pose = robot.data.body_link_pose_w[env_ids, int(palm_ids[0])]
  tag_pos_w, tag_quat_w = _palm_pose_to_tag_pose(palm_pose[:, :3], palm_pose[:, 3:7])
  pos = pos_tag[env_ids].clone()
  if object_position_noise_m > 0:
    pos += torch.empty_like(pos).uniform_(
      -object_position_noise_m, object_position_noise_m
    )
  pos_w = tag_pos_w + quat_apply(tag_quat_w, pos)
  quat_w = quat_mul(tag_quat_w, quat_tag[env_ids])
  obj.write_root_link_pose_to_sim(torch.cat((pos_w, quat_w), -1), env_ids=env_ids)
  del lin_ref, ang_ref
  obj.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
  )
