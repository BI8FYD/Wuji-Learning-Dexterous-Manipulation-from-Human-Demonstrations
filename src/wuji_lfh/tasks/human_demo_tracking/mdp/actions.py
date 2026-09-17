"""Residual joint-position action around immutable q_ik reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


class ResidualJointPositionEMAAction(JointPositionAction):
  cfg: "ResidualJointPositionEMAActionCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._scale = float(cfg.residual_scale)
    self._alpha = float(cfg.ema_alpha)
    self._command = env.command_manager.get_term(cfg.command_name)
    if isinstance(self._target_ids, slice):
      target_names = tuple(self._entity.joint_names[self._target_ids])
    else:
      target_names = tuple(
        self._entity.joint_names[int(index)] for index in self._target_ids
      )
    if target_names != self._command.reference.joint_names:
      raise ValueError(f"action/reference joint order mismatch: {target_names}")
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower, self._upper = limits[..., 0], limits[..., 1]
    ref = self._command.q
    if torch.any(ref < self._lower[0] - 1.0e-5) or torch.any(
      ref > self._upper[0] + 1.0e-5
    ):
      raise ValueError("reference q exceeds the DemoTrack joint command limits")
    self._filtered_residual = torch.zeros_like(self._command.current_q)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    residual = torch.clamp(actions, -1.0, 1.0) * self._scale
    self._filtered_residual.mul_(1.0 - self._alpha).add_(residual, alpha=self._alpha)
    # Smooth only the learned correction. Filtering the complete target would
    # lag and distort q_ref even when the policy emits an exact zero residual.
    target = self._command.current_q + self._filtered_residual
    self._processed_actions = torch.clamp(target, self._lower, self._upper)

  @property
  def processed_action(self):
    return self._processed_actions

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._filtered_residual[env_ids] = 0.0


@dataclass(kw_only=True)
class ResidualJointPositionEMAActionCfg(JointPositionActionCfg):
  command_name: str = "demo_trajectory"
  residual_scale: float = 0.2
  ema_alpha: float = 0.5

  def build(self, env) -> ResidualJointPositionEMAAction:
    return ResidualJointPositionEMAAction(self, env)
