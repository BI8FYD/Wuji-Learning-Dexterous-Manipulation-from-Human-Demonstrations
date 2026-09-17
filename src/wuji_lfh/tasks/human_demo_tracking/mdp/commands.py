"""Reference-trajectory command for demonstration tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ..reference import load_reference


class DemoTrajectoryCommand(CommandTerm):
  cfg: "DemoTrajectoryCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    reference = load_reference(cfg.reference_path)
    if abs(reference.dt - env.step_dt) > 1.0e-6:
      raise ValueError(
        f"reference dt {reference.dt} must equal policy step_dt {env.step_dt}"
      )
    self.reference = reference
    self.q = torch.as_tensor(reference.q.copy(), device=self.device)
    self.qd = torch.as_tensor(reference.qd.copy(), device=self.device)
    self.object_position = torch.as_tensor(
      reference.object_position.copy(), device=self.device
    )
    self.object_quaternion = torch.as_tensor(
      reference.object_quaternion_wxyz.copy(), device=self.device
    )
    self.object_linear_velocity = torch.as_tensor(
      reference.object_linear_velocity.copy(), device=self.device
    )
    self.object_angular_velocity = torch.as_tensor(
      reference.object_angular_velocity.copy(), device=self.device
    )
    valid_ids = reference.valid_reset_mask.nonzero()[0]
    valid_ids = valid_ids[valid_ids <= reference.length - cfg.min_remaining_steps]
    if len(valid_ids) == 0:
      raise ValueError(
        "reference has no reset frame with "
        f"min_remaining_steps={cfg.min_remaining_steps}"
      )
    self.valid_reset_ids = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
    self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.start_phase = torch.zeros_like(self.phase)
    self._phase_prepared = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.current_q, self.current_object_pose), dim=-1)

  def sample(self, offset: int = 0):
    index = torch.clamp(self.phase + int(offset), max=self.reference.length - 1)
    return (
      self.q[index], self.qd[index], self.object_position[index],
      self.object_quaternion[index], self.object_linear_velocity[index],
      self.object_angular_velocity[index],
    )

  @property
  def current_q(self):
    return self.sample()[0]

  @property
  def current_object_pose(self):
    sample = self.sample()
    return torch.cat((sample[2], sample[3]), dim=-1)

  @property
  def trajectory_end(self):
    return self.phase >= self.reference.length - 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pending = env_ids[~self._phase_prepared[env_ids]]
    if self.cfg.random_start and len(pending) > 0:
      choice = torch.randint(
        0, len(self.valid_reset_ids), (len(pending),), device=self.device
      )
      phase = self.valid_reset_ids[choice]
    elif len(pending) > 0:
      phase = torch.zeros(len(pending), dtype=torch.long, device=self.device)
    if len(pending) > 0:
      self.phase[pending] = phase
      self.start_phase[pending] = phase
    self._phase_prepared[env_ids] = False

  def prepare_reset(self, env_ids: torch.Tensor) -> None:
    """Select reset phase before reset events write the matching physical state."""
    if self.cfg.random_start:
      choice = torch.randint(
        0, len(self.valid_reset_ids), (len(env_ids),), device=self.device
      )
      phase = self.valid_reset_ids[choice]
    else:
      phase = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    self.phase[env_ids] = phase
    self.start_phase[env_ids] = phase
    self._phase_prepared[env_ids] = True

  def _update_command(self) -> None:
    self.phase = torch.clamp(self.phase + 1, max=self.reference.length - 1)

  def _update_metrics(self) -> None:
    self.metrics["phase"] = self.phase.float() / max(self.reference.length - 1, 1)


@dataclass(kw_only=True)
class DemoTrajectoryCommandCfg(CommandTermCfg):
  reference_path: str
  random_start: bool = True
  min_remaining_steps: int = 40
  lookahead_frames: tuple[int, ...] = (0, 4)

  def build(self, env) -> DemoTrajectoryCommand:
    return DemoTrajectoryCommand(self, env)
