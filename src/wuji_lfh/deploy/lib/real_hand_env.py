"""DemoTrack specialization of the shared real-hand control loop."""

from __future__ import annotations

import time

import numpy as np
from deploy.reorient.lib.real_hand_env import RealHandEnv


class DemoTrackRealHandEnv(RealHandEnv):
  REFERENCE_ALIGNMENT_SEC = 2.0

  def _reset_hand(self) -> np.ndarray:
    """Smoothly align hardware with q_ref[0], matching simulation reset."""
    current = self._hand.read_encoders().astype(np.float64)
    command = self.command_manager.get_term("demo_trajectory")
    target = command.q[0].cpu().numpy().astype(np.float64)
    target = self._clamp_to_joint_limits(target)
    steps = max(1, round(self.REFERENCE_ALIGNMENT_SEC / self._ctrl_dt))
    for index in range(steps):
      phase = (index + 1) / steps
      smooth = phase * phase * (3.0 - 2.0 * phase)
      self._hand.write_target(current + smooth * (target - current))
      time.sleep(self._ctrl_dt)
    return self._hand.read_encoders()

  def _update_external_command(self, goal_quat: np.ndarray) -> None:
    del goal_quat
    command = self.command_manager.get_term("demo_trajectory")
    command.phase += (~command.trajectory_end).long()
