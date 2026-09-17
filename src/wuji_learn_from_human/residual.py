"""Framework-independent residual action processing for WujiHand."""

from __future__ import annotations

import numpy as np


class ResidualController:
  """Add a filtered learned correction to an immutable IK reference.

  Filtering is deliberately applied only to the learned residual. Filtering the
  full target would introduce lag even when the policy requests zero correction.
  """

  def __init__(
    self, lower_limits: np.ndarray, upper_limits: np.ndarray,
    *, residual_scale: float = 0.2, ema_alpha: float = 0.5,
  ) -> None:
    self.lower_limits = np.asarray(lower_limits, dtype=np.float32)
    self.upper_limits = np.asarray(upper_limits, dtype=np.float32)
    if self.lower_limits.shape != (20,) or self.upper_limits.shape != (20,):
      raise ValueError("joint limits must each have shape [20]")
    if np.any(self.lower_limits >= self.upper_limits):
      raise ValueError("each lower limit must be smaller than its upper limit")
    if residual_scale <= 0.0 or not np.isfinite(residual_scale):
      raise ValueError("residual_scale must be finite and positive")
    if not 0.0 < ema_alpha <= 1.0:
      raise ValueError("ema_alpha must be in (0, 1]")
    self.residual_scale = float(residual_scale)
    self.ema_alpha = float(ema_alpha)
    self._filtered_residual = np.zeros(20, dtype=np.float32)

  def reset(self) -> None:
    """Clear policy state at an episode or real-hand reset."""
    self._filtered_residual.fill(0.0)

  def step(self, reference_q: np.ndarray, policy_action: np.ndarray) -> np.ndarray:
    """Return a clipped joint target for one control tick."""
    reference_q = np.asarray(reference_q, dtype=np.float32)
    policy_action = np.asarray(policy_action, dtype=np.float32)
    if reference_q.shape != (20,) or policy_action.shape != (20,):
      raise ValueError("reference_q and policy_action must each have shape [20]")
    if not np.isfinite(reference_q).all() or not np.isfinite(policy_action).all():
      raise ValueError("reference_q and policy_action must be finite")
    residual = np.clip(policy_action, -1.0, 1.0) * self.residual_scale
    self._filtered_residual *= 1.0 - self.ema_alpha
    self._filtered_residual += self.ema_alpha * residual
    return np.clip(
      reference_q + self._filtered_residual, self.lower_limits, self.upper_limits
    )
