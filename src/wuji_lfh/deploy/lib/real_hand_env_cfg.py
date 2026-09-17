"""Build a real-hand environment from the training DemoTrack config."""

from __future__ import annotations

from wuji_lfh.tasks.human_demo_tracking.config.wuji_hand.env_cfg import (
  wuji_hand_demo_track_env_cfg,
)
from wuji_lfh.tasks.human_demo_tracking.mdp import observations as train_obs
from wuji_lfh.tasks.human_demo_tracking.reference import load_reference, sha256_file

from . import real_hand_obs


def make_demo_track_real_hand_env_cfg(reference_path: str, policy_config=None):
  reference = load_reference(reference_path)
  if policy_config and "reference_sha256" in policy_config:
    actual_hash = sha256_file(reference.path)
    if actual_hash != policy_config["reference_sha256"]:
      raise ValueError("reference trajectory does not match exported policy metadata")
  cfg = wuji_hand_demo_track_env_cfg(
    reference_path, play=True, num_envs=1, dr_stage=1
  )
  cfg.scene.num_envs = 1
  cfg.events = {}
  cfg.rewards = {}
  cfg.terminations = {}
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.recorders = {}
  cfg.observations = {"policy": cfg.observations["policy"]}
  if policy_config:
    action = cfg.actions["joint_pos"]
    if "residual_scale" in policy_config:
      action.residual_scale = float(policy_config["residual_scale"])
    if "ema_alpha" in policy_config:
      action.ema_alpha = float(policy_config["ema_alpha"])
    if "ctrl_dt" in policy_config:
      cfg.decimation = max(
        1, round(float(policy_config["ctrl_dt"]) / cfg.sim.mujoco.timestep)
      )
  swaps = {
    train_obs.cube_pos_in_tag: real_hand_obs.cube_pos_in_tag_from_zmq,
    train_obs.object_position_error: real_hand_obs.object_position_error_from_zmq,
    train_obs.object_orientation_error_6d: real_hand_obs.object_orientation_error_6d_from_zmq,
  }
  for term in cfg.observations["policy"].terms.values():
    if term.func in swaps:
      term.func = swaps[term.func]
  return cfg
