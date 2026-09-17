"""Manager-based residual demonstration-tracking environment."""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from wuji_lfh.shared import build_hand_contact_sensors

from . import mdp

HISTORY_LENGTH = 3
LOOKAHEAD_FRAMES = 4  # 200 ms at the canonical 20 Hz control rate.


def _observations(dr_stage: int):
  history = HISTORY_LENGTH
  q_noise = UniformNoiseCfg(n_min=-0.02, n_max=0.02) if dr_stage >= 2 else None
  object_noise = (
    UniformNoiseCfg(n_min=-0.003, n_max=0.003) if dr_stage >= 2 else None
  )
  policy = {
    "q": ObservationTermCfg(
      func=mdp.normalized_q,
      noise=q_noise,
      history_length=history,
    ),
    "q_error": ObservationTermCfg(func=mdp.normalized_tracking_error, history_length=history),
    "q_ref": ObservationTermCfg(
      func=mdp.normalized_reference, params={"offset": 0}, history_length=history
    ),
    "q_ref_future": ObservationTermCfg(
      func=mdp.normalized_reference,
      params={"offset": LOOKAHEAD_FRAMES}, history_length=history,
    ),
    "object_position": ObservationTermCfg(
      func=mdp.cube_pos_in_tag,
      noise=object_noise, history_length=history,
    ),
    "object_position_error": ObservationTermCfg(
      func=mdp.object_position_error, params={"offset": 0}, history_length=history
    ),
    "object_orientation_error": ObservationTermCfg(
      func=mdp.object_orientation_error_6d, params={"offset": 0}, history_length=history
    ),
    "object_position_error_future": ObservationTermCfg(
      func=mdp.object_position_error,
      params={"offset": LOOKAHEAD_FRAMES}, history_length=history,
    ),
    "object_orientation_error_future": ObservationTermCfg(
      func=mdp.object_orientation_error_6d,
      params={"offset": LOOKAHEAD_FRAMES}, history_length=history,
    ),
    "previous_action": ObservationTermCfg(func=mdp.previous_residual_action, history_length=history),
    "phase": ObservationTermCfg(func=mdp.phase_encoding, history_length=history),
  }
  critic = dict(policy)
  critic.update({
    "joint_velocity": ObservationTermCfg(func=mdp.joint_vel_rel),
    "reference_joint_velocity": ObservationTermCfg(func=mdp.reference_joint_velocity),
    "object_velocity": ObservationTermCfg(func=mdp.object_velocity_in_tag),
  })
  return {
    "policy": ObservationGroupCfg(terms=policy, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }


def _events(dr_stage: int):
  events = {
    "reset_reference_state": EventTermCfg(
      func=mdp.reset_to_reference, mode="reset",
      params={"joint_noise_rad": 0.01, "object_position_noise_m": 0.001},
    ),
  }
  if dr_stage >= 2:
    events.update({
      "robot_friction": EventTermCfg(
        mode="startup", func=mdp.dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=(".*palm_.*", ".*finger.*_col")),
          "operation": "scale", "ranges": (0.9, 1.1),
        },
      ),
      "object_friction": EventTermCfg(
        mode="startup", func=mdp.dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg("object", geom_names=(".*",)),
          "operation": "scale", "ranges": (0.9, 1.1),
        },
      ),
      "object_mass": EventTermCfg(
        mode="startup", func=mdp.randomize_body_mass_and_inertia,
        params={
          "asset_cfg": SceneEntityCfg("object", body_names=("cube",)),
          "scale_range": (0.9, 1.1),
        },
      ),
      "pd_gains": EventTermCfg(
        mode="startup", func=mdp.dr.pd_gains,
        params={
          "asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.9, 1.1),
          "kd_range": (0.9, 1.1), "distribution": "log_uniform", "operation": "scale",
        },
      ),
    })
  return events


def make_demo_track_env_cfg(
  reference_path: str | None = None, *, play: bool = False,
  num_envs: int = 4096, dr_stage: int = 1,
) -> ManagerBasedRlEnvCfg:
  reference_path = reference_path or os.environ.get("WUJI_DEMO_TRACK_REFERENCE", "")
  commands: dict[str, CommandTermCfg] = {
    "demo_trajectory": mdp.DemoTrajectoryCommandCfg(
      reference_path=reference_path, random_start=not play,
      resampling_time_range=(1.0e9, 1.0e9),
    )
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.ResidualJointPositionEMAActionCfg(
      entity_name="robot", actuator_names=(".*",), command_name="demo_trajectory",
      residual_scale=0.2, ema_alpha=0.5,
    )
  }
  rewards = {
    "object_position": RewardTermCfg(
      func=mdp.object_position_tracking, weight=3.0, params={"sigma": 0.05}
    ),
    "object_orientation": RewardTermCfg(
      func=mdp.object_orientation_tracking, weight=4.0, params={"sigma": 0.5}
    ),
    # Non-saturating terms preserve a learning signal before precise tracking
    # has emerged; the exponential terms dominate near the reference.
    "object_position_coarse": RewardTermCfg(
      func=mdp.object_position_error_l2, weight=-2.0
    ),
    "object_orientation_coarse": RewardTermCfg(
      func=mdp.object_orientation_error, weight=-0.25
    ),
    "joint_imitation": RewardTermCfg(func=mdp.joint_reference_tracking, weight=0.5),
    "residual": RewardTermCfg(func=mdp.residual_action_penalty, weight=-0.02),
    "residual_rate": RewardTermCfg(func=mdp.residual_action_rate, weight=-0.05),
    "torque": RewardTermCfg(func=mdp.torque_penalty, weight=-0.02),
    "finger_collision": RewardTermCfg(
      func=mdp.finger_self_collision_penalty, weight=-1.0,
      params={"sensor_cfg": SceneEntityCfg("finger_collision")},
    ),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "trajectory_end": TerminationTermCfg(func=mdp.trajectory_end, time_out=True),
    "nonfinite": TerminationTermCfg(func=mdp.nonfinite_state),
    "object_lost": TerminationTermCfg(
      func=mdp.object_reference_deviation, params={"max_distance_m": 0.12}
    ),
  }
  sensors = build_hand_contact_sensors(
    tip_collision_geoms=(".*_finger[1-5]_link4_col",),
    tip_body_names=(".*_finger[1-5]_link4",),
    undesired_object_contact_bodies=(".*_palm_link", ".*_finger[1-5]_link[1-3]"),
  )
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=num_envs,
      env_spacing=0.75, extent=0.8, sensors=sensors,
    ),
    observations=_observations(dr_stage), actions=actions, commands=commands,
    events=_events(dr_stage), rewards=rewards, terminations=terminations,
    curriculum={}, metrics={},
    sim=SimulationCfg(
      nconmax=180, njmax=1500,
      mujoco=MujocoCfg(timestep=0.01, iterations=10, ls_iterations=20),
    ),
    decimation=5, episode_length_s=50.0,
  )
  if play:
    # Interactive replay represents one physical hand and one cube. Multiple
    # parallel environments overlap in the viewer and look like layered cubes.
    cfg.scene.num_envs = 1
    cfg.observations["policy"].enable_corruption = False
  return cfg
