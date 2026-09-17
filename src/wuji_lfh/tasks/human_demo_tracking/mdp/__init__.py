# ruff: noqa: F401

from mjlab.envs.mdp import dr as dr
from mjlab.envs.mdp.observations import joint_vel_rel as joint_vel_rel
from mjlab.envs.mdp.terminations import time_out as time_out

from wuji_lfh.shared import (
  finger_self_collision_penalty,
  randomize_body_mass_and_inertia,
  torque_penalty,
)

from .actions import ResidualJointPositionEMAActionCfg
from .commands import DemoTrajectoryCommandCfg
from .events import reset_to_reference
from .observations import (
  cube_pos_in_tag,
  normalized_q,
  normalized_reference,
  normalized_tracking_error,
  object_orientation_error_6d,
  object_position_error,
  object_velocity_in_tag,
  phase_encoding,
  previous_residual_action,
  reference_joint_velocity,
)
from .rewards import (
  joint_reference_tracking,
  object_orientation_error,
  object_orientation_tracking,
  object_position_error_l2,
  object_position_tracking,
  residual_action_penalty,
  residual_action_rate,
)
from .terminations import nonfinite_state, object_reference_deviation, trajectory_end
