"""Compatibility boundary to tested wuji-mjlab hand/contact primitives.

The base stores these reusable helpers under its reorient namespace. We reuse
these implementations without copying its orientation-goal command, cage
state, goal rewards, or curriculum into the demonstration-tracking task.
"""

# ruff: noqa: F401
from wuji_mjlab.tasks.reorient.mdp.events import randomize_body_mass_and_inertia
from wuji_mjlab.tasks.reorient.mdp.observations import (
  _palm_pose_to_tag_pose,
  cube_pos_in_tag,
)
from wuji_mjlab.tasks.reorient.mdp.rewards import (
  finger_self_collision_penalty,
  torque_penalty,
)
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_STATE as OBJECT_INIT_STATE,
)
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_ROBOT_INIT_STATE as HAND_INIT_STATE,
)
from wuji_mjlab.tasks.reorient.reorient_terms import (
  build_reorient_sensors as build_hand_contact_sensors,
)
