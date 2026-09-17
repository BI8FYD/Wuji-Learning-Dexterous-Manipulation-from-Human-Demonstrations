"""WujiHand wrist-marker calibration defaults.

The transform is explicit input data, rather than a reward or policy parameter.
Measure and replace it for every robot/camera installation before deployment.
"""

from __future__ import annotations

import math

# Wrist marker pose expressed in the palm frame, in metres and wxyz order.
DEFAULT_TAG_IN_PALM_POSITION_M = (0.0262, 0.0, -0.0563)
DEFAULT_TAG_IN_PALM_QUATERNION_WXYZ = (
  math.cos(math.radians(45.0)),
  0.0,
  math.sin(math.radians(45.0)),
  0.0,
)
