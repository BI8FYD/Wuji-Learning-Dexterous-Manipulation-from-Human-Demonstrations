"""Existing palm-to-wrist-tag calibration defaults, in metres and wxyz.

Override these when exporting for a different physical wrist-marker mount.
"""

import math

TAG_IN_PALM_POS = (0.0262, 0.0, -0.0563)
TAG_IN_PALM_QUAT_WXYZ = (
  math.cos(math.radians(45.0)), 0.0, math.sin(math.radians(45.0)), 0.0,
)
