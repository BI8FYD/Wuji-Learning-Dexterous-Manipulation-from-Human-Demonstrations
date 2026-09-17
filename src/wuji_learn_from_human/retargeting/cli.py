"""Export an IK retargeting run as a WujiHand reference trajectory."""

from __future__ import annotations

import argparse

from ..calibration import (
  DEFAULT_TAG_IN_PALM_POSITION_M,
  DEFAULT_TAG_IN_PALM_QUATERNION_WXYZ,
)
from .exporter import export_ik_run


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ik-run", required=True, help="directory containing q_ik.npz")
  parser.add_argument("--output", required=True)
  parser.add_argument("--control-dt", type=float, default=0.05)
  parser.add_argument("--lowpass-hz", type=float, default=8.0)
  parser.add_argument(
    "--robot-to-tag-position", type=float, nargs=3,
    default=DEFAULT_TAG_IN_PALM_POSITION_M,
  )
  parser.add_argument(
    "--robot-to-tag-quaternion-wxyz", type=float, nargs=4,
    default=DEFAULT_TAG_IN_PALM_QUATERNION_WXYZ,
  )
  args = parser.parse_args()
  print(export_ik_run(
    args.ik_run, args.output, control_dt=args.control_dt,
    lowpass_hz=args.lowpass_hz,
    robot_to_tag_position_m=args.robot_to_tag_position,
    robot_to_tag_quaternion_wxyz=args.robot_to_tag_quaternion_wxyz,
  ))


if __name__ == "__main__":
  main()
