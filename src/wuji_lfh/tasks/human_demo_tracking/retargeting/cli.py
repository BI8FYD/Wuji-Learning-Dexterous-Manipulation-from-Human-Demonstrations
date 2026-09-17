"""Export a WrenchRetarget q_ik run as a DemoTrack reference."""

from __future__ import annotations

import argparse

from wuji_lfh.calibration import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ

from .exporter import export_wrench_run


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--control-dt", type=float, default=0.05)
  parser.add_argument("--lowpass-hz", type=float, default=8.0)
  parser.add_argument(
    "--robot-to-tag-position", type=float, nargs=3, default=TAG_IN_PALM_POS
  )
  parser.add_argument(
    "--robot-to-tag-quaternion-wxyz", type=float, nargs=4,
    default=TAG_IN_PALM_QUAT_WXYZ,
  )
  args = parser.parse_args()
  result = export_wrench_run(
    args.run_dir, args.output, control_dt=args.control_dt, lowpass_hz=args.lowpass_hz,
    robot_to_tag_position_m=args.robot_to_tag_position,
    robot_to_tag_quaternion_wxyz=args.robot_to_tag_quaternion_wxyz,
  )
  print(result)


if __name__ == "__main__":
  main()
