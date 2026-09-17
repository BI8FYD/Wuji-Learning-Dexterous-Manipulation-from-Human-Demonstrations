#!/usr/bin/env python3
"""Evaluate in simulation, or use --real for the preserved WujiHand loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wuji_lfh.cli import launch  # noqa: E402

if __name__ == "__main__":
  launch("play")
