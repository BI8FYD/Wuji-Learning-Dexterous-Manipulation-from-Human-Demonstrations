#!/usr/bin/env python3
"""Reuse the task-aware wuji-mjlab ONNX actor and normalizer exporter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wuji_lfh.cli import base_root  # noqa: E402

if __name__ == "__main__":
  base_root()
  from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import main

  main()
