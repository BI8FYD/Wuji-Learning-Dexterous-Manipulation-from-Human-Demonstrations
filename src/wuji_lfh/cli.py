"""Thin entry points around the existing wuji-mjlab train/play infrastructure."""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path

DEFAULT_TASK = "WujiHand_HumanDemoTracking"


def base_root() -> Path:
  configured = os.environ.get("WUJI_MJLAB_ROOT")
  if configured:
    root = Path(configured).expanduser().resolve()
  else:
    spec = importlib.util.find_spec("wuji_mjlab")
    root = (
      Path(spec.origin).resolve().parents[2]
      if spec and spec.origin
      else Path(__file__).resolve().parents[2].parent / "wuji-mjlab"
    )
  if not (root / "scripts/train/train_rsl_rl.py").is_file():
    raise RuntimeError(
      "A wuji-mjlab source checkout is required. Install it in editable mode "
      "or set WUJI_MJLAB_ROOT to its checkout directory."
    )
  sys.path[:0] = [str(root), str(root / "src")]
  return root


def launch(mode: str) -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--reference", type=Path)
  if mode == "play":
    parser.add_argument("--real", action="store_true")
  known, remaining = parser.parse_known_args()
  reference = known.reference or os.environ.get("WUJI_DEMO_TRACK_REFERENCE")
  if reference:
    reference = Path(reference).expanduser().resolve()
    if not reference.is_file():
      parser.error(f"reference trajectory not found: {reference}")
    os.environ["WUJI_DEMO_TRACK_REFERENCE"] = str(reference)
  elif not any(arg in ("--help", "-h") for arg in remaining):
    parser.error("provide --reference FILE or WUJI_DEMO_TRACK_REFERENCE")

  root = base_root()
  if getattr(known, "real", False):
    from wuji_lfh.deploy.play_real import main

    sys.argv = [sys.argv[0], *remaining]
    if reference:
      sys.argv.extend(["--reference", str(reference)])
    raise SystemExit(main())

  import wuji_lfh.tasks  # noqa: F401

  if not any(arg == "--task" or arg.startswith("--task=") for arg in remaining):
    remaining.extend(["--task", DEFAULT_TASK])
  script = root / "scripts" / mode / f"{mode}_rsl_rl.py"
  sys.argv = [str(script), *remaining]
  runpy.run_path(str(script), run_name="__main__")
