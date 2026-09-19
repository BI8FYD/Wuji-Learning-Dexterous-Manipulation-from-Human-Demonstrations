#!/usr/bin/env python3
"""DexterHand -> bundled IK retargeting -> WujiHand RL reference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--start", type=float, default=0.0, help="inclusive seconds")
  parser.add_argument("--stop", type=float, required=True, help="exclusive seconds")
  parser.add_argument(
    "--config", type=Path,
    default=ROOT / "configs/retarget/wuji_mano_right.yaml",
  )
  parser.add_argument("--mano-model", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--control-dt", type=float, default=0.05)
  parser.add_argument("--lowpass-hz", type=float, default=8.0)
  parser.add_argument("--robot-to-tag-position", type=float, nargs=3)
  parser.add_argument("--robot-to-tag-quaternion-wxyz", type=float, nargs=4)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  source = args.input.expanduser().resolve()
  config = args.config.expanduser().resolve()
  mano_model = args.mano_model.expanduser().resolve()
  if not source.is_file() or not config.is_file():
    parser.error("--input and --config must be existing files")
  if not mano_model.exists():
    parser.error("--mano-model must point to a licensed MANO model or directory")
  with np.load(source, allow_pickle=True) as archive:
    metadata = archive["metadata"].item()
    count = len(archive["hand_poses"])
  fps = float(metadata["fps"])
  if not np.isfinite(fps) or fps <= 0:
    parser.error("dataset metadata.fps must be positive and finite")
  if not np.isfinite([args.start, args.stop]).all():
    parser.error("--start and --stop must be finite")
  start, stop = int(np.rint(args.start * fps)), int(np.rint(args.stop * fps))
  if not 0 <= start < stop <= count:
    parser.error("selected time interval is outside the dataset")
  backend = (yaml.safe_load(config.read_text()).get("retarget") or {}).get(
    "backend", "current_direction"
  )
  if backend not in {"current_direction", "wuji_mano"}:
    parser.error(f"unsupported retarget backend: {backend}")

  run = (args.output_dir or ROOT / "outputs" / source.parent.name).expanduser().resolve()
  raw = run / "retarget/raw.pkl"
  normalized = run / "retarget/normalized.pkl"
  calibration = run / "retarget/mano_calibration.npz"
  q_ik = run / "q_ik.npz"
  reference = run / "reference.npz"
  if any(path.exists() for path in (raw, normalized, q_ik, reference, run / "run.json")):
    parser.error("output already exists; select a new --output-dir")

  prefix = [sys.executable, "-m"]
  extract = [
    *prefix, "wuji_lfh.retargeting.extract_keypoints",
    "--input", str(source), "--output", str(raw),
    "--start", str(start), "--stop", str(stop),
    "--hand", "right", "--mano-model", str(mano_model),
  ]
  normalize = [
    *prefix, "wuji_lfh.retargeting.normalize",
    "--input", str(raw), "--output", str(normalized), "--hand", "right",
  ]
  ik = [
    *prefix, "wuji_lfh.retargeting.ik_solver",
    "--input", str(normalized), "--output", str(q_ik),
    "--config", str(config), "--hand", "right",
  ]
  if backend == "wuji_mano":
    extract.extend(["--geometry", "native_wuji", "--calibration-output", str(calibration)])
    normalize.extend(["--passthrough", "--no-correct-segments"])
    ik.extend(["--mano-calibration", str(calibration), "--evidence-dir", str(run / "retarget/evidence")])
  env = os.environ.copy()
  env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
  for command in (extract, normalize, ik):
    print("[command] " + " ".join(command), flush=True)
    if not args.dry_run:
      subprocess.run(command, env=env, check=True)
  if args.dry_run:
    print(f"[reference] {reference} (dry run: no files written)")
    return

  from wuji_lfh.tasks.human_demo_tracking.reference import sha256_file
  from wuji_lfh.tasks.human_demo_tracking.retargeting.exporter import export_ik_run

  manifest = {
    "schema": "wuji-lfh-ik-run-v1",
    "source": {"path": str(source), "sha256": sha256_file(source)},
    "frames": {"start": start, "stop": stop},
    "stages": {"retarget": "complete"},
    "stage_outputs": {"retarget": {"raw": {
      "path": "retarget/raw.pkl", "sha256": sha256_file(raw),
    }}},
  }
  (run / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
  overrides = {}
  if args.robot_to_tag_position is not None:
    overrides["robot_to_tag_position_m"] = args.robot_to_tag_position
  if args.robot_to_tag_quaternion_wxyz is not None:
    overrides["robot_to_tag_quaternion_wxyz"] = args.robot_to_tag_quaternion_wxyz
  export_ik_run(
    run, reference, control_dt=args.control_dt, lowpass_hz=args.lowpass_hz,
    **overrides,
  )
  print(f"[reference] {reference}")


if __name__ == "__main__":
  main()
