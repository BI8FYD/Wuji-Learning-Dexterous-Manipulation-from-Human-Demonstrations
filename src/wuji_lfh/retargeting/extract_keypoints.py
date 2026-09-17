#!/usr/bin/env python3
"""Extract DexterHand MANO data as replay keypoints for the selected backend.

The shape-specific canonical skeleton preserves wrist/MCP geometry, drives four
straight neutral finger chains with MANO rotations, and uses virtual fingertips.
The output remains pre-normalization and can be passed to ``scripts/retarget/normalize.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "dataset" / "human_demo" / "source.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "retarget" / "Cuboid_00-right.pkl"
DEFAULT_MANO_PATH = ROOT / "dataset" / "models" / "MANO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mano-model", type=Path,
        help="default: MANO_LEFT/RIGHT.pkl selected from input metadata",
    )
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--start", type=int, default=0,
                        help="first source frame, inclusive (default: 0)")
    parser.add_argument("--stop", type=int, default=1333,
                        help="source stop frame, exclusive (default: 1333)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--geometry", choices=("legacy_projected", "native_wuji"),
        default="legacy_projected",
        help="legacy projected skeleton or native MANO geometry for wuji_mano",
    )
    parser.add_argument(
        "--calibration-output", type=Path,
        help="required with --geometry native_wuji; fixed-beta neutral MANO sidecar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"DexterHand NPZ not found: {input_path}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    from wuji_lfh.retargeting.dexterhand import (
        load_dexterhand_npz,
        mano_neutral_wuji_keypoints,
        mano_to_mediapipe_keypoints,
        mano_to_wuji_keypoints,
        mirror_left_mano_landmarks_to_right,
        resolve_mano_model_path,
    )

    # Reuse the canonical replay writer used by the normalization stage.
    from wuji_lfh.retargeting.normalize import save_replay_pkl

    arrays, metadata = load_dexterhand_npz(input_path)
    source_hand = str(metadata.get("hand_side", "left")).lower()
    if source_hand not in {"left", "right"}:
        raise ValueError(f"invalid metadata hand_side: {source_hand!r}")
    mano_path = resolve_mano_model_path(
        args.mano_model if args.mano_model is not None else DEFAULT_MANO_PATH,
        source_hand,
    )
    frame_count = len(arrays["hand_poses"])
    if not 0 <= args.start < args.stop <= frame_count:
        raise ValueError(
            f"invalid frame interval [{args.start}, {args.stop}) for {frame_count} frames"
        )
    fps = float(metadata.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid FPS in DexterHand metadata: {fps}")

    poses = arrays["hand_poses"][args.start:args.stop]
    shapes = arrays["hand_shapes"][args.start:args.stop]
    if args.geometry == "native_wuji":
        if args.calibration_output is None:
            raise ValueError("--calibration-output is required for --geometry native_wuji")
        landmarks, position_tips = mano_to_wuji_keypoints(
            poses, shapes, mano_path, batch_size=args.batch_size,
            is_rhand=source_hand == "right", return_position_tips=True,
        )
        neutral, betas, neutral_position_tips = mano_neutral_wuji_keypoints(
            shapes, mano_path, is_rhand=source_hand == "right", return_position_tips=True,
        )
    else:
        if args.calibration_output is not None:
            raise ValueError("--calibration-output is only valid for --geometry native_wuji")
        landmarks = mano_to_mediapipe_keypoints(
            poses, shapes, mano_path, batch_size=args.batch_size,
            is_rhand=source_hand == "right",
        )
    if args.hand != source_hand:
        landmarks = mirror_left_mano_landmarks_to_right(landmarks)
        if args.geometry == "native_wuji":
            neutral = mirror_left_mano_landmarks_to_right(neutral[None])[0]
            position_tips = position_tips.copy()
            position_tips[..., 0] *= -1.0
            neutral_position_tips = neutral_position_tips.copy()
            neutral_position_tips[..., 0] *= -1.0
    if landmarks.shape != (args.stop - args.start, 21, 3):
        raise RuntimeError(f"unexpected decoded landmark shape: {landmarks.shape}")
    if not np.isfinite(landmarks).all():
        raise ValueError("decoded landmarks contain NaN or infinite values")

    # Output time starts at zero while retaining the source recording's FPS.
    timestamps = np.arange(len(landmarks), dtype=np.float64) / fps
    save_replay_pkl(output_path, landmarks, timestamps, args.hand)
    if args.geometry == "native_wuji":
        calibration_output = args.calibration_output.expanduser().resolve()
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            calibration_output,
            neutral_keypoints_m=neutral.astype(np.float32), betas=betas.astype(np.float32),
            position_tips_m=position_tips.astype(np.float32),
            neutral_position_tips_m=neutral_position_tips.astype(np.float32),
        )

    session = str(metadata.get("mocap_session_name", input_path.stem))
    print(f"[input] {input_path}")
    print(
        f"[extract] session={session}, source_hand={source_hand}, "
        f"target_hand={args.hand}, source_frames=[{args.start}, {args.stop})"
    )
    print(f"[done] wrote {output_path}")
    print(f"[done] records={len(landmarks)}, fps={fps:g}, duration={len(landmarks) / fps:.3f}s")
    print(f"[done] {args.hand}_fingers: float32{landmarks.shape[1:]}, MP-compatible order")
    if args.geometry == "native_wuji":
        print("[geometry] native MANO 21 points; separate 90% distal-axis position targets")
        print(f"[calibration] {args.calibration_output.expanduser().resolve()}")
        print("[next] pass through normalization; do not apply video scale or segment correction")
    else:
        print("[geometry] fixed shape-specific bone lengths; canonical four-finger FK; virtual tips")
        print("[next] normalize with: python scripts/retarget/normalize.py --input " + str(output_path))


if __name__ == "__main__":
    main()
