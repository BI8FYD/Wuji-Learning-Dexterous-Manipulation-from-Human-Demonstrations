#!/usr/bin/env python3
"""Normalize DexterHand MediaPipe-style replay PKL data for Wuji Retargeter.

This module owns only preprocessing before ``Retargeter.retarget``: schema
validation, wrist centering, global hand scale, and finger segment correction.
Wrist-frame/MANO-axis transforms stay in the official Retargeter and run once.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "outputs" / "retarget" / "Cuboid_00-right.pkl"

# Exact values from upstream wuji-retargeting's video_mediapipe.py, in metres.
# Each tuple is [MCP->PIP, PIP->DIP, DIP->TIP].
REFERENCE_SEGMENT_LENGTHS = {
    "thumb": (0.0505, 0.0318, 0.0302),
    "index": (0.0418, 0.0243, 0.0223),
    "middle": (0.0489, 0.0289, 0.0227),
    "ring": (0.0422, 0.0274, 0.0227),
    "pinky": (0.0343, 0.0195, 0.0201),
}
FINGER_INDICES = {
    "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def load_replay_pkl(
    path: Path, hand: str, start: int = 0, stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one hand as float32 MediaPipe-style landmarks ``[T,21,3]``."""
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".pkl" or not path.is_file():
        raise FileNotFoundError(f"DexterHand replay PKL not found: {path}")
    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"expected a non-empty list of replay records: {path}")
    frame_stop = len(records) if stop is None else stop
    if not 0 <= start < frame_stop <= len(records):
        raise ValueError(f"invalid frame range [{start}, {frame_stop}) for {len(records)} frames")

    key = f"{hand}_fingers"
    try:
        landmarks = np.stack([
            np.asarray(record[key], dtype=np.float32)
            for record in records[start:frame_stop]
        ])
        timestamps = np.asarray(
            [float(record["t"]) for record in records[start:frame_stop]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid replay record structure in {path}") from error
    if landmarks.shape != (frame_stop - start, 21, 3):
        raise ValueError(f"expected {key} [T,21,3], got {landmarks.shape}")
    if not np.isfinite(landmarks).all() or not np.isfinite(timestamps).all():
        raise ValueError("replay contains NaN or infinite values")
    zero_frames = np.all(np.isclose(landmarks, 0.0), axis=(1, 2))
    if np.any(zero_frames):
        raise ValueError(f"selected hand contains all-zero frames: {np.flatnonzero(zero_frames)[:10].tolist()}")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("replay timestamps must be strictly increasing")
    timestamps -= timestamps[0]
    return landmarks, timestamps, np.arange(start, frame_stop, dtype=np.int64), path.stem


def normalize_landmarks(
    landmarks: np.ndarray, *, wrist_to_middle_mcp_m: float = 0.09,
    correct_segments: bool = True,
) -> np.ndarray:
    """Apply Wuji video-input geometric preprocessing to ``[T,21,3]`` data."""
    keypoints = np.asarray(landmarks, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
        raise ValueError(f"expected landmarks [T,21,3], got {keypoints.shape}")
    if not np.isfinite(keypoints).all():
        raise ValueError("landmarks contain NaN or infinite values")
    if not np.isfinite(wrist_to_middle_mcp_m) or wrist_to_middle_mcp_m <= 0:
        raise ValueError("wrist_to_middle_mcp_m must be positive and finite")

    normalized = keypoints - keypoints[:, 0:1]
    palm_scale = np.linalg.norm(normalized[:, 9], axis=1)
    if np.any(palm_scale < 1e-6):
        raise ValueError("degenerate wrist-to-middle-MCP vector at frames "
                         f"{np.flatnonzero(palm_scale < 1e-6)[:10].tolist()}")
    normalized *= (wrist_to_middle_mcp_m / palm_scale)[:, None, None]

    if correct_segments:
        source = normalized.copy()
        for finger, indices in FINGER_INDICES.items():
            for target_length, (begin, end) in zip(
                REFERENCE_SEGMENT_LENGTHS[finger], zip(indices[:-1], indices[1:])
            ):
                segment = source[:, end] - source[:, begin]
                length = np.linalg.norm(segment, axis=1)
                if np.any(length < 1e-6):
                    raise ValueError(f"degenerate {finger} segment {begin}->{end} at frames "
                                     f"{np.flatnonzero(length < 1e-6)[:10].tolist()}")
                normalized[:, end] = normalized[:, begin] + segment / length[:, None] * target_length
    return normalized.astype(np.float32, copy=False)


def save_replay_pkl(path: Path, landmarks: np.ndarray, timestamps: np.ndarray, hand: str) -> None:
    """Write the exact list-of-records format consumed by MediaPipeReplay."""
    output = path.expanduser().resolve()
    if output.suffix.lower() != ".pkl":
        raise ValueError(f"output must use .pkl: {output}")
    if len(landmarks) != len(timestamps):
        raise ValueError("landmarks and timestamps have different lengths")
    empty = np.zeros((21, 3), dtype=np.float32)
    records = [{
        "t": float(timestamp),
        "left_fingers": frame.copy() if hand == "left" else empty.copy(),
        "right_fingers": frame.copy() if hand == "right" else empty.copy(),
    } for timestamp, frame in zip(timestamps, landmarks)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path,
        help="default: <input-stem>-normalized.pkl next to the input",
    )
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--wrist-to-middle-mcp", type=float, default=0.09)
    parser.add_argument("--correct-segments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--passthrough", action="store_true",
        help="validate and preserve native MANO metric geometry without video preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, timestamps, _, sequence = load_replay_pkl(args.input, args.hand, args.start, args.stop)
    if args.passthrough:
        if args.wrist_to_middle_mcp != 0.09 or args.correct_segments:
            raise ValueError(
                "--passthrough cannot be combined with video scale or segment correction"
            )
        normalized = raw.astype(np.float32, copy=True)
    else:
        normalized = normalize_landmarks(
            raw, wrist_to_middle_mcp_m=args.wrist_to_middle_mcp,
            correct_segments=args.correct_segments,
        )
    output = args.output or args.input.with_name(f"{args.input.stem}-normalized.pkl")
    save_replay_pkl(output, normalized, timestamps, args.hand)
    print(f"[done] {sequence}: {len(normalized)} frames -> {output.resolve()}")
    print(f"[done] {args.hand}_fingers: float32{normalized.shape[1:]}, metres")
    middle_scale = np.linalg.norm(normalized[:, 9] - normalized[:, 0], axis=1)
    label = "native wrist->middle MCP" if args.passthrough else "wrist->middle MCP"
    print(f"[check] {label}: {middle_scale.min():.6f}..{middle_scale.max():.6f} m")


if __name__ == "__main__":
    main()
