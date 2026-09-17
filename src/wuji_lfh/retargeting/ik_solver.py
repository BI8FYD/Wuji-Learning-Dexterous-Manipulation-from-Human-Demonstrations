#!/usr/bin/env python3
"""Deterministically export one unfiltered IK solution per keypoint frame."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "outputs/retarget/normalized.pkl"
DEFAULT_OUTPUT = ROOT / "outputs/q_ik.npz"
DEFAULT_CONFIG = ROOT / "configs/retarget/local_segment_vector_right.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--segment-start", type=int, action="append", default=[],
        help="independent segment start frame; may be repeated (frame 0 is implicit)",
    )
    parser.add_argument(
        "--mano-calibration", type=Path,
        help="fixed-beta neutral MANO sidecar; required by retarget.backend=wuji_mano",
    )
    parser.add_argument(
        "--evidence-dir", type=Path,
        help="backend side evidence directory; required by retarget.backend=wuji_mano",
    )
    return parser.parse_args()


def _infer_fps(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        raise ValueError("at least two frames are required to infer FPS")
    deltas = np.diff(timestamps)
    dt = float(np.median(deltas))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("input timestamps must be finite and strictly increasing")
    if not np.allclose(deltas, dt, rtol=1e-3, atol=1e-7):
        raise ValueError(
            "offline export requires a uniformly sampled input; normalize/resample upstream"
        )
    return 1.0 / dt


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    values = np.abs(np.asarray(values, dtype=np.float64))
    if not values.size:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(np.percentile(values, 95)), float(values.max())


def _print_diagnostics(q_ik, fps, joint_names, optimizer, solver_failures,
                       direction_errors, tip_errors_mm) -> None:
    increments = np.diff(q_ik.astype(np.float64), axis=0)
    velocities = increments * fps
    accelerations = np.diff(q_ik.astype(np.float64), n=2, axis=0) * fps ** 2
    dq_mean, dq_p95, dq_max = _summary(increments)
    _, velocity_p95, velocity_max = _summary(velocities)
    _, acceleration_p95, acceleration_max = _summary(accelerations)
    if increments.size:
        jump_frame, jump_joint = np.unravel_index(
            int(np.argmax(np.abs(increments))), increments.shape
        )
        jump_frame += 1
        jump_joint_name = joint_names[jump_joint]
    else:
        jump_frame, jump_joint_name = 0, "n/a"

    lower = np.asarray(
        getattr(optimizer, "optimization_lower_limits", optimizer.robot.joint_limits[:, 0])
    )
    upper = np.asarray(
        getattr(optimizer, "optimization_upper_limits", optimizer.robot.joint_limits[:, 1])
    )
    tolerance = 1e-5
    limit_violations = int(np.sum(
        (q_ik < lower[None] - tolerance) | (q_ik > upper[None] + tolerance)
    ))
    flex_indices = np.asarray(getattr(optimizer, "flexion_q_indices", []), dtype=int)
    reverse_flexion = 0
    if flex_indices.size:
        neutral = np.asarray(optimizer.neutral_qpos)[flex_indices]
        reverse_flexion = int(np.sum(
            q_ik[:, flex_indices] < neutral[None] - tolerance
        ))

    print("[diagnostics] joint increment [rad]")
    print(f"  mean |Δq|: {dq_mean:.8f}")
    print(f"  p95  |Δq|: {dq_p95:.8f}")
    print(f"  max  |Δq|: {dq_max:.8f}")
    print(f"  max jump: frame {jump_frame}, joint {jump_joint_name}")
    print("[diagnostics] joint velocity [rad/s]")
    print(f"  p95/max: {velocity_p95:.6f} / {velocity_max:.6f}")
    print("[diagnostics] joint acceleration [rad/s^2]")
    print(f"  p95/max: {acceleration_p95:.6f} / {acceleration_max:.6f}")
    print("[diagnostics] validity")
    print(f"  non-finite q: {int(np.size(q_ik) - np.isfinite(q_ik).sum())}")
    print(f"  solver failure count: {solver_failures}")
    print(f"  joint limit violation count: {limit_violations}")
    print(f"  reverse-flexion count: {reverse_flexion}")
    if direction_errors:
        directions = np.concatenate(direction_errors)
        tips = np.concatenate(tip_errors_mm)
        print(
            "  bone direction error mean/max [deg]: "
            f"{directions.mean():.6f} / {directions.max():.6f}"
        )
        print(
            "  fingertip position error mean/max [mm]: "
            f"{tips.mean():.6f} / {tips.max():.6f}"
        )


def _load_backend(config_path: Path) -> str:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    backend = str((config.get("retarget") or {}).get("backend", "current_direction"))
    if backend not in {"current_direction", "wuji_mano"}:
        raise ValueError(f"unsupported retarget.backend: {backend!r}")
    return backend


def _print_wuji_diagnostics(q_ik, fps, joint_names, optimizer, diagnostics) -> None:
    failures = int(np.sum(diagnostics["solver_result"] <= 0))
    tip_errors = np.linalg.norm(
        diagnostics["robot_tip_wrist_m"] - diagnostics["target_tip_wrist_m"], axis=-1
    ) * 1000.0
    elapsed = float(diagnostics["elapsed_seconds"])
    print(f"[solve] elapsed={elapsed:.3f}s, throughput={len(q_ik) / elapsed:.1f} frame/s")
    _print_diagnostics(
        q_ik, fps, joint_names, optimizer, failures,
        [diagnostics["direction_error_deg"]], [tip_errors],
    )
    print("[diagnostics] adaptive fingertip wrist-frame error [mm]")
    print(f"  mean/p95/max: {tip_errors.mean():.6f} / "
          f"{np.percentile(tip_errors, 95):.6f} / {tip_errors.max():.6f}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    for path, label in ((input_path, "keypoint replay"), (config_path, "retarget config")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    backend = _load_backend(config_path)
    if backend == "wuji_mano":
        if args.mano_calibration is None or args.evidence_dir is None:
            raise ValueError("wuji_mano requires --mano-calibration and --evidence-dir")
        if not args.mano_calibration.expanduser().resolve().is_file():
            raise FileNotFoundError(f"MANO calibration not found: {args.mano_calibration}")

    # Namespaced imports keep the copied optimizer and preprocessing unambiguous.
    from wuji_lfh.retargeting.normalize import load_replay_pkl
    keypoints, timestamps, _, _ = load_replay_pkl(input_path, args.hand)
    fps = _infer_fps(timestamps)
    frame_count = len(keypoints)
    segment_starts = {0, *args.segment_start}
    if any(frame < 0 or frame >= frame_count for frame in segment_starts):
        raise ValueError(
            f"segment starts must be in [0, {frame_count}), got {sorted(segment_starts)}"
        )

    if backend == "wuji_mano":
        from wuji_lfh.retargeting.wuji_mano_backend import WujiManoBackend

        retargeter = WujiManoBackend(
            config_path, args.hand, args.mano_calibration.expanduser().resolve()
        )
        q_ik, joint_names, diagnostics = retargeter.solve_sequence(
            keypoints, segment_starts=segment_starts
        )
        if q_ik.shape != (frame_count, len(joint_names)):
            raise RuntimeError(f"unexpected adaptive q shape: {q_ik.shape}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, q_ik=q_ik, fps=np.asarray(fps), joint_names=joint_names)
        retargeter.write_evidence(args.evidence_dir.expanduser().resolve(), keypoints, diagnostics)
        print(f"[input] frames={frame_count}, fps={fps:.9g}, hand={args.hand}, backend=wuji_mano")
        print(f"[output] {output_path}")
        print(f"[output] fields=q_ik,fps,joint_names; q_ik.shape={q_ik.shape}")
        print(f"[output] frame invariant: {frame_count} == {len(q_ik)}")
        _print_wuji_diagnostics(q_ik, fps, joint_names, retargeter.optimizer, diagnostics)
        print(f"[evidence] {args.evidence_dir.expanduser().resolve()}")
        return

    from wuji_lfh.retargeting.mechanical_neutral_vector import create_retargeter
    retargeter = create_retargeter(config_path, args.hand)
    optimizer = retargeter.optimizer
    required = ("neutral_qpos", "flexion_q_indices", "kinematic_diagnostics")
    missing = [name for name in required if not hasattr(optimizer, name)]
    if missing:
        raise TypeError(
            f"offline exporter requires MechanicalNeutralVectorOptimizer; missing {missing}"
        )
    joint_names = np.asarray(optimizer.robot.dof_joint_names, dtype=str)
    q_ik = np.empty((frame_count, len(joint_names)), dtype=np.float32)
    solver_failures = 0
    direction_errors: list[np.ndarray] = []
    tip_errors_mm: list[np.ndarray] = []
    start_time = time.perf_counter()

    for frame_index, frame in enumerate(keypoints):
        if frame_index in segment_starts:
            retargeter.reset()
        elif not np.array_equal(optimizer.last_qpos, q_ik[frame_index - 1].astype(np.float64)):
            raise RuntimeError("optimizer warm-start state diverged from previous q_ik")

        _, verbose = retargeter.retarget_verbose(frame, apply_filter=False)
        solution = np.asarray(verbose["qpos_unfiltered"], dtype=np.float32)
        if solution.shape != (len(joint_names),):
            raise RuntimeError(f"unexpected q shape at frame {frame_index}: {solution.shape}")
        q_ik[frame_index] = solution
        result_code = int(optimizer.opt.last_optimize_result())
        if result_code <= 0:
            solver_failures += 1
        diagnostic = optimizer.kinematic_diagnostics(
            solution, verbose["mediapipe_kp"]
        )
        direction_errors.append(diagnostic["direction_error_deg"])
        tip_errors_mm.append(diagnostic["tip_error_cm"] * 10.0)
        if (frame_index + 1) % 5000 == 0:
            print(f"[solve] {frame_index + 1}/{frame_count}")

    if len(q_ik) != frame_count:
        raise RuntimeError("offline frame invariant violated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, q_ik=q_ik, fps=np.asarray(fps), joint_names=joint_names)
    elapsed = time.perf_counter() - start_time
    print(f"[input] frames={frame_count}, fps={fps:.9g}, hand={args.hand}")
    print(f"[output] {output_path}")
    print(f"[output] fields=q_ik,fps,joint_names; q_ik.shape={q_ik.shape}")
    print(f"[output] frame invariant: {frame_count} == {len(q_ik)}")
    print(f"[solve] elapsed={elapsed:.3f}s, throughput={frame_count / elapsed:.1f} frame/s")
    _print_diagnostics(
        q_ik, fps, joint_names, optimizer, solver_failures,
        direction_errors, tip_errors_mm,
    )


if __name__ == "__main__":
    main()
