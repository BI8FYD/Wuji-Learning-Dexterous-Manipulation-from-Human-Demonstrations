#!/usr/bin/env python3
"""Run a DemoTrack ONNX policy on the physical Wuji Hand.

Execution is deliberately gated: the digital-twin viewer opens first and no
policy action is sent until the operator presses Space after inspecting the
live cube and the phase-zero reference ghost.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from deploy.reorient.lib.hand_driver import MockHandDriver, WujiHandDriver
from deploy.reorient.lib.onnx_policy import ONNXPolicy
from deploy.reorient.lib.zmq_bridge import CubeReceiver

from wuji_lfh.deploy.lib import (
  DemoTrackRealHandEnv,
  make_demo_track_real_hand_env_cfg,
)


def _viz_mj_data(env):
  """Return the MjData buffer refreshed from real encoder positions."""
  return env._mj_data if hasattr(env, "_mj_data") else env.sim.mj_data


def _tag_pose_in_mjworld(env):
  """Convert the wrist-Tag origin to MuJoCo world coordinates for rendering."""
  from deploy.reorient.lib.frame_transform import quat_apply, quat_mul

  from wuji_lfh.calibration import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ

  model = env.sim.mj_model
  data = _viz_mj_data(env)
  palm_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link"
  )
  if palm_id < 0:
    raise RuntimeError("Could not find robot/right_palm_link for viewer sync")
  palm_pos = np.asarray(data.xpos[palm_id], dtype=np.float64)
  palm_quat = np.asarray(data.xquat[palm_id], dtype=np.float64)
  tag_quat = quat_mul(palm_quat, np.asarray(TAG_IN_PALM_QUAT_WXYZ))
  tag_pos = palm_pos + quat_apply(palm_quat, np.asarray(TAG_IN_PALM_POS))
  return tag_pos, tag_quat


def _write_cube_to_viz(env, cube_pos_tag, cube_quat_tag):
  """Insert the vision cube pose into MjData for visualization only."""
  from deploy.reorient.lib.frame_transform import quat_apply, quat_mul

  model = env.sim.mj_model
  data = _viz_mj_data(env)
  cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/cube")
  if cube_body < 0:
    raise RuntimeError("Could not find object/cube for viewer sync")
  joint_id = int(model.body_jntadr[cube_body])
  if joint_id < 0:
    raise RuntimeError("The demo cube has no free joint for viewer sync")
  qpos_adr = int(model.jnt_qposadr[joint_id])

  tag_pos, tag_quat = _tag_pose_in_mjworld(env)
  cube_pos = tag_pos + quat_apply(tag_quat, np.asarray(cube_pos_tag))
  cube_quat = quat_mul(tag_quat, np.asarray(cube_quat_tag))
  data.qpos[qpos_adr:qpos_adr + 3] = cube_pos
  data.qpos[qpos_adr + 3:qpos_adr + 7] = cube_quat
  mujoco.mj_forward(model, data)


def _draw_reference_cube(viewer, env, ref_pos_tag, ref_quat_tag, cube_size_m):
  """Draw the current reference cube as a transparent orange overlay."""
  from deploy.reorient.lib.frame_transform import quat_apply, quat_mul

  tag_pos, tag_quat = _tag_pose_in_mjworld(env)
  ref_pos = tag_pos + quat_apply(tag_quat, np.asarray(ref_pos_tag))
  ref_quat = quat_mul(tag_quat, np.asarray(ref_quat_tag))
  rotation = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, ref_quat)

  viewer.user_scn.ngeom = 0
  geom = viewer.user_scn.geoms[0]
  mujoco.mjv_initGeom(
    geom,
    mujoco.mjtGeom.mjGEOM_BOX.value,
    np.full(3, float(cube_size_m) / 2.0, dtype=np.float64),
    ref_pos,
    rotation,
    np.array([1.0, 0.45, 0.0, 0.35], dtype=np.float32),
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  viewer.user_scn.ngeom = 1


def _pose_errors(observed_pos, observed_quat, ref_pos, ref_quat):
  """Return translation error [m] and shortest rotation error [rad]."""
  pos_error = float(np.linalg.norm(np.asarray(observed_pos) - np.asarray(ref_pos)))
  q_obs = np.asarray(observed_quat, dtype=np.float64)
  q_ref = np.asarray(ref_quat, dtype=np.float64)
  q_obs /= max(np.linalg.norm(q_obs), 1.0e-12)
  q_ref /= max(np.linalg.norm(q_ref), 1.0e-12)
  rot_error = 2.0 * np.arccos(np.clip(abs(np.dot(q_obs, q_ref)), 0.0, 1.0))
  return pos_error, float(rot_error)


def _sync_digital_twin(
  env, hand, cube, viewer, ref_pos, ref_quat, cube_size_m, max_pose_age_s
):
  """Refresh hand/cube/ghost and return pose validity plus alignment errors."""
  env_ids = torch.tensor([0], device=env.device)
  encoder_q = hand.read_encoders()
  env.scene["robot"].write_joint_position_to_sim(
    torch.from_numpy(encoder_q).float().unsqueeze(0).to(env.device), env_ids=env_ids
  )
  env._fast_forward()

  fresh = cube.has_fresh_pose(max_age_s=max_pose_age_s)
  if fresh:
    cube_pos, cube_quat = cube.latest()
    _write_cube_to_viz(env, cube_pos, cube_quat)
    pos_error, rot_error = _pose_errors(cube_pos, cube_quat, ref_pos, ref_quat)
  else:
    pos_error, rot_error = float("inf"), float("inf")

  if viewer.is_running():
    _draw_reference_cube(viewer, env, ref_pos, ref_quat, cube_size_m)
    viewer.sync()
  return fresh, pos_error, rot_error


def _wait_for_operator_start(
  env, hand, cube, viewer, ref_pos, ref_quat, cube_size_m, max_pose_age_s, requests
):
  """Render alignment continuously; return only after a viewer key confirms."""
  print("\n[align] Textured cube: live vision pose. Orange ghost: reference phase 0.")
  print("[align] Align the physical cube by eye, then press SPACE in the viewer.")
  print("[align] There is no numerical pose threshold; operator confirmation decides.")
  print("[align] Press Q or Esc, Ctrl+C, or close the viewer to abort.")

  last_report = 0.0
  while viewer.is_running():
    fresh, pos_error, rot_error = _sync_digital_twin(
      env, hand, cube, viewer, ref_pos, ref_quat, cube_size_m, max_pose_age_s
    )
    now = time.monotonic()
    if now - last_report >= 2.0:
      if fresh:
        print(
          f"[align] pos_err={pos_error * 1000:.1f} mm  "
          f"rot_err={np.degrees(rot_error):.1f} deg  vision=OK"
        )
      else:
        age = cube.fresh_pose_age_s()
        age_text = "none" if age is None else f"{age * 1000:.0f} ms"
        print(f"[align] vision=WAITING/STALE (last valid pose age: {age_text})")
      last_report = now
    try:
      request = requests.pop(0) if requests else None
    except IndexError:
      request = None
    if request in {"quit", "q", "abort"}:
      return False
    if request == "start":
      if fresh:
        return True
      print("[align] Cannot start until a fresh calibrated cube pose arrives.")
    elif request:
      print("[align] Unknown viewer command.")
    time.sleep(0.05)
  print("[align] Viewer closed; policy will not start.")
  return False


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument(
    "--duration", type=float, default=None,
    help="run time in seconds; defaults to exactly one reference trajectory",
  )
  parser.add_argument("--cube-port", type=int, default=5555)
  parser.add_argument(
    "--max-pose-age", type=float, default=0.25,
    help="maximum age in seconds of a calibrated vision pose before hold "
    "(default: 0.25)",
  )
  parser.add_argument("--mock-hand", action="store_true")
  parser.add_argument("--effort-limit", type=float)
  parser.add_argument("--lowpass-cutoff", type=float)
  args = parser.parse_args()
  if args.max_pose_age <= 0:
    parser.error("--max-pose-age must be positive")

  policy = ONNXPolicy(args.ckpt)
  cfg = make_demo_track_real_hand_env_cfg(args.reference, policy.config)
  driver = (
    MockHandDriver()
    if args.mock_hand
    else WujiHandDriver(
      effort_limit=args.effort_limit, lowpass_cutoff=args.lowpass_cutoff
    )
  )
  cube = CubeReceiver(port=args.cube_port)
  viewer = None
  env = None
  viewer_requests: list[str] = []

  def on_viewer_key(keycode: int) -> None:
    """Viewer callback: Space starts; Q/Esc aborts before any policy action."""
    if keycode == 32:  # GLFW_KEY_SPACE
      viewer_requests.append("start")
    elif keycode in (81, 113, 256):  # GLFW_KEY_Q, q, GLFW_KEY_ESCAPE
      viewer_requests.append("quit")

  try:
    with driver as hand:
      env = DemoTrackRealHandEnv(cfg=cfg, hand_driver=hand)
      env._cube_zmq = cube
      policy.validate_against_env(env)
      env.reset()

      command = env.command_manager.get_term("demo_trajectory")
      ref_pos_t, ref_quat_t = command.sample()[2:4]
      ref_pos = ref_pos_t[0].cpu().numpy()
      ref_quat = ref_quat_t[0].cpu().numpy()
      cube_size_m = float(command.reference.object_size_m[0])
      duration = (
        args.duration
        if args.duration is not None
        else command.reference.length * env._ctrl_dt
      )
      print(
        f"[setup] reference={command.reference.length} frames × {env._ctrl_dt:.3f}s "
        f"= {command.reference.length * env._ctrl_dt:.1f}s; run duration={duration:.1f}s"
      )

      env.sim.mj_model.vis.quality.shadowsize = 0
      env.sim.mj_model.vis.quality.offsamples = 0
      viewer = mujoco.viewer.launch_passive(
        env.sim.mj_model,
        _viz_mj_data(env),
        key_callback=on_viewer_key,
        show_left_ui=False,
        show_right_ui=False,
      )
      if not _wait_for_operator_start(
        env,
        hand,
        cube,
        viewer,
        ref_pos,
        ref_quat,
        cube_size_m,
        args.max_pose_age,
        viewer_requests,
      ):
        return 0

      # Clear the three-frame actor history after manual alignment. The hand
      # is already at q_ref[0], so this re-primes policy state at phase zero.
      print("[run] Alignment confirmed; resetting policy history at phase 0.")
      obs, _ = env.reset()
      deadline = time.monotonic() + duration
      while time.monotonic() < deadline:
        if not cube.has_fresh_pose(max_age_s=args.max_pose_age):
          print("[run] Cube vision became stale; safe-stopping the hand.")
          env._safe_stop(
            f"Cube vision stale for more than {args.max_pose_age * 1000:.0f} ms"
          )
          break

        current = env.command_manager.get_term("demo_trajectory")
        ref_pos_t, ref_quat_t = current.sample()[2:4]
        ref_pos = ref_pos_t[0].cpu().numpy()
        ref_quat = ref_quat_t[0].cpu().numpy()
        action = policy(obs["policy"][0].cpu().numpy())
        obs, *_ = env.step(torch.from_numpy(action).float().unsqueeze(0))
        _sync_digital_twin(
          env, hand, cube, viewer, ref_pos, ref_quat, cube_size_m, args.max_pose_age
        )
        if not viewer.is_running():
          print("[run] Viewer closed; safe-stopping the hand.")
          env._safe_stop("Operator closed digital-twin viewer")
          break
  finally:
    if viewer is not None:
      try:
        viewer.close()
      except Exception:
        pass
    if env is not None:
      env.close()
    cube.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
