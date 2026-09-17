# Simulator and WujiHand integration contract

This repository keeps the human-demo, IK-reference, residual-action, and
policy-reference validation logic independent of one simulation stack or robot
SDK. An integration supplies the following pieces.

## Residual-RL environment

At every control tick, the environment must:

1. read `q_ref(t)` from a loaded `ReferenceTrajectory`;
2. form the policy observation only from signals that have a real counterpart;
3. apply `ResidualController.step(q_ref(t), action)` within physical joint
   limits; and
4. reset robot and object state from the same selected reference phase.

The recommended actor signals are encoder position, joint tracking error,
current/future reference joint position, object pose and reference error in the
wrist-tag frame, previous residual action, and phase. Object and joint
velocities are useful critic-only privileged inputs because they need not be
available with equal latency on hardware.

Train with domain randomization only after the reference replay and reset
alignment are stable. Perturb relevant friction, object mass, actuator gains,
pose noise, latency, and observation dropout using measured ranges.

## Policy export

Export the following metadata alongside the policy:

```json
{
  "reference_sha256": "...",
  "reference_schema_version": "wuji-human-demo-v1",
  "reference_joint_names": ["right_finger1_joint1", "..."],
  "control_dt": 0.05,
  "residual_scale": 0.2,
  "ema_alpha": 0.5
}
```

Construct `PolicyReferenceMetadata` from this mapping and run
`validate_policy_reference()` before a physical execution. The check rejects a
policy with a different reference file, schema, joint order, or control period.

## WujiHand playback

A hardware adapter is responsible for translating the validated 20-D joint
target into the WujiHand SDK command. Before enabling it:

- smoothly align the hand to `q_ref[0]` with a bounded trajectory;
- verify that the object observer reports a fresh calibrated wrist-tag-frame
  pose;
- reset the residual filter and policy history after manual alignment; and
- command a safe hold/stop if the observer becomes stale, a limit is exceeded,
  or the operator aborts.

The adapter must never silently substitute a different reference, calibration,
joint order, or control period.
