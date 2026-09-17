# Wuji: Learning Dexterous Manipulation from Human Demonstrations

Learning dexterous manipulation policies for WujiHand from human demonstrations
via IK retargeting and residual reinforcement learning.

```text
              Human Demo
                  ↓
           IK Retargeting
                  ↓
         Reference Motion
                  ↓
            Residual RL
                  ↓
              WujiHand
```

> **Demo GIF / Video:** coming soon.

`wuji-learn-from-human` is the standalone reference and control contract for
this pipeline. It turns a human demonstration plus its IK solution into an
immutable, calibrated trajectory; trains a residual policy around that motion
in a simulator; and verifies that the policy, reference, observation pipeline,
and physical WujiHand run at the same control contract.

## Why residual RL?

IK retargeting gives the hand a meaningful motion prior but not a reliable
contact policy: geometry, contact dynamics, calibration error, and perception
noise remain. The policy therefore produces only a bounded joint correction:

```text
q_target(t) = q_ref(t) + EMA(residual_scale × clip(policy(obs_t), -1, 1))
```

The exponential moving average is applied to the learned correction only. A
zero-action policy tracks `q_ref` without introducing reference lag.

## Pipeline

1. **Human in-hand demonstration** — capture synchronised hand landmarks and
   object position/orientation.
2. **IK retargeting** — solve the 20 WujiHand joint angles `q_ik` in the
   canonical encoder order.
3. **Reference motion** — export a rate-limited, resampled, wrist-tag-frame
   `.npz` reference with provenance hashes and valid reset frames.
4. **Residual RL** — use the reference as the nominal action and train a
   bounded correction in simulation. The actor should contain only signals
   available on the robot: encoders, reference phase, reference joints, and
   calibrated object pose.
5. **Sim2Real WujiHand** — export the policy with reference hash, joint order,
   control period, residual scale, and filter parameters. Check these fields
   before sending a command to hardware.

The wrist-marker calibration is an input to the reference artifact, not an RL
parameter. Re-measure it for each robot/camera installation.

## Install

```bash
git clone https://github.com/wuji-technology/wuji-learn-from-human.git
cd wuji-learn-from-human
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

## Export a reference motion

The input is an IK run directory containing `q_ik.npz`, `run.json`, the
synchronised source motion, and the retargeting landmark output specified by
the manifest. `q_ik.npz` must contain `q_ik`, `fps`, and the canonical
20-joint `joint_names` array.

```bash
wuji-export-reference \
  --ik-run /path/to/ik-run \
  --output data/reference_v1.npz \
  --control-dt 0.05 \
  --lowpass-hz 8
```

The exporter validates joint order, transforms the object trajectory into the
calibrated `wuji_wrist_tag` frame, applies zero-phase anti-alias filtering,
resamples joint position/object pose, computes velocities, and records hashes
for every source input. The reference format is deliberately simulator
independent.

## Use the residual controller

```python
import numpy as np

from wuji_learn_from_human.residual import ResidualController

controller = ResidualController(
    lower_limits=np.full(20, -1.0),
    upper_limits=np.full(20, 1.0),
    residual_scale=0.2,
    ema_alpha=0.5,
)
joint_target = controller.step(q_ref_at_t, policy_action)
```

At deployment, load the reference with `load_reference()` and call
`validate_policy_reference()` against metadata exported with the policy. See
[the integration contract](docs/INTEGRATION.md) for the expected simulator and
WujiHand adapter responsibilities.

## Reference schema

The `wuji-human-demo-v1` archive includes:

| Field | Meaning |
| --- | --- |
| `q_ref`, `q_ref_velocity` | 20-D WujiHand joint reference and velocity |
| `object_position`, `object_quaternion_wxyz` | Object pose in `wuji_wrist_tag` |
| `object_linear_velocity`, `object_angular_velocity` | Object velocity in the tag frame |
| `valid_reset_mask` | Reference phases allowed for random simulated resets |
| `dt`, `joint_names`, `object_size_m` | Timing, actuation order, and simulated object dimensions |
| calibration and source hashes | Deployment-frame alignment and provenance |

Large demonstrations, checkpoints, policy exports, and videos are ignored by
default. Publish a short demo GIF at `docs/assets/demo.gif` or link a video in
this README when results are ready.

## Safety notes

- Validate open-loop reference replay in simulation before any learned policy.
- Start with a hardware-free driver or digital twin, then confirm calibration
  and fresh object-pose observations before enabling torque/position commands.
- Stop safely when perception becomes stale or reference/policy metadata does
  not match exactly.
- This repository does not replace joint limits, collision checks, watchdogs,
  emergency stop procedures, or the WujiHand hardware safety manual.

## License

Apache-2.0. See [LICENSE](LICENSE).
