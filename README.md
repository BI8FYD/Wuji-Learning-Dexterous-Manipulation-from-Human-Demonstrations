# Wuji: Learning Dexterous Manipulation from Human Demonstrations

Learning in-hand manipulation policies for WujiHand from human demonstrations
via IK retargeting and residual reinforcement learning.

**Human Demo → IK Retargeting → Reference Motion → Residual RL → Sim2Real WujiHand**

- **IK retargeting:** map human hand motion to WujiHand joint references.
- **Residual RL:** learn bounded corrections around the reference motion to
  account for contact dynamics and physical constraints.
- **Simulation / deployment:** train and evaluate in simulation, then run the
  exported policy on WujiHand through the existing sim-to-real bridge.

Get started with [Quick Start](#quick-start).

[ Demo GIF / Video — coming soon ]

## Quick Start

### Installation

Requirements: Linux x86_64, an NVIDIA GPU, and [pixi](https://pixi.sh).
Keep working `wuji-mjlab` and `WrenchRetarget` checkouts alongside this repository,
with WrenchRetarget's `external/wuji-retargeting` submodule initialized.

```bash
git clone https://github.com/BI8FYD/Wuji-Learning-In-hand-Manipulation-from-Human-Demonstrations.git wuji-learn-from-human
cd wuji-learn-from-human
pixi install
```

### Retarget → Train → Play

Place the DexterHand Cuboid 02 recording at `dataset/Cuboid_02/source.npz` and
your licensed MANO models under `dataset/models/MANO/`. See
[the dataset format](dataset/README.md). Adjust the time interval to your recording.

```bash
# Retarget the human demonstration and generate the WujiHand reference.
pixi run -e retarget python scripts/retarget/ik_retarget.py \
  --input dataset/Cuboid_02/source.npz \
  --start 95 --stop 115 \
  --mano-model dataset/models/MANO \
  --output-dir outputs/Cuboid_02

# Train the residual policy.
pixi run train \
  --reference outputs/Cuboid_02/reference.npz \
  --agent.logger tensorboard --agent.upload-model False

# Replay a trained checkpoint in simulation.
pixi run play \
  --reference outputs/Cuboid_02/reference.npz \
  --checkpoint-file <path-to-checkpoint.pt>
```

### Sim2Real

After hardware and camera calibration, export the checkpoint and start playback:

```bash
pixi run python scripts/export_onnx.py <path-to-checkpoint.pt>
pixi run -e deploy vision
pixi run -e deploy play-real \
  --reference outputs/Cuboid_02/reference.npz --ckpt <path-to-policy.onnx>
```

Run vision in a separate terminal. See the
[wuji-mjlab deployment guide](https://docs.wuji.tech/docs/en/wuji-mjlab/latest/sim2real/)
for hardware setup. Add `--mock-hand` for the hardware-free driver.

## Human Demonstrations

The human demonstration comes from **DexterHand Cuboid 02**. The selected hand
and object motion is retargeted using the working WrenchRetarget IK implementation
to produce synchronized WujiHand reference trajectories. Dataset files and
licensed MANO models are not bundled.

## Training Environment

The training environment is based on
[wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab), reusing its WujiHand
assets, mjlab/MuJoCo-Warp simulation, PPO backend, and deployment infrastructure.
This repository adds the demonstration-tracking task, IK-to-reference pipeline,
and residual control while preserving the existing RL workflow. The default task
is `WujiHand_HumanDemoTracking`; base training/play overrides remain supported.

## License

Apache-2.0. Dataset, MANO, and upstream dependencies retain their own licenses.
