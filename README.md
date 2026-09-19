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

## Quick Start

### Installation

Requirements: Linux x86_64, an NVIDIA GPU, and [pixi](https://pixi.sh).
Keep a working `wuji-mjlab` checkout alongside this repository.
The retargeting implementation and kinematic models are bundled here;
no separate retargeting project is required.

```bash
git clone https://github.com/BI8FYD/Wuji-Learning-In-hand-Manipulation-from-Human-Demonstrations.git wuji-learn-from-human
cd wuji-learn-from-human
pixi install
```

### Retarget → Train → Play

Place the DexterHand Cuboid 02 recording at `dataset/Cuboid_02-fps_60-right.npz` and
your licensed MANO models under `dataset/MANO models/`. See
[the dataset format](dataset/README.md). Adjust the time interval to your recording.

```bash
# Retarget the human demonstration and generate the WujiHand reference.
pixi run -e retarget python scripts/retarget/ik_retarget.py \
  --input dataset/Cuboid_02-fps_60-right.npz \
  --start 95 --stop 115 \
  --mano-model "dataset/MANO models/" \
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

## License

Apache-2.0. Bundled retargeting code and kinematic models retain their MIT
licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
Dataset, MANO, and upstream dependencies retain their own licenses.
