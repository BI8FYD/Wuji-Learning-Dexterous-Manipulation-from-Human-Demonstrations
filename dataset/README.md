# DexterHand human demonstrations

Store each selected human demonstration as `dataset/<human_demo>/source.npz`.
The repository contains placeholders only; copy or symlink your own dataset.

```text
dataset/
├── <human_demo>/
│   └── source.npz
└── models/
    └── MANO/          # licensed MANO_LEFT.pkl / MANO_RIGHT.pkl, untracked
```

The expected DexterHand NPZ fields are:

- `metadata`: scalar dictionary including `fps`, `hand_side`, `object_class`,
  and `object_size` in metres;
- `hand_poses`: `[T,45]` MANO pose;
- `hand_shapes`: `[T,10]` MANO shape;
- `hand_orientations_axis_angle`, `hand_translations`: `[T,3]`;
- `object_orientations_quat_xyzw`: `[T,4]`;
- `object_translations`: `[T,3]`.

Hand and object arrays must use the same source frames and FPS. Select the
time interval with `--start` and `--stop` in seconds. The current RL binding is
for the right WujiHand and a cuboid with equal side lengths. The retargeting
preprocessing mirrors left-hand demonstrations as in WrenchRetarget.

The script writes raw/normalized landmarks, IK joints, a manifest, and the
WujiHand reference under `outputs/<human_demo>/`. Input datasets and MANO models
are not committed or redistributed.
