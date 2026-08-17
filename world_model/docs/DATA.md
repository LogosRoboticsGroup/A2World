# LIBERO Data Format

## Processed Layout

```text
libero_processed/
├── clips/
│   ├── agentview_<sample>.mp4
│   └── eye_in_hand_<sample>.mp4
├── metadata/
│   └── agentview_<sample>.npz
├── dataset.json
└── task_info.json            # optional
```

The agent-view clip, wrist-view clip, and metadata file share the same suffix. `a2world-prepare-libero` creates this layout from official LIBERO HDF5 demonstrations.

## Action Representation

A2World uses a shared 14D dual-arm representation. Each arm contributes:

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_open]
```

Single-arm LIBERO actions occupy the first seven values and zero-pad the second arm. The public LIBERO recipe reproduces the training preprocessing:

- translation and rotation commands are multiplied by `0.05`;
- the first-arm gripper command is remapped from LIBERO's `[-1, 1]` convention to `[1, 0]`;
- 20 actions condition each 21-frame video chunk, giving a flattened conditioning width of `280`.

## Metadata Keys

The converter writes:

- `actions`: raw LIBERO servo actions padded to `[T, 2, 7]`;
- `end_position`: absolute end-effector positions `[T, 2, 3]`;
- `end_orientation`: quaternions in SciPy `xyzw` order `[T, 2, 4]`;
- `effector_position`: gripper state `[T, 2, ...]`;
- `start_frame`, `clip_len`, and `fps`.

## Commands

```bash
a2world-prepare-libero \
  --input /path/to/libero/demonstrations \
  --output /path/to/processed/libero

a2world-validate-data /path/to/processed/libero
```

Set `A2WORLD_LIBERO_DATA` to the processed root before training.
