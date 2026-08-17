# A2World World Model

This directory is the standalone world-model release for **A2World**. It contains the multi-view action-conditioned base model, the history-aware A2World-sim extension, LIBERO data conversion and fine-tuning, checkpoint inspection, and open-loop/autoregressive rollout generation.

The internal `cosmos_predict2` and `imaginaire` package names are retained for checkpoint compatibility. Public release utilities live in the smaller `a2world` package.

## Contents

```text
world_model/
├── a2world/                 # Checkpoint, action, data, and rollout utilities
├── cosmos_predict2/         # A2World model and inherited Cosmos runtime
├── imaginaire/              # Training and distributed runtime
├── examples/assets/         # Local pretrained and LIBERO rollout samples
├── examples/inference_demo.sh
├── scripts/finetune_libero.sh
├── docs/DATA.md
└── release/                 # Model card, weight manifest, and weight license
```

## Environment

The reference software environment follows Cosmos-Predict2:

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.6
- flash-attn 2.6.3
- Transformer Engine 1.13.0
- Megatron-Core 0.10.0

Using an existing compatible Cosmos environment:

```bash
conda activate cosmos-predict2
cd world_model
python -m pip uninstall -y cosmos-predict2
python -m pip install -e . --no-deps
a2world-env-check --require-cuda
```

This is the recommended setup when a compatible Cosmos-Predict2 installation and NVIDIA base assets are already available.

Creating a pinned environment:

```bash
conda create -n a2world-world-model python=3.10 -y
conda activate a2world-world-model
python -m pip install --upgrade pip uv
cd world_model
uv sync --extra cu126 --active --inexact
git clone https://github.com/NVIDIA/apex.git /tmp/apex
python -m pip install -v --no-build-isolation /tmp/apex
python -m pip install -e . --no-deps
```

## NVIDIA Base Assets

Log in to Hugging Face, accept the required NVIDIA model terms, and download the tokenizer and initialization assets:

```bash
python scripts/download_base_models.py --checkpoint-dir checkpoints
```

If assets live elsewhere, pass their parent directory through the demo or set:

```bash
export COSMOS_PREDICT2_ARGS="--checkpoints /path/to/checkpoints"
```

## A2World Checkpoints

The release expects two derivative checkpoints:

| File | Use | Download |
|---|---|---|
| `a2world-pretrained.pt` | Multi-view A2World pretrained on heterogeneous robot trajectories | [Hugging Face](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-pretrained.pt) |
| `a2world-libero.pt` | History-aware A2World-sim adapted on LIBERO | [Hugging Face](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-libero.pt) |

Use local weights before upload with `--checkpoint`:

```bash
a2world-checkpoint /path/to/a2world-pretrained.pt /path/to/a2world-libero.pt
a2world-checkpoint --audit-privacy /path/to/a2world-pretrained.pt /path/to/a2world-libero.pt
```

Re-export a trusted historical A2World checkpoint with:

```bash
python -m scripts.export_release_checkpoint \
  --checkpoint /path/to/historical/run/or/model.pt \
  --output /path/to/a2world-libero.pt \
  --source ema \
  --dtype bfloat16 \
  --trust-checkpoint
```

Run `python scripts/create_checkpoint_bundle.py` to convert a bare model file or historical run directory into the run-style layout expected by fine-tuning.

## LIBERO Data

Convert official LIBERO HDF5 demonstrations into paired agent-view/wrist-view clips and metadata:

```bash
a2world-prepare-libero \
  --input /path/to/libero/demonstrations \
  --output /path/to/processed/libero \
  --clip-length 64 \
  --stride 16

a2world-validate-data /path/to/processed/libero
```

See `docs/DATA.md` for the layout and action convention.

## LIBERO Fine-Tuning

Bundle the pretrained model and launch full-parameter A2World-sim adaptation:

```bash
python scripts/create_checkpoint_bundle.py \
  --checkpoint /path/to/a2world-pretrained.pt \
  --output checkpoints/a2world-pretrained

export A2WORLD_PRETRAIN_CKPT=$PWD/checkpoints/a2world-pretrained
export A2WORLD_LIBERO_DATA=/path/to/processed/libero
GPUS=8 ./scripts/finetune_libero.sh
```

Override any LazyConfig value at the end of the command, for example:

```bash
GPUS=8 ./scripts/finetune_libero.sh \
  trainer.max_iter=100 \
  dataloader_train.batch_size=1 \
  checkpoint.save_iter=0
```

## Local Rollout Examples

The repository includes initial observations and action sequences for both released checkpoints under `examples/assets/`.

### Pretrained Checkpoint

#### AgiBot

<img src="https://logosroboticsgroup.github.io/A2World/world_model/examples/assets/pretrained/agibot/initial_observation.jpg" width="768" alt="A2World pretrained AgiBot sample initial observation"/>

```bash
a2world-demo \
  --variant pretrained \
  --input examples/assets/pretrained/agibot/left_wrist.png \
          examples/assets/pretrained/agibot/head.png \
          examples/assets/pretrained/agibot/right_wrist.png \
  --view-ids 0 2 1 \
  --actions examples/assets/pretrained/agibot/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/pretrained_agibot.mp4
```

#### DROID

<img src="https://logosroboticsgroup.github.io/A2World/world_model/examples/assets/pretrained/droid/initial_observation.jpg" width="768" alt="A2World pretrained DROID sample initial observation"/>

```bash
a2world-demo \
  --variant pretrained \
  --input examples/assets/pretrained/droid/exterior_1.png \
          examples/assets/pretrained/droid/wrist.png \
          examples/assets/pretrained/droid/exterior_2.png \
  --view-ids 2 0 3 \
  --actions examples/assets/pretrained/droid/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/pretrained_droid.mp4
```

#### RoboCOIN

<img src="https://logosroboticsgroup.github.io/A2World/world_model/examples/assets/pretrained/robocoin/initial_observation.jpg" width="768" alt="A2World pretrained RoboCOIN sample initial observation"/>

```bash
a2world-demo \
  --variant pretrained \
  --input examples/assets/pretrained/robocoin/high.png \
          examples/assets/pretrained/robocoin/left_wrist.png \
          examples/assets/pretrained/robocoin/right_wrist.png \
  --view-ids 2 0 1 \
  --actions examples/assets/pretrained/robocoin/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/pretrained_robocoin.mp4
```

### LIBERO Checkpoint

The LIBERO checkpoint uses agent-view camera embedding `2` and wrist-view embedding `0`. The included action sequence contains 320 future action steps for a 32-second autoregressive rollout from the two initial observations.

<img src="https://logosroboticsgroup.github.io/A2World/world_model/examples/assets/libero/initial_observation.jpg" width="512" alt="A2World-sim LIBERO sample initial observation"/>

```bash
a2world-demo \
  --variant libero \
  --input examples/assets/libero/agentview.png \
          examples/assets/libero/eye_in_hand.png \
  --actions examples/assets/libero/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/libero_long_rollout.mp4 \
  --autoregressive
```

Source code is Apache-2.0. A2World checkpoints are derivative NVIDIA Cosmos models and are governed by the NVIDIA Open Model License in `release/NVIDIA_OPEN_MODEL_LICENSE.md`.

Use `--view-ids` for other camera layouts. Add `--preflight` to validate checkpoint structure, inputs, view IDs, and actions without allocating the model.
