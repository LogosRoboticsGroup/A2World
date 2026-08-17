#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch

from a2world.checkpoints import audit_checkpoint_privacy, resolve_checkpoint_path

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

TRAINING_METADATA_KEYS = frozenset(
    {
        "accum_image_sample_counter",
        "accum_iteration",
        "accum_train_in_hours",
        "accum_video_sample_counter",
    }
)
RELEASE_MTIME = 315532800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a safe A2World inference checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Historical model file or run directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt file")
    parser.add_argument("--source", choices=("regular", "ema"), default="ema")
    parser.add_argument("--dtype", choices=("preserve", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow loading a trusted historical pickle")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trust_checkpoint:
        raise SystemExit("Historical Cosmos checkpoints require the explicit --trust-checkpoint flag")

    source_path = resolve_checkpoint_path(args.checkpoint)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
    prefix = "net_ema." if args.source == "ema" else "net."
    target_dtype = DTYPES.get(args.dtype)
    exported = {}
    skipped_non_tensors = 0
    skipped_training_metadata = 0

    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        relative_key = key[len(prefix):]
        if relative_key in TRAINING_METADATA_KEYS or relative_key.endswith("._extra_state"):
            skipped_training_metadata += 1
            continue
        if not isinstance(value, torch.Tensor):
            skipped_non_tensors += 1
            continue
        if target_dtype is not None and value.is_floating_point():
            value = value.to(target_dtype)
        exported[f"net.{relative_key}"] = value.contiguous()

    if not exported:
        raise ValueError(f"No tensors found with prefix {prefix!r} in {source_path}")
    torch.save(exported, output_path)
    audit_checkpoint_privacy(output_path)
    os.utime(output_path, (RELEASE_MTIME, RELEASE_MTIME))
    tensor_bytes = sum(value.numel() * value.element_size() for value in exported.values())
    print(
        f"exported {len(exported)} tensors ({tensor_bytes / 2**30:.2f} GiB) from {prefix} "
        f"to {output_path}; skipped {skipped_non_tensors} non-tensor entries and "
        f"{skipped_training_metadata} training-metadata entries"
    )


if __name__ == "__main__":
    main()
