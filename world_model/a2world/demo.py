import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPOSITORY = os.environ.get("A2WORLD_MODEL_REPOSITORY", "Fleurrr/A2World-World-Model")
MODEL_FILENAMES = {
    "pretrained": "a2world-pretrained.pt",
    "libero": "a2world-libero.pt",
}
ACTION_FORMATS = {
    "pretrained": "auto",
    "libero": "libero-servo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an A2World checkpoint and run an inference demo")
    parser.add_argument("--variant", choices=MODEL_FILENAMES, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One conditioning image/video per view")
    parser.add_argument("--actions", type=Path, required=True, help="JSON or NPZ action annotation")
    parser.add_argument("--base-checkpoints", type=Path, required=True, help="Directory containing NVIDIA base assets")
    parser.add_argument("--checkpoint", type=Path, help="Use a local A2World checkpoint instead of downloading")
    parser.add_argument("--view-ids", type=int, nargs="+", help="Camera embedding ID for each input view")
    parser.add_argument("--output", type=Path, default=Path("outputs/a2world_demo.mp4"))
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAMES[args.variant],
        )
    )


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args)
    command = [
        sys.executable,
        "-m",
        "a2world.rollout",
        "--checkpoint",
        str(checkpoint),
        "--input",
        *[str(path) for path in args.input],
        "--actions",
        str(args.actions),
        "--action-format",
        ACTION_FORMATS[args.variant],
        "--output",
        str(args.output),
        "--num-sampling-steps",
        str(args.num_sampling_steps),
        "--guidance",
        str(args.guidance),
        "--seed",
        str(args.seed),
        "--variant",
        args.variant,
    ]
    if args.autoregressive:
        command.append("--autoregressive")
    if args.view_ids is not None:
        command.extend(["--view-ids", *[str(view_id) for view_id in args.view_ids]])
    if not args.history:
        command.append("--no-history")
    if args.preflight:
        command.append("--preflight")

    environment = os.environ.copy()
    environment["COSMOS_PREDICT2_ARGS"] = f"--checkpoints {shlex.quote(str(args.base_checkpoints.resolve()))}"
    print("Running:", shlex.join(command))
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
