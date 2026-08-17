#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state

from a2world.actions import load_actions
from a2world.checkpoints import EXPECTED_ACTION_DIM, inspect_checkpoint
from a2world.history import action_path_features, sample_history_indices
from cosmos_predict2.configs.base.config_multiview_action import (
    get_cosmos_predict2_multiview_pipeline,
    get_cosmos_predict2_multiview_state_pred_pipeline,
)
from cosmos_predict2.pipelines.video2world_multiview_action import MultiviewVideo2WorldActionConditionedPipeline
from cosmos_predict2.pipelines.video2world_multiview_action_state_pred import (
    MultiviewVideo2WorldActionConditionedStatePredPipeline,
)
from imaginaire.utils import distributed, misc
from imaginaire.utils.io import save_image_or_video

ACTION_CHUNK_SIZE = 20
PIXEL_FRAMES = 21
LATENT_FRAMES = 6

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an A2World multi-view rollout")
    parser.add_argument("--checkpoint", "--dit-path", dest="checkpoint", required=True)
    parser.add_argument("--variant", choices=("pretrained", "libero"), default="libero")
    parser.add_argument("--input", nargs="+", required=True, help="One image/video path per view")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--output", default="outputs/a2world_rollout.mp4")
    parser.add_argument(
        "--action-format",
        choices=(
            "auto",
            "libero-servo",
            "pose-delta",
            "agibot-pose",
            "droid-state",
            "openx-state",
            "precomputed",
        ),
        default="auto",
    )
    parser.add_argument("--view-ids", type=int, nargs="+", help="Camera embedding IDs; LIBERO defaults to 2 0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--guardrail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def read_frame(path: str, frame_index: int) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read image: {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames = media.read_video(path)
    if frame_index >= len(frames):
        raise IndexError(f"Frame {frame_index} is outside {path} ({len(frames)} frames)")
    return np.asarray(frames[frame_index])


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def resolve_view_ids(args: argparse.Namespace) -> list[int]:
    if args.view_ids is not None:
        if len(args.view_ids) != len(args.input):
            raise ValueError("--view-ids must contain one ID per --input path")
        view_ids = args.view_ids
    elif len(args.input) == 2:
        view_ids = [2, 0]
    else:
        view_ids = list(range(len(args.input)))
    if any(view_id < 0 or view_id > 3 for view_id in view_ids):
        raise ValueError("A2World camera embedding IDs must be in [0, 3]")
    return view_ids


def pack_conditioning_frames(frames: list[np.ndarray]) -> torch.Tensor:
    packed = np.zeros((len(frames) * PIXEL_FRAMES, *frames[0].shape), dtype=np.uint8)
    for view_index, frame in enumerate(frames):
        packed[view_index * PIXEL_FRAMES] = frame
    return torch.from_numpy(packed).permute(3, 0, 1, 2).contiguous()


def setup_pipeline(args: argparse.Namespace):
    checkpoint = inspect_checkpoint(args.checkpoint)
    if checkpoint.variant != args.variant:
        raise ValueError(f"Checkpoint looks like {checkpoint.variant}, but --variant={args.variant}")
    config_getter = (
        get_cosmos_predict2_multiview_state_pred_pipeline
        if args.variant == "libero"
        else get_cosmos_predict2_multiview_pipeline
    )
    config = config_getter(model_size="2B", resolution="720", fps=10, views=3, frames=21)
    config.net.action_dim = EXPECTED_ACTION_DIM
    config.net.use_history = args.variant == "libero" and args.history
    config.guardrail_config.enabled = args.guardrail
    config.prompt_refiner_config.enabled = False

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.num_gpus > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)

    pipeline_type = (
        MultiviewVideo2WorldActionConditionedStatePredPipeline
        if args.variant == "libero"
        else MultiviewVideo2WorldActionConditionedPipeline
    )
    return pipeline_type.from_config(
        config=config,
        dit_path=checkpoint.path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        load_prompt_refiner=False,
    )


def validate_inputs(args: argparse.Namespace) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    checkpoint = inspect_checkpoint(args.checkpoint)
    if checkpoint.variant != args.variant:
        raise ValueError(f"Checkpoint looks like {checkpoint.variant}, but --variant={args.variant}")
    actions = load_actions(args.actions, args.action_format)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected [T, 14] actions, got {actions.shape}")
    if len(actions) < ACTION_CHUNK_SIZE:
        raise ValueError(f"Need at least {ACTION_CHUNK_SIZE} actions, got {len(actions)}")
    frames = [resize_frame(read_frame(path, args.start_frame), args.width, args.height) for path in args.input]
    view_ids = resolve_view_ids(args)
    return actions, frames, view_ids


def generate_rollout(args: argparse.Namespace, pipeline) -> None:
    actions, current_frames, view_ids = validate_inputs(args)
    latent_view_indices = torch.tensor(view_ids, dtype=torch.int64).repeat_interleave(LATENT_FRAMES)
    history_frames = [[frame.copy() for _ in range(args.history_length)] for frame in current_frames]
    history_actions = np.zeros((max(args.history_length - 1, 0), 14), dtype=np.float32)
    chunk_starts = range(0, len(actions), ACTION_CHUNK_SIZE) if args.autoregressive else range(1)
    display_chunks = []

    for chunk_index, start in enumerate(chunk_starts):
        action_chunk = actions[start : start + ACTION_CHUNK_SIZE]
        if len(action_chunk) < ACTION_CHUNK_SIZE:
            break
        conditioning = pack_conditioning_frames(current_frames)
        history_tensor = None
        state = None
        if args.variant == "libero" and args.history:
            indices = sample_history_indices(history_actions, args.history_length)
            selected = [np.stack([view[index] for index in indices]) for view in history_frames]
            history_video = np.concatenate(selected, axis=0)
            history_tensor = torch.from_numpy(history_video).permute(3, 0, 1, 2).unsqueeze(0)
            state = action_path_features(history_actions, action_chunk, indices)[None]

        common = dict(
            video_tensor=conditioning,
            input_path=None,
            actions=action_chunk,
            blacks=None,
            history=history_tensor,
            latent_view_indices=latent_view_indices,
            num_conditional_frames=1,
            guidance=args.guidance,
            n_views=len(current_frames),
            num_sampling_step=args.num_sampling_steps,
            seed=args.seed + chunk_index,
        )
        if args.variant == "libero":
            video = pipeline(state=state, **common)
        else:
            video = pipeline(**common)
        if video is None:
            raise RuntimeError("A2World rollout was rejected by the guardrail")

        per_view = rearrange(video, "b c (v t) h w -> b c v t h w", v=len(current_frames))
        display = rearrange(per_view, "b c v t h w -> b c t h (v w)")
        display_chunks.append(display if chunk_index == 0 else display[:, :, 1:])
        generated = ((per_view[0].permute(1, 2, 3, 4, 0).cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        current_frames = [view[-1] for view in generated]

        if args.variant == "libero" and args.history:
            for view_index, view in enumerate(generated):
                history_frames[view_index].extend(view[1:])
            history_actions = np.concatenate([history_actions, action_chunk], axis=0)

    if not display_chunks:
        raise ValueError("No complete 20-action chunk was available")
    output = torch.cat(display_chunks, dim=2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(output, args.output, fps=10)


def cleanup_distributed() -> None:
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.preflight:
        actions, frames, view_ids = validate_inputs(args)
        print(
            f"preflight ok: variant={args.variant}, views={len(frames)}, view_ids={view_ids}, "
            f"actions={actions.shape}, resolution={args.width}x{args.height}"
        )
        return
    try:
        pipeline = setup_pipeline(args)
        generate_rollout(args, pipeline)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
