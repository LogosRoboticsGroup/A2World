from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


SELF_FORCING_KEYS = frozenset(
    {
        "delta_forcing",
        "delta_training",
        "forcing_predict_frame_ids",
        "forcing_video",
        "forcing_video_gap",
        "history",
        "history_action",
        "history_video_length",
        "sampled_history_frame_ids_forcing",
        "sampled_history_frame_ids_training",
        "self_forcing_gap",
        "target_condition_frame_id",
        "video_history",
    }
)


@dataclass(frozen=True)
class SelfForcingSchedule:
    warmup_steps: int = 2000
    probability: float = 0.4
    min_sampling_steps: int = 1
    max_sampling_steps: int = 5

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("self-forcing warmup_steps must be non-negative")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("self-forcing probability must be in [0, 1]")
        if self.min_sampling_steps < 1:
            raise ValueError("self-forcing min_sampling_steps must be positive")
        if self.max_sampling_steps < self.min_sampling_steps:
            raise ValueError("self-forcing max_sampling_steps must be >= min_sampling_steps")

    def probability_at(self, iteration: int) -> float:
        return 0.0 if iteration < self.warmup_steps else self.probability


def synchronized_bernoulli(probability: float, device: torch.device) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    value = torch.empty((), device=device, dtype=torch.float32)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        value.uniform_()
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(value, src=0)
    return bool(value.item() < probability)


def synchronized_randint(low: int, high: int, device: torch.device) -> int:
    if low > high:
        raise ValueError(f"Invalid randint range: {low} > {high}")
    if low == high:
        return low
    value = torch.empty((), device=device, dtype=torch.int64)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        value.random_(low, high + 1)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(value, src=0)
    return int(value.item())


def _uniform_scalar(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        flattened = value.detach().reshape(-1)
        if flattened.numel() == 0:
            raise ValueError(f"{name} cannot be empty")
        if not torch.equal(flattened, flattened[:1].expand_as(flattened)):
            raise ValueError(f"{name} must be uniform within a batch, got {flattened.tolist()}")
        return int(flattened[0].item())
    return int(value)


@dataclass(frozen=True)
class SelfForcingBatch:
    target_video: torch.Tensor
    forcing_video: torch.Tensor
    training_history: torch.Tensor
    forcing_history: torch.Tensor
    target_actions: torch.Tensor
    forcing_actions: torch.Tensor
    training_state: torch.Tensor
    forcing_state: torch.Tensor
    forcing_frame_ids: torch.Tensor
    training_history_frame_ids: torch.Tensor
    forcing_history_frame_ids: torch.Tensor
    target_condition_frame_ids: torch.Tensor
    gaps: torch.Tensor
    latent_view_indices: torch.Tensor
    n_views: int
    fps: int
    target_frames: int
    history_frames: int

    @classmethod
    def from_data_batch(cls, data_batch: dict[str, Any]) -> SelfForcingBatch:
        required = {
            "action",
            "delta_forcing",
            "delta_training",
            "forcing_predict_frame_ids",
            "forcing_video",
            "fps",
            "history",
            "history_action",
            "latent_view_indices_B_T",
            "sample_n_views",
            "sampled_history_frame_ids_forcing",
            "sampled_history_frame_ids_training",
            "target_condition_frame_id",
            "video",
            "video_history",
        }
        missing = sorted(required.difference(data_batch))
        if missing:
            raise KeyError(f"Missing self-forcing batch keys: {missing}")

        n_views = _uniform_scalar(data_batch["sample_n_views"], "sample_n_views")
        fps = _uniform_scalar(data_batch["fps"], "fps")
        gap_value = data_batch.get("self_forcing_gap")
        if gap_value is None:
            legacy_gap = data_batch.get("forcing_video_gap")
            if legacy_gap is None:
                raise KeyError("Missing self_forcing_gap")
            gap_value = legacy_gap + 1

        target_video = data_batch["video"]
        forcing_video = data_batch["forcing_video"]
        training_history = data_batch["video_history"]
        forcing_history = data_batch["history"]
        if target_video.ndim != 5 or forcing_video.shape != target_video.shape:
            raise ValueError(
                f"Expected matching target/forcing videos [B,C,T,H,W], got "
                f"{tuple(target_video.shape)} and {tuple(forcing_video.shape)}"
            )
        if n_views < 1 or target_video.shape[2] % n_views:
            raise ValueError(f"Video time dimension {target_video.shape[2]} is incompatible with {n_views} views")
        if training_history.ndim != 5 or forcing_history.shape != training_history.shape:
            raise ValueError(
                f"Expected matching histories [B,C,T,H,W], got "
                f"{tuple(training_history.shape)} and {tuple(forcing_history.shape)}"
            )
        if training_history.shape[2] % n_views:
            raise ValueError(
                f"History time dimension {training_history.shape[2]} is incompatible with {n_views} views"
            )

        batch_size = target_video.shape[0]
        target_frames = target_video.shape[2] // n_views
        history_frames = training_history.shape[2] // n_views
        gaps = torch.as_tensor(gap_value, device=target_video.device, dtype=torch.long).reshape(-1)
        if gaps.numel() == 1:
            gaps = gaps.expand(batch_size)
        if gaps.shape != (batch_size,) or bool(((gaps < 1) | (gaps >= target_frames)).any()):
            raise ValueError(f"Invalid self-forcing gaps {gaps.tolist()} for {target_frames} target frames")

        target_actions = data_batch["action"]
        forcing_actions = data_batch["history_action"]
        expected_action_shape = (batch_size, target_frames - 1, target_actions.shape[-1])
        if target_actions.shape != expected_action_shape or forcing_actions.shape != expected_action_shape:
            raise ValueError(
                f"Expected action tensors {expected_action_shape}, got "
                f"{tuple(target_actions.shape)} and {tuple(forcing_actions.shape)}"
            )

        training_state = data_batch["delta_training"]
        forcing_state = data_batch["delta_forcing"]
        expected_state_prefix = (batch_size, target_frames, history_frames)
        if training_state.shape[:3] != expected_state_prefix or forcing_state.shape != training_state.shape:
            raise ValueError(
                f"Expected matching state tensors beginning with {expected_state_prefix}, got "
                f"{tuple(training_state.shape)} and {tuple(forcing_state.shape)}"
            )

        forcing_frame_ids = data_batch["forcing_predict_frame_ids"].long()
        training_history_frame_ids = data_batch["sampled_history_frame_ids_training"].long()
        forcing_history_frame_ids = data_batch["sampled_history_frame_ids_forcing"].long()
        target_condition_frame_ids = torch.as_tensor(
            data_batch["target_condition_frame_id"], device=target_video.device, dtype=torch.long
        ).reshape(-1)
        if target_condition_frame_ids.numel() == 1:
            target_condition_frame_ids = target_condition_frame_ids.expand(batch_size)
        if target_condition_frame_ids.shape != (batch_size,):
            raise ValueError(
                f"Unexpected target condition frame IDs shape: {tuple(target_condition_frame_ids.shape)}"
            )
        if forcing_frame_ids.shape != (batch_size, target_frames):
            raise ValueError(f"Unexpected forcing frame IDs shape: {tuple(forcing_frame_ids.shape)}")
        if training_history_frame_ids.shape != (batch_size, history_frames):
            raise ValueError(
                f"Unexpected training history IDs shape: {tuple(training_history_frame_ids.shape)}"
            )
        if forcing_history_frame_ids.shape != (batch_size, history_frames):
            raise ValueError(
                f"Unexpected forcing history IDs shape: {tuple(forcing_history_frame_ids.shape)}"
            )
        condition_ids = forcing_frame_ids.gather(1, gaps[:, None]).squeeze(1)
        if not torch.equal(condition_ids, target_condition_frame_ids):
            raise ValueError(
                f"Self-forcing alignment mismatch: generated IDs {condition_ids.tolist()} do not match "
                f"target IDs {target_condition_frame_ids.tolist()} at gaps {gaps.tolist()}"
            )

        return cls(
            target_video=target_video,
            forcing_video=forcing_video,
            training_history=training_history,
            forcing_history=forcing_history,
            target_actions=target_actions,
            forcing_actions=forcing_actions,
            training_state=training_state,
            forcing_state=forcing_state,
            forcing_frame_ids=forcing_frame_ids,
            training_history_frame_ids=training_history_frame_ids,
            forcing_history_frame_ids=forcing_history_frame_ids,
            target_condition_frame_ids=target_condition_frame_ids,
            gaps=gaps,
            latent_view_indices=data_batch["latent_view_indices_B_T"],
            n_views=n_views,
            fps=fps,
            target_frames=target_frames,
            history_frames=history_frames,
        )


def build_training_batch(
    data_batch: dict[str, Any],
    *,
    video: torch.Tensor,
    history: torch.Tensor,
    state: torch.Tensor,
) -> dict[str, Any]:
    result = {key: value for key, value in data_batch.items() if key not in SELF_FORCING_KEYS}
    result["video"] = video
    result["history"] = history
    result["state"] = state
    return result


def replace_history_frames(
    history: torch.Tensor,
    history_frame_ids: torch.Tensor,
    generated_frames: torch.Tensor,
    generated_frame_ids: torch.Tensor,
) -> torch.Tensor:
    if history.ndim != 6 or generated_frames.ndim != 6:
        raise ValueError("history and generated_frames must have shape [B,C,V,T,H,W]")
    batch_size, channels, views, history_length, height, width = history.shape
    if generated_frames.shape[:3] != (batch_size, channels, views):
        raise ValueError("Generated frames do not match history batch/channel/view dimensions")
    generated_length = generated_frames.shape[3]
    if history_frame_ids.shape != (batch_size, history_length):
        raise ValueError(f"Unexpected history_frame_ids shape: {tuple(history_frame_ids.shape)}")
    if generated_frame_ids.shape != (batch_size, generated_length):
        raise ValueError(f"Unexpected generated_frame_ids shape: {tuple(generated_frame_ids.shape)}")

    matches = history_frame_ids[:, :, None] == generated_frame_ids[:, None, :]
    generated_positions = torch.arange(generated_length, device=history.device).view(1, 1, generated_length)
    matched_positions = torch.where(matches, generated_positions, -1).max(dim=-1).values
    valid = matched_positions >= 0
    safe_positions = matched_positions.clamp_min(0)
    gather_index = safe_positions[:, None, None, :, None, None].expand(
        batch_size, channels, views, history_length, height, width
    )
    selected = torch.gather(generated_frames, dim=3, index=gather_index)
    return torch.where(valid[:, None, None, :, None, None], selected, history)


def inject_generated_condition(
    batch: SelfForcingBatch,
    generated_video: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, total_frames, height, width = generated_video.shape
    expected_total_frames = batch.n_views * batch.target_frames
    if generated_video.shape != batch.target_video.shape or total_frames != expected_total_frames:
        raise ValueError(
            f"Generated video shape {tuple(generated_video.shape)} does not match target "
            f"{tuple(batch.target_video.shape)}"
        )

    generated_views = generated_video.view(
        batch_size, channels, batch.n_views, batch.target_frames, height, width
    )
    target_views = batch.target_video.clone().view(
        batch_size, channels, batch.n_views, batch.target_frames, height, width
    )
    condition_index = batch.gaps[:, None, None, None, None, None].expand(
        batch_size, channels, batch.n_views, 1, height, width
    )
    generated_condition = torch.gather(generated_views, dim=3, index=condition_index).squeeze(3)
    target_views[:, :, :, 0] = generated_condition

    history_views = batch.training_history.view(
        batch_size, channels, batch.n_views, batch.history_frames, height, width
    )
    updated_history = replace_history_frames(
        history_views,
        batch.training_history_frame_ids,
        generated_views[:, :, :, 1:],
        batch.forcing_frame_ids[:, 1:],
    )
    return target_views.flatten(2, 3), updated_history.flatten(2, 3)
