from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfForcingFrameLayout:
    gap: int
    target_condition_frame_id: int
    extended_history_frame_ids: tuple[int, ...]
    forcing_frame_ids: tuple[int, ...]
    forcing_history_frame_ids: tuple[int, ...]
    training_history_frame_ids: tuple[int, ...]

    @classmethod
    def build(
        cls,
        history_frame_ids: list[int],
        target_frame_ids: list[int],
        gap: int,
    ) -> SelfForcingFrameLayout:
        if not target_frame_ids:
            raise ValueError("target_frame_ids cannot be empty")
        if not 1 <= gap < len(target_frame_ids):
            raise ValueError(f"Invalid self-forcing gap {gap} for {len(target_frame_ids)} target frames")

        start = history_frame_ids[0] if history_frame_ids else 0
        prepend = [max(0, start - gap + index) for index in range(gap)]
        extended_history = prepend + list(history_frame_ids)
        forcing_frames = extended_history[-gap:] + target_frame_ids[: len(target_frame_ids) - gap]
        forcing_history = extended_history[:-gap] or [0]
        training_history = extended_history[gap:] or [0]
        if len(forcing_frames) != len(target_frame_ids):
            raise RuntimeError("Self-forcing rollout length does not match target length")
        if forcing_frames[gap] != target_frame_ids[0]:
            raise RuntimeError(
                f"Target condition frame {target_frame_ids[0]} is not aligned at gap {gap}: "
                f"{forcing_frames}"
            )

        return cls(
            gap=gap,
            target_condition_frame_id=target_frame_ids[0],
            extended_history_frame_ids=tuple(extended_history),
            forcing_frame_ids=tuple(forcing_frames),
            forcing_history_frame_ids=tuple(forcing_history),
            training_history_frame_ids=tuple(training_history),
        )
