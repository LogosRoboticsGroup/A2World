import torch

from cosmos_predict2.data.action_conditioned.self_forcing import SelfForcingFrameLayout
from cosmos_predict2.models.self_forcing import (
    SelfForcingBatch,
    SelfForcingSchedule,
    build_training_batch,
    inject_generated_condition,
    replace_history_frames,
)


def make_batch() -> dict:
    batch_size = 2
    views = 2
    target_frames = 5
    history_frames = 3
    video = torch.zeros(batch_size, 1, views * target_frames, 1, 1, dtype=torch.uint8)
    history = torch.zeros(batch_size, 1, views * history_frames, 1, 1, dtype=torch.uint8)
    return {
        "video": video,
        "forcing_video": video.clone(),
        "video_history": history,
        "history": history.clone(),
        "action": torch.zeros(batch_size, target_frames - 1, 14),
        "history_action": torch.zeros(batch_size, target_frames - 1, 14),
        "delta_training": torch.zeros(batch_size, target_frames, history_frames, 22),
        "delta_forcing": torch.zeros(batch_size, target_frames, history_frames, 22),
        "forcing_predict_frame_ids": torch.tensor([[8, 9, 10, 11, 12], [19, 20, 21, 22, 23]]),
        "sampled_history_frame_ids_training": torch.tensor([[7, 9, 11], [18, 20, 24]]),
        "sampled_history_frame_ids_forcing": torch.tensor([[5, 6, 7], [15, 16, 17]]),
        "target_condition_frame_id": torch.tensor([10, 20]),
        "self_forcing_gap": torch.tensor([2, 1]),
        "latent_view_indices_B_T": torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]]),
        "sample_n_views": torch.tensor([views, views]),
        "fps": torch.tensor([10, 10]),
        "history_video_length": torch.tensor([history_frames, history_frames]),
        "unrelated": torch.tensor([1, 2]),
    }


def test_frame_layout_alignment_with_empty_history() -> None:
    layout = SelfForcingFrameLayout.build([], [10, 11, 12, 13, 14], gap=2)
    assert layout.forcing_frame_ids == (0, 0, 10, 11, 12)
    assert layout.forcing_history_frame_ids == (0,)
    assert layout.training_history_frame_ids == (0,)
    assert layout.forcing_frame_ids[layout.gap] == layout.target_condition_frame_id


def test_frame_layout_alignment_with_real_history() -> None:
    layout = SelfForcingFrameLayout.build([2, 3, 4, 5], [6, 7, 8, 9, 10], gap=2)
    assert layout.extended_history_frame_ids == (0, 1, 2, 3, 4, 5)
    assert layout.forcing_frame_ids == (4, 5, 6, 7, 8)
    assert layout.forcing_history_frame_ids == (0, 1, 2, 3)
    assert layout.training_history_frame_ids == (2, 3, 4, 5)


def test_schedule_warmup() -> None:
    schedule = SelfForcingSchedule(warmup_steps=3, probability=0.4, min_sampling_steps=1, max_sampling_steps=5)
    assert schedule.probability_at(0) == 0
    assert schedule.probability_at(2) == 0
    assert schedule.probability_at(3) == 0.4


def test_batch_validation_and_off_by_one_condition_injection() -> None:
    data = make_batch()
    batch = SelfForcingBatch.from_data_batch(data)
    generated_views = torch.zeros(2, 1, 2, 5, 1, 1, dtype=torch.uint8)
    for batch_index in range(2):
        for view_index in range(2):
            for frame_index in range(5):
                generated_views[batch_index, 0, view_index, frame_index, 0, 0] = (
                    50 * batch_index + 10 * view_index + frame_index
                )
    generated = generated_views.flatten(2, 3)
    video, history = inject_generated_condition(batch, generated)
    video_views = video.view(2, 1, 2, 5, 1, 1)
    assert video_views[0, 0, 0, 0, 0, 0].item() == 2
    assert video_views[0, 0, 1, 0, 0, 0].item() == 12
    assert video_views[1, 0, 0, 0, 0, 0].item() == 51
    assert video_views[1, 0, 1, 0, 0, 0].item() == 61
    assert history.shape == data["video_history"].shape


def test_history_replacement_uses_last_duplicate_prediction() -> None:
    history = torch.zeros(1, 1, 1, 3, 1, 1, dtype=torch.uint8)
    generated = torch.tensor([[[[[[10]], [[20]], [[30]]]]]], dtype=torch.uint8)
    result = replace_history_frames(
        history,
        torch.tensor([[5, 6, 9]]),
        generated,
        torch.tensor([[5, 5, 6]]),
    )
    assert result.flatten().tolist() == [20, 30, 0]


def test_build_training_batch_does_not_mutate_input() -> None:
    data = make_batch()
    original_keys = set(data)
    result = build_training_batch(
        data,
        video=data["video"],
        history=data["video_history"],
        state=data["delta_training"],
    )
    assert set(data) == original_keys
    assert "forcing_video" in data
    assert "forcing_video" not in result
    assert result["unrelated"] is data["unrelated"]
    assert result["history"] is data["video_history"]
