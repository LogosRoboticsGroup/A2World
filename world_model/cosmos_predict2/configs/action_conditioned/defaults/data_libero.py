from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.configs.action_conditioned.paths import LIBERO_DATA
from cosmos_predict2.data.action_conditioned.action_conditioned_libero_servo_dataset import (
    ActionConditionedServoLiberoDataset,
)
from imaginaire.lazy_config import LazyCall as L


def distributed_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


libero_train_dataset = L(ActionConditionedServoLiberoDataset)(
    data_path=LIBERO_DATA,
    num_frames=21,
    video_size=[256, 256],
    mode="train",
    load_history_video=True,
    use_history_sampling=True,
    history_video_length=20,
    load_t5_embeddings=False,
    load_hand=True,
    use_self_forcing=True,
    servo=True,
)

libero_val_dataset = L(ActionConditionedServoLiberoDataset)(
    data_path=LIBERO_DATA,
    num_frames=21,
    video_size=[256, 256],
    mode="val",
    load_history_video=True,
    use_history_sampling=True,
    history_video_length=20,
    load_t5_embeddings=False,
    load_hand=True,
    use_self_forcing=True,
    servo=True,
)

libero_train_dataloader = L(DataLoader)(
    dataset=libero_train_dataset,
    sampler=L(distributed_sampler)(dataset=libero_train_dataset),
    batch_size=4,
    num_workers=8,
    drop_last=True,
)

libero_val_dataloader = L(DataLoader)(
    dataset=libero_val_dataset,
    sampler=L(distributed_sampler)(dataset=libero_val_dataset),
    batch_size=1,
    num_workers=4,
    drop_last=True,
)


def register_training_and_val_data_action_conditioned() -> None:
    config_store = ConfigStore.instance()
    config_store.store(group="dataloader_train", package="dataloader_train", name="a2world_libero_train", node=libero_train_dataloader)
    config_store.store(group="dataloader_val", package="dataloader_val", name="a2world_libero_val", node=libero_val_dataloader)
