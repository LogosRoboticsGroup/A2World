from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.base.config_multiview_action import get_cosmos_predict2_multiview_state_pred_pipeline
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from cosmos_predict2.models.video2world_multiview_action_state_pred_model import (
    Predict2ActionConditionedMultiviewModelConfig,
    Predict2Video2WorldActionConditionedMultiviewStatePredModel,
)
from imaginaire.constants import get_cosmos_predict2_action_conditioned_checkpoint
from imaginaire.lazy_config import LazyCall as L


A2WORLD_LIBERO_MODEL = {
    "trainer": {"distributed_parallelism": "fsdp"},
    "model": L(Predict2Video2WorldActionConditionedMultiviewStatePredModel)(
        config=Predict2ActionConditionedMultiviewModelConfig(
            pipe_config=get_cosmos_predict2_multiview_state_pred_pipeline(
                model_size="2B", resolution="720", fps=10, views=3, frames=21
            ),
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=get_cosmos_predict2_action_conditioned_checkpoint(model_size="2B", resolution="480", fps=4),
                text_encoder_path="",
            ),
            fsdp_shard_size=-1,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
}


def register_a2world_model() -> None:
    ConfigStore.instance().store(
        group="model",
        package="_global_",
        name="a2world_libero_2b_fsdp",
        node=A2WORLD_LIBERO_MODEL,
    )
