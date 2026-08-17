from hydra.core.config_store import ConfigStore


A2WORLD_FINETUNE_LIBERO = {
    "defaults": [
        {"override /model": "a2world_libero_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "a2world_libero_train"},
        {"override /dataloader_val": "a2world_libero_val"},
        "_self_",
    ],
    "job": {"group": "finetune", "name": "a2world_libero_2b_${now:%Y-%m-%d}_${now:%H-%M-%S}"},
    "model": {
        "config": {
            "train_architecture": "base",
            "pipe_config": {
                "state_t": 6,
                "ema": {"enabled": True},
                "prompt_refiner_config": {"enabled": False},
                "guardrail_config": {"enabled": False},
                "min_num_conditional_frames_per_view": 1,
                "max_num_conditional_frames_per_view": 1,
            },
        }
    },
    "model_parallel": {"context_parallel_size": 1},
    "trainer": {
        "max_iter": 30_000,
        "logging_iter": 10,
        "validation_iter": 2_000,
        "run_validation": True,
        "max_val_iter": 20,
        "grad_accum_iter": 1,
        "distributed_parallelism": "fsdp",
    },
    "checkpoint": {"save_iter": 2_000, "load_training_state": False},
}


ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="a2world_finetune_libero",
    node=A2WORLD_FINETUNE_LIBERO,
)
