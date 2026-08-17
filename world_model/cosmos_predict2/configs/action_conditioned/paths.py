import os
from pathlib import Path


def env_path(name: str) -> str:
    value = os.environ.get(name, "")
    return str(Path(value).expanduser()) if value else ""


LIBERO_DATA = env_path("A2WORLD_LIBERO_DATA")
PRETRAIN_CHECKPOINT = env_path("A2WORLD_PRETRAIN_CKPT")
RESUME_CHECKPOINT = env_path("A2WORLD_RESUME_CKPT")
