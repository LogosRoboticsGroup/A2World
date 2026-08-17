#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

require_dir A2WORLD_LIBERO_DATA
require_checkpoint_run A2WORLD_PRETRAIN_CKPT

launch_training \
  cosmos_predict2/configs/base/config_a2world.py \
  a2world_finetune_libero \
  "$@"
