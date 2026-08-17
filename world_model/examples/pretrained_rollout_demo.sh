#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <checkpoint.pt> <base-checkpoints> <view0.mp4|png> <view1.mp4|png> <actions.npz> [output.mp4] [extra args...]" >&2
  exit 2
fi

checkpoint=$1
base_checkpoints=$2
view0=$3
view1=$4
actions=$5
output=${6:-outputs/a2world_pretrained_rollout.mp4}

if [[ $# -ge 6 ]]; then
  shift 6
else
  shift 5
fi

a2world-demo \
  --variant pretrained \
  --checkpoint "${checkpoint}" \
  --base-checkpoints "${base_checkpoints}" \
  --input "${view0}" "${view1}" \
  --actions "${actions}" \
  --output "${output}" \
  "$@"
