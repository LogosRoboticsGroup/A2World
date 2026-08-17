#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <pretrained|libero> <front.mp4> <wrist.mp4> <actions.npz> <base-checkpoints> [output.mp4]" >&2
  exit 2
fi

variant=$1
front=$2
wrist=$3
actions=$4
base_checkpoints=$5
output=${6:-outputs/a2world_demo.mp4}

a2world-demo \
  --variant "${variant}" \
  --input "${front}" "${wrist}" \
  --actions "${actions}" \
  --base-checkpoints "${base_checkpoints}" \
  --output "${output}" \
  --autoregressive
