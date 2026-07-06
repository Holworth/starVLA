#!/usr/bin/env bash
# Sweep NCCL env configurations over the QwenPI collective microbenchmark.
# Runs inside the starvla enroot container. Each config ~40 s.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"

run_cfg() {
  local label="$1"; shift
  echo "=================================================================="
  echo "### CONFIG: ${label}   [$*]"
  ENROOT_MOUNT_HOME=no enroot start --rw \
    --env NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$@" \
    --mount "$(dirname "${PROJ}"):/code" \
    --mount "${PROJ}/scripts:/scripts" \
    starvla bash -lc \
    'torchrun --nproc_per_node=8 --master_port=29517 /scripts/starvla/nccl_microbench.py 2>/dev/null | grep -E "allreduce|allgather"'
}

run_cfg "baseline (auto)"
run_cfg "NVLS"                    --env NCCL_ALGO=NVLS
run_cfg "channels 24"             --env NCCL_MIN_NCHANNELS=24
run_cfg "channels 32"             --env NCCL_MIN_NCHANNELS=32
run_cfg "buffsize 8M"             --env NCCL_BUFFSIZE=8388608
run_cfg "buffsize 16M"            --env NCCL_BUFFSIZE=16777216
run_cfg "CGA 2"                   --env NCCL_CGA_CLUSTER_SIZE=2
run_cfg "NVLS + chan32"           --env NCCL_ALGO=NVLS --env NCCL_MIN_NCHANNELS=32
run_cfg "chan32 + buff16M"        --env NCCL_MIN_NCHANNELS=32 --env NCCL_BUFFSIZE=16777216
echo "=================================================================="
echo "sweep done"
