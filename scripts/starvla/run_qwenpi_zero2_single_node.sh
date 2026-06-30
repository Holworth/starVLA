#!/usr/bin/env bash
# Single-node starVLA QwenPI full-unfreeze run with DeepSpeed ZeRO-2.
#
# Defaults match the measured throughput setup:
#   8x H20, per-GPU batch 8, LIBERO libero_goal, 30 training steps.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${HERE}/../.." && pwd)"

GPUS="${GPUS:-8}"
BS="${BS:-8}"
MAX_STEPS="${MAX_STEPS:-30}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_ID="${RUN_ID:-qwenpi_zero2_1node_bs${BS}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJ}/data/starvla_libero/run_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"

mkdir -p "${LOG_DIR}"

echo "run_id=${RUN_ID}"
echo "gpus=${GPUS} per_gpu_batch=${BS} max_steps=${MAX_STEPS} grad_accum=${GRAD_ACCUM}"
echo "log=${LOG_FILE}"

"${HERE}/run_container.sh" bash -lc "cd /code/starVLA && accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes ${GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B \
  --datasets.vla_data.data_root_dir /sv_data \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size ${BS} \
  --trainer.freeze_modules '' \
  --trainer.gradient_accumulation_steps ${GRAD_ACCUM} \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.eval_interval 100000 \
  --trainer.save_interval 100000 \
  --trainer.logging_frequency 1 \
  --run_root_dir /tmp/ck \
  --run_id ${RUN_ID}" 2>&1 | tee "${LOG_FILE}"
