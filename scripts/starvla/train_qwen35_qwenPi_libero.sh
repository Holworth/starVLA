#!/usr/bin/env bash
# starVLA QwenPI + Qwen3.5-4B + LIBERO(libero_goal),8 卡,per-GPU bs=8,30 步。容器内跑。
# 改参数 → 复制本文件改值另存(不要加输入参数)。
set -euo pipefail
cd /code/starVLA
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B \
  --datasets.vla_data.data_root_dir /sv_data \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 30 \
  --trainer.eval_interval 100000 \
  --trainer.save_interval 100000 \
  --trainer.logging_frequency 1 \
  --run_root_dir /tmp/ck --run_id starvla_qwenpi_8gpu_bs8
