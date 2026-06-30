#!/usr/bin/env bash
# 8 卡吞吐扫描: per-GPU bs ∈ {2,4,8,16}(写死)。日志 -> data/starvla_libero/sweep_logs/。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=/home/chenchaox/project/phyai_fd/data/starvla_libero/sweep_logs
mkdir -p "$LOGDIR"
for BS in 2 4 8 16; do
  echo "===== per-GPU bs=$BS ====="
  "$HERE/run_container.sh" bash -lc "cd /code/starVLA && accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
    starVLA/training/train_starvla.py \
    --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
    --framework.name QwenPI --framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B \
    --datasets.vla_data.data_root_dir /sv_data --datasets.vla_data.data_mix libero_goal \
    --datasets.vla_data.per_device_batch_size $BS \
    --trainer.freeze_modules '' --trainer.gradient_accumulation_steps 1 \
    --trainer.max_train_steps 30 --trainer.eval_interval 100000 --trainer.save_interval 100000 \
    --trainer.logging_frequency 1 --run_root_dir /tmp/ck --run_id sweep_bs$BS" \
    > "$LOGDIR/bs$BS.log" 2>&1 && echo "bs=$BS done" || echo "bs=$BS FAILED"
done
echo "logs -> $LOGDIR"
