#!/usr/bin/env bash
# nsys profile of starVLA QwenPI + Qwen3.5-4B,容器内原生跑 nsys(挂 host 的 nsys 2024.4.2,
# 避开 host 包 enroot 的 nvidia-hook 冲突)。单卡 + 冻结 backbone(全参单卡 OOM),bs=4,
# 跑 14 步,靠 nvtx_patch(TRIGGER=cuda)在 step8 cudaProfilerStart、step12 stop,只抓稳态。
# NVTX: dataloader / train_step_i / model_forward / profile_window。
set -euo pipefail
PROJ=/home/chenchaox/project/phyai_fd
HOST_NSYS=/opt/nvidia/nsight-systems/2024.4.2
mkdir -p "$PROJ/profiles"

ENROOT_MOUNT_HOME=no enroot start --rw \
  --env NVIDIA_VISIBLE_DEVICES=0 --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE=disabled --env PYTHONWARNINGS=ignore \
  --env HF_HOME=/model/huggingface --env HF_HUB_CACHE=/model/huggingface/hub \
  --env PYTHONPATH=/scripts/starvla \
  --env STARVLA_PROFILE_TRIGGER=cuda \
  --env STARVLA_PROFILE_START_STEP=8 --env STARVLA_PROFILE_END_STEP=12 \
  --mount $PROJ/code:/code --mount $PROJ/model:/model \
  --mount $PROJ/data/starvla_libero:/sv_data --mount $PROJ/scripts:/scripts \
  --mount $PROJ/profiles:/profiles \
  --mount $HOST_NSYS:/opt/nsys_host \
  starvla bash -lc '
cd /code/starVLA
/opt/nsys_host/target-linux-x64/nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,cublas,cudnn,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output /profiles/starvla_qwenpi_nsys \
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    /scripts/starvla/profile_entry.py \
    --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
    --framework.name QwenPI --framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B \
    --datasets.vla_data.data_root_dir /sv_data --datasets.vla_data.data_mix libero_goal \
    --datasets.vla_data.per_device_batch_size 4 \
    --trainer.freeze_modules qwen_vl_interface \
    --trainer.gradient_accumulation_steps 1 --trainer.max_train_steps 14 \
    --trainer.eval_interval 100000 --trainer.save_interval 100000 \
    --trainer.logging_frequency 1 --run_root_dir /tmp/ck --run_id nsys_profile
echo "=== stats ==="
/opt/nsys_host/target-linux-x64/nsys stats /profiles/starvla_qwenpi_nsys.nsys-rep > /profiles/starvla_qwenpi_nsys.stats.txt 2>/dev/null && echo "stats ok" || echo "stats failed"
'
echo "profile -> $PROJ/profiles/starvla_qwenpi_nsys.nsys-rep"
