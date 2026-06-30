#!/usr/bin/env bash
# 起 starVLA 容器(固定挂载/env),在容器内执行 "$@"。无参数 → 交互 bash。
set -euo pipefail
PROJ=/home/chenchaox/project/phyai_fd
ENROOT_MOUNT_HOME=no enroot start --rw \
  --env NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE=disabled --env PYTHONWARNINGS=ignore \
  --env HF_HOME=/model/huggingface --env HF_HUB_CACHE=/model/huggingface/hub \
  --mount ${PROJ}/code:/code \
  --mount ${PROJ}/model:/model \
  --mount ${PROJ}/data/starvla_libero:/sv_data \
  --mount ${PROJ}/scripts:/scripts \
  starvla "${@:-bash}"
