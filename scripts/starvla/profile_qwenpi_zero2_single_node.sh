#!/usr/bin/env bash
# Nsight Systems profile for single-node starVLA QwenPI full-unfreeze training.
#
# Each distributed rank is launched under its own nsys process, producing
# per-rank reports: profiles/$RUN_ID/$RUN_ID.rank{0..7}.nsys-rep.
set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    export "${arg}"
  else
    echo "unsupported argument: ${arg} (expected KEY=VALUE)" >&2
    exit 2
  fi
done

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${HERE}/../.." && pwd)"

if [[ -f "${PROJ}/starVLA/training/train_starvla.py" ]]; then
  STARVLA_SRC_HOST="${STARVLA_SRC_HOST:-${PROJ}}"
else
  STARVLA_SRC_HOST="${STARVLA_SRC_HOST:-${PROJ}/code/starVLA}"
fi
CODE_MOUNT_HOST="$(cd "$(dirname "${STARVLA_SRC_HOST}")" && pwd)"
STARVLA_SRC_IN="/code/$(basename "${STARVLA_SRC_HOST}")"

GPUS="${GPUS:-8}"
BS="${BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
# DiT precision (framework.action_model.dit_dtype): bf16 = the measured config,
# none = single-switch rollback to the caller's fp32 (pre-optimization behavior).
DIT_DTYPE="${DIT_DTYPE:-bf16}"
# MRoPE position-id cache (framework.qwenvl.mrope_posid_cache): true = the
# measured config, MROPE_CACHE=false rolls back to HF per-step compute.
MROPE_CACHE="${MROPE_CACHE:-true}"
# Structured worker-layout + bounded per-rank GPU metadata caches. This one
# switch rolls back MRoPE/ViT/split/index/causal-mask metadata consumers to
# their stock HF paths for paired correctness runs.
METADATA_CACHE="${METADATA_CACHE:-1}"
PROFILE_START_STEP="${PROFILE_START_STEP:-${START_STEP:-11}}"
PROFILE_END_STEP="${PROFILE_END_STEP:-${END_STEP:-13}}"
if (( PROFILE_END_STEP < PROFILE_START_STEP )); then
  echo "PROFILE_END_STEP must be >= PROFILE_START_STEP" >&2
  exit 2
fi
MAX_STEPS="${MAX_STEPS:-$((PROFILE_END_STEP + 1))}"
PROFILE_RANKS="${PROFILE_RANKS:-all}"
RUN_ID="${RUN_ID:-qwenpi_zero2_1node_bs${BS}_ranknsys_s${PROFILE_START_STEP}_e${PROFILE_END_STEP}_$(date +%Y%m%d_%H%M%S)}"

NSYS_HOST_DIR="${NSYS_HOST_DIR:-${PROJ}/tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli}"
NSYS_VERSION_DIR="${NSYS_VERSION_DIR:-2026.3.1}"
NSYS_BIN="/opt/nvidia/nsight-systems/${NSYS_VERSION_DIR}/target-linux-x64/nsys"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,cublas,cudnn,osrt}"
NSYS_CAPTURE_MODE="${NSYS_CAPTURE_MODE:-cuda}"
NSYS_NVTX_CAPTURE="${NSYS_NVTX_CAPTURE:-profile_window}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"
NSYS_KILL="${NSYS_KILL:-sigterm}"
NSYS_CUDA_MEMORY_USAGE="${NSYS_CUDA_MEMORY_USAGE:-false}"
NSYS_START_STAGGER_SEC="${NSYS_START_STAGGER_SEC:-0}"
ENROOT_NAME="${ENROOT_NAME:-starvla}"

OUT_DIR="${OUT_DIR:-${PROJ}/profiles/${RUN_ID}}"
OUT_BASE="${OUT_DIR}/${RUN_ID}"
OUT_DIR_IN="/profiles/${RUN_ID}"
OUT_BASE_IN="${OUT_DIR_IN}/${RUN_ID}"
LOG_FILE="${OUT_BASE}.command.log"

mkdir -p "${OUT_DIR}"

cat >"${OUT_BASE}.cmd" <<EOF
ENROOT_NAME=${ENROOT_NAME} ENROOT_MOUNT_HOME=no enroot start --rw ... ${ENROOT_NAME} bash -lc 'accelerate launch --no_python ... /scripts/starvla/nsys_rank_wrapper.sh ...'
EOF

{
  echo "run_id=${RUN_ID}"
  echo "gpus=${GPUS} per_gpu_batch=${BS} max_steps=${MAX_STEPS} grad_accum=${GRAD_ACCUM}"
  echo "dit_dtype=${DIT_DTYPE}"
  echo "metadata_cache=${METADATA_CACHE} mrope_cache=${MROPE_CACHE}"
  echo "profile_prebuild_seed=${STARVLA_PROFILE_PREBUILD_SEED:-}"
  echo "profile_window=${PROFILE_START_STEP}..${PROFILE_END_STEP}"
  echo "profile_ranks=${PROFILE_RANKS}"
  echo "out_dir=${OUT_DIR}"
  echo "nsys=${NSYS_BIN}"
  echo "nsys_trace=${NSYS_TRACE}"
  echo "nsys_capture_mode=${NSYS_CAPTURE_MODE} nvtx_capture=${NSYS_NVTX_CAPTURE}"
  echo "nsys_capture_range_end=${NSYS_CAPTURE_RANGE_END} kill=${NSYS_KILL}"
  echo "nsys_cuda_memory_usage=${NSYS_CUDA_MEMORY_USAGE}"
  echo "nsys_start_stagger_sec=${NSYS_START_STAGGER_SEC}"
  echo "enroot_name=${ENROOT_NAME}"
  echo "starvla_src_host=${STARVLA_SRC_HOST}"
  echo "starvla_src_in=${STARVLA_SRC_IN}"
  echo "log=${LOG_FILE}"
} | tee "${LOG_FILE}"

set +e
ENROOT_MOUNT_HOME=no enroot start --rw \
  --env NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_MODE=disabled --env PYTHONWARNINGS=ignore \
  --env HF_HOME=/model/huggingface --env HF_HUB_CACHE=/model/huggingface/hub \
  --env PYTHONPATH=/scripts/starvla \
  --env STARVLA_METADATA_CACHE="${METADATA_CACHE}" \
  --env STARVLA_CHECK_METADATA_CACHE="${STARVLA_CHECK_METADATA_CACHE:-}" \
  --env STARVLA_PROFILE_PREBUILD_SEED="${STARVLA_PROFILE_PREBUILD_SEED:-}" \
  --env STARVLA_CHECK_POSIDS="${STARVLA_CHECK_POSIDS:-}" \
  --env STARVLA_EMIT_NVTX="${STARVLA_EMIT_NVTX:-}" \
  --env STARVLA_NVTX_ACTION_HEAD="${STARVLA_NVTX_ACTION_HEAD:-}" \
  --env STARVLA_FUSED_TEXT_STACK=1 \
  --env STARVLA_FLA_TRACE=1 \
  --env STARVLA_FUSED_GROUP_SIZE=8 \
  --env STARVLA_FUSED_COMPILE_MODE=reduce-overhead \
  --env STARVLA_FUSED_VISION=1 \
  --env STARVLA_FAST_MM_MERGE=1 \
  --env STARVLA_INDEX_MM_MERGE=1 \
  --env STARVLA_VIT_INPUT_CACHE="${STARVLA_VIT_INPUT_CACHE:-1}" \
  --env STARVLA_CONV_TORCH_FALLBACK="${STARVLA_CONV_TORCH_FALLBACK:-}" \
  --env STARVLA_PROFILE_START_STEP="${PROFILE_START_STEP}" \
  --env STARVLA_PROFILE_END_STEP="${PROFILE_END_STEP}" \
  --env STARVLA_PROFILE_TRIGGER="${NSYS_CAPTURE_MODE}" \
  --env STARVLA_PROFILE_RANKS="${PROFILE_RANKS}" \
  --env STARVLA_NSYS_RUN_ID="${RUN_ID}" \
  --env STARVLA_NSYS_OUT_DIR="${OUT_DIR_IN}" \
  --env STARVLA_NSYS_BIN="${NSYS_BIN}" \
  --env STARVLA_NSYS_TRACE="${NSYS_TRACE}" \
  --env STARVLA_NSYS_CAPTURE_MODE="${NSYS_CAPTURE_MODE}" \
  --env STARVLA_NSYS_NVTX_CAPTURE="${NSYS_NVTX_CAPTURE}" \
  --env STARVLA_NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END}" \
  --env STARVLA_NSYS_KILL="${NSYS_KILL}" \
  --env STARVLA_NSYS_CUDA_MEMORY_USAGE="${NSYS_CUDA_MEMORY_USAGE}" \
  --env STARVLA_NSYS_START_STAGGER_SEC="${NSYS_START_STAGGER_SEC}" \
  --mount "${NSYS_HOST_DIR}:/opt/nvidia/nsight-systems" \
  --mount "${CODE_MOUNT_HOST}:/code" \
  --mount "${PROJ}/model:/model" \
  --mount "${PROJ}/profiles:/profiles" \
  --mount "${PROJ}/data/starvla_libero:/sv_data" \
  --mount "${PROJ}/scripts:/scripts" \
  "${ENROOT_NAME}" bash -lc "cd ${STARVLA_SRC_IN} && \
    ${NSYS_BIN} --version && \
    set +e; \
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2_qwenpi.yaml \
      --num_processes ${GPUS} \
      --no_python \
      /scripts/starvla/nsys_rank_wrapper.sh \
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
      --framework.action_model.dit_dtype ${DIT_DTYPE} \
      --datasets.vla_data.preprocess_in_collate true \
      --datasets.vla_data.num_workers ${NUM_WORKERS:-8} \
      ${VIDEO_BACKEND:+--datasets.vla_data.video_backend ${VIDEO_BACKEND}} \
      --datasets.vla_data.collate_pad_to 192 \
      --framework.qwenvl.mrope_posid_cache ${MROPE_CACHE} \
      --framework.qwenvl.attn_implementation mixed \
      --framework.action_model.compile_dit true \
      --framework.action_model.compile_mode reduce-overhead \
      --framework.action_model.pad_encoder_seq_to 192 \
      --run_root_dir /tmp/ck \
      --run_id ${RUN_ID}; \
    status=\$?; \
    echo accelerate_exit_status=\${status}; \
    /scripts/starvla/collect_rank_nsys_stats.sh ${OUT_BASE_IN}; \
    exit \${status}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "launcher_exit_status=${status}" | tee -a "${LOG_FILE}"
echo "reports=${OUT_BASE}.rank*.nsys-rep" | tee -a "${LOG_FILE}"
exit "${status}"
