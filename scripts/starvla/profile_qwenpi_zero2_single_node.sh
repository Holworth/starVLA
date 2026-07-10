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
# graph = opaque CUDA-graph blocks (low overhead); node = per-kernel visibility
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-graph}"
ENROOT_NAME="${ENROOT_NAME:-starvla}"
# Optional NCCL overrides (e.g. NCCL_PROTO=Simple, NCCL_ALGO=NVLS). Left unset
# by default so NCCL keeps auto-selecting; only touched when the caller asks.
NCCL_PROTO="${NCCL_PROTO:-}"
NCCL_ALGO="${NCCL_ALGO:-}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-}"
LOGGING_FREQ="${LOGGING_FREQ:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

OUT_DIR="${OUT_DIR:-${PROJ}/profiles/${RUN_ID}}"
OUT_BASE="${OUT_DIR}/${RUN_ID}"
OUT_DIR_IN="/profiles/${RUN_ID}"
OUT_BASE_IN="${OUT_DIR_IN}/${RUN_ID}"
LOG_FILE="${OUT_BASE}.command.log"

mkdir -p "${OUT_DIR}"

# Build the NCCL_PROTO override as an optional --env arg (array, so it's
# simply absent -- not an empty NCCL_PROTO="" -- when unset) plus a re-export
# run right before accelerate launch. The container's `bash -lc` login shell
# sources /etc/profile & friends, and NGC/PyTorch base images sometimes bake
# in their own NCCL_* defaults there, which would otherwise silently clobber
# whatever `enroot --env` set before the profile scripts even ran.
EXTRA_ENV_ARGS=()
NCCL_PROTO_EXPORT_CMD=""
# NCCL_P2P_LEVEL: required on split-topology boxes (e.g. h200-nvl 4+4 quads
# joined only by SYS paths) where default cross-quad P2P/CUMEM silently never
# delivers and the first collective hangs; NCCL_P2P_LEVEL=NVL keeps P2P inside
# each NVLink island and falls back to SHM across.
for nccl_var in NCCL_PROTO NCCL_ALGO NCCL_NVLS_ENABLE NCCL_P2P_LEVEL NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_BUFFSIZE STARVLA_COLLATE_TIMING STARVLA_DEFER_AG STARVLA_FUSED_TEXT_STACK STARVLA_FUSED_COMPILE_MODE STARVLA_FUSED_GROUP_SIZE STARVLA_FLA_TRACE STARVLA_FUSED_VISION STARVLA_FAST_MM_MERGE STARVLA_GRAD_COPY_STREAM STARVLA_CHECK_POSIDS STARVLA_COMPILED_AUTOGRAD STARVLA_CA_MODE; do
  nccl_val="${!nccl_var:-}"
  if [[ -n "${nccl_val}" ]]; then
    EXTRA_ENV_ARGS+=(--env "${nccl_var}=${nccl_val}")
    NCCL_PROTO_EXPORT_CMD+="export ${nccl_var}=${nccl_val} && "
  fi
done

cat >"${OUT_BASE}.cmd" <<EOF
ENROOT_NAME=${ENROOT_NAME} ENROOT_MOUNT_HOME=no enroot start --rw ... ${ENROOT_NAME} bash -lc 'accelerate launch --no_python ... /scripts/starvla/nsys_rank_wrapper.sh ...'
EOF

{
  echo "run_id=${RUN_ID}"
  echo "gpus=${GPUS} per_gpu_batch=${BS} max_steps=${MAX_STEPS} grad_accum=${GRAD_ACCUM}"
  echo "profile_window=${PROFILE_START_STEP}..${PROFILE_END_STEP}"
  echo "profile_ranks=${PROFILE_RANKS}"
  echo "out_dir=${OUT_DIR}"
  echo "nsys=${NSYS_BIN}"
  echo "nsys_trace=${NSYS_TRACE}"
  echo "nsys_capture_mode=${NSYS_CAPTURE_MODE} nvtx_capture=${NSYS_NVTX_CAPTURE}"
  echo "nsys_capture_range_end=${NSYS_CAPTURE_RANGE_END} kill=${NSYS_KILL}"
  echo "nsys_cuda_memory_usage=${NSYS_CUDA_MEMORY_USAGE}"
  echo "nsys_start_stagger_sec=${NSYS_START_STAGGER_SEC}"
  echo "nccl_proto=${NCCL_PROTO:-<unset, NCCL auto-selects>}"
  echo "nccl_algo=${NCCL_ALGO:-<unset>} nccl_nvls_enable=${NCCL_NVLS_ENABLE:-<unset>}"
  echo "logging_freq=${LOGGING_FREQ} extra_train_args=${EXTRA_TRAIN_ARGS:-<none>}"
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
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING \
  --env NCCL_DEBUG_FILE="${OUT_DIR_IN}/nccl_debug.%h.%p.log" \
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
  --env STARVLA_NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE}" \
  --env STARVLA_NSYS_START_STAGGER_SEC="${NSYS_START_STAGGER_SEC}" \
  "${EXTRA_ENV_ARGS[@]}" \
  --mount "${NSYS_HOST_DIR}:/opt/nvidia/nsight-systems" \
  --mount "${CODE_MOUNT_HOST}:/code" \
  --mount "${PROJ}/model:/model" \
  --mount "${PROJ}/profiles:/profiles" \
  --mount "${PROJ}/data/starvla_libero:/sv_data" \
  --mount "${PROJ}/scripts:/scripts" \
  "${ENROOT_NAME}" bash -lc "cd ${STARVLA_SRC_IN} && \
    ${NCCL_PROTO_EXPORT_CMD}echo '--- NCCL_* env right before accelerate launch ---'; \
    env | grep -i '^NCCL_' || true; \
    ${NSYS_BIN} --version && \
    set +e; \
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
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
      --trainer.logging_frequency ${LOGGING_FREQ} \
      --run_root_dir /tmp/ck \
      --run_id ${RUN_ID} ${EXTRA_TRAIN_ARGS}; \
    status=\$?; \
    echo accelerate_exit_status=\${status}; \
    /scripts/starvla/collect_rank_nsys_stats.sh ${OUT_BASE_IN}; \
    exit \${status}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "launcher_exit_status=${status}" | tee -a "${LOG_FILE}"
echo "reports=${OUT_BASE}.rank*.nsys-rep" | tee -a "${LOG_FILE}"
exit "${status}"
