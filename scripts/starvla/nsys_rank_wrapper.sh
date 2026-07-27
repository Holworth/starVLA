#!/usr/bin/env bash
# Run one accelerate/deepspeed local rank under Nsight Systems.
set -euo pipefail

rank="${LOCAL_RANK:-${RANK:-0}}"
world="${WORLD_SIZE:-1}"
run_id="${STARVLA_NSYS_RUN_ID:?STARVLA_NSYS_RUN_ID is required}"
out_dir="${STARVLA_NSYS_OUT_DIR:?STARVLA_NSYS_OUT_DIR is required}"
nsys_bin="${STARVLA_NSYS_BIN:-/opt/nvidia/nsight-systems/2026.3.1/target-linux-x64/nsys}"
trace="${STARVLA_NSYS_TRACE:-cuda,nvtx,cublas,cudnn,osrt}"
cuda_graph_trace="${STARVLA_NSYS_CUDA_GRAPH_TRACE:-node}"
capture_mode="${STARVLA_NSYS_CAPTURE_MODE:-nvtx}"
nvtx_capture="${STARVLA_NSYS_NVTX_CAPTURE:-profile_window}"
capture_end="${STARVLA_NSYS_CAPTURE_RANGE_END:-stop}"
kill_signal="${STARVLA_NSYS_KILL:-sigterm}"
cuda_memory_usage="${STARVLA_NSYS_CUDA_MEMORY_USAGE:-false}"
profile_ranks="${STARVLA_PROFILE_RANKS:-all}"

mkdir -p "${out_dir}"

should_profile=0
if [[ "${profile_ranks}" == "all" ]]; then
  should_profile=1
else
  IFS=',' read -ra selected_ranks <<<"${profile_ranks}"
  for selected in "${selected_ranks[@]}"; do
    if [[ "${selected}" == "${rank}" ]]; then
      should_profile=1
      break
    fi
  done
fi

if [[ "${should_profile}" == "1" ]]; then
  out_base="${out_dir}/${run_id}.rank${rank}"
  session="sv_${run_id}_rank${rank}"
  export STARVLA_NSYS_SESSION="${session}"
  export STARVLA_PROFILE_TRIGGER="${capture_mode}"
  echo "[nsys_rank_wrapper] profiling rank ${rank}/${world}: ${out_base}.nsys-rep" >&2
  nsys_args=(
    profile --force-overwrite=true
    --trace="${trace}"
    --cuda-graph-trace="${cuda_graph_trace}"
    --sample=none --cpuctxsw=none
    --cuda-memory-usage="${cuda_memory_usage}"
    --export=sqlite
    --output "${out_base}"
  )
  if [[ "${capture_mode}" == "session" ]]; then
    nsys_args+=(--start-later=true --session-new="${session}")
  elif [[ "${capture_mode}" == "nvtx" ]]; then
    nsys_args+=(--capture-range=nvtx --nvtx-capture="${nvtx_capture}" --capture-range-end="${capture_end}" --kill="${kill_signal}")
  elif [[ "${capture_mode}" == "none" ]]; then
    nsys_args+=(--capture-range=none)
  else
    nsys_args+=(--capture-range=cudaProfilerApi --capture-range-end="${capture_end}" --kill="${kill_signal}")
  fi
  exec "${nsys_bin}" "${nsys_args[@]}" python /scripts/starvla/profile_entry.py "$@"
fi

echo "[nsys_rank_wrapper] running rank ${rank}/${world} without nsys" >&2
export STARVLA_PROFILE_TRIGGER=nvtx
exec python /scripts/starvla/profile_entry.py "$@"
