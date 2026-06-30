#!/usr/bin/env bash
# Export per-rank Nsight Systems summaries for files matching OUT_BASE.rank*.nsys-rep.
set -u

out_base="${1:?usage: collect_rank_nsys_stats.sh OUT_BASE}"
nsys_bin="${STARVLA_NSYS_BIN:-/opt/nvidia/nsight-systems/2026.3.1/target-linux-x64/nsys}"

shopt -s nullglob
reports=("${out_base}".rank*.nsys-rep)

echo "rank_reports=${#reports[@]}"
for report in "${reports[@]}"; do
  base="${report%.nsys-rep}"
  echo "stats_for=${report}"
  "${nsys_bin}" stats --force-export=true "${report}" >"${base}.stats.txt" 2>"${base}.stats.err" || true
  "${nsys_bin}" stats --force-export=true --report nvtx_sum "${report}" >"${base}.nvtx_sum.txt" 2>"${base}.nvtx_sum.err" || true
  "${nsys_bin}" stats --force-export=true --report cuda_api_sum "${report}" >"${base}.cuda_api_sum.txt" 2>"${base}.cuda_api_sum.err" || true
  "${nsys_bin}" stats --force-export=true --report cuda_gpu_kern_sum "${report}" >"${base}.cuda_gpu_kern_sum.txt" 2>"${base}.cuda_gpu_kern_sum.err" || true
done
