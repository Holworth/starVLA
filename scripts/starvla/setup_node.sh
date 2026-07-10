#!/bin/bash
# One-shot node prep for the QwenPI profiling/optimization harness.
#
# enroot containers are per-SLURM-job: every new allocation loses the container
# AND the torch 2.7 in-container upgrade. This script makes any fresh 8xH200
# node experiment-ready, unattended, in ~4 min (baseline) / ~10 min (TORCH27=1).
#
# Usage (repo root, inside the SLURM allocation on the target node):
#   bash scripts/starvla/setup_node.sh            # container only -> committed baseline (torch 2.6) runnable
#   TORCH27=1 bash scripts/starvla/setup_node.sh  # + torch 2.7.1 stack -> best config (288 ms) runnable
#
# Idempotent: every step checks before acting. Ends by printing the NCCL env
# this node needs and the ready-to-paste best-config command.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENROOT_NAME=${ENROOT_NAME:-starvla}
SQSH=${SQSH:-containers/starvla-profile/starvla.sqsh}

# ---------------------------------------------------------------- container
if enroot list | grep -qx "${ENROOT_NAME}"; then
    echo "[setup] container '${ENROOT_NAME}' already exists"
else
    echo "[setup] creating enroot container '${ENROOT_NAME}' from ${SQSH} (~4 min)"
    enroot create --name "${ENROOT_NAME}" "${SQSH}"
fi

# ---------------------------------------------------------------- topology
# h200-nvl boxes are 4+4 NVLink quads with SYS between the quads: the first
# NCCL collective HANGS (30 min -> SIGABRT) unless P2P is capped at NVLink
# reach. Detect by looking for SYS in the GPU<->GPU cells of the topo matrix
# (GPU->NIC cells are SYS even on full-NVLink viking nodes, so scope the check
# to the first 8 data columns of GPU rows only).
NGPU=$(nvidia-smi --list-gpus | wc -l)
# awk must consume all input (END-block, no early exit): an early exit sends
# SIGPIPE to nvidia-smi, which under pipefail+set -e kills the whole script.
SPLIT=$(nvidia-smi topo -m | awk -v n="${NGPU}" '/^GPU/ {for (i = 2; i <= n + 1; i++) if ($i == "SYS") found = 1} END {if (found) print "yes"}')
if [ "${SPLIT}" = "yes" ]; then
    NCCL_HINT="NCCL_P2P_LEVEL=NVL "
    echo "[setup] SPLIT topology (4+4 quads) -> prefix every run with NCCL_P2P_LEVEL=NVL (REQUIRED, first collective hangs without it)"
    echo "[setup] NOTE: timing on this node is NOT comparable to viking full-NVLink numbers; use it for same-node A/B deltas only"
else
    NCCL_HINT=""
    echo "[setup] full-NVLink topology (viking-class), no NCCL_P2P_LEVEL needed"
fi

# ---------------------------------------------------------------- torch 2.7
# Container-level upgrade, lost on every job change. Recipe validated in
# docs/qwenpi_zero2_h200_optimization_log.md Round 8: torch 2.7.1 + NCCL 2.26
# + triton 3.3 alone was -13% step time, and is required for the fused-stack
# CUDA-Graph config (STARVLA_FLA_TRACE needs the fla edit done below).
if [ "${TORCH27:-0}" = "1" ]; then
    ENROOT_MOUNT_HOME=no enroot start --rw "${ENROOT_NAME}" bash -lc '
        set -euo pipefail
        if python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith(\"2.7\") else 1)" 2>/dev/null; then
            echo "[setup] torch 2.7 already installed in container"
        else
            echo "[setup] installing torch 2.7.1 stack (~6 min)"
            pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
            pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
            pip install --no-cache-dir --no-deps --force-reinstall "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        fi
        # fla: drop the @torch.compiler.disable on chunk_gated_delta_rule so
        # dynamo traces into it (STARVLA_FLA_TRACE then constant-folds the
        # device checks -> break-free single-graph groups).
        FLA=$(python -c "import fla, os; print(os.path.dirname(fla.__file__))")
        F="${FLA}/ops/gated_delta_rule/chunk.py"
        LN=$(grep -n "^@torch.compiler.disable$" "${F}" | head -1 | cut -d: -f1 || true)
        if [ -n "${LN}" ]; then
            cp "${F}" "${F}.bak"
            sed -i "${LN}s/^@torch.compiler.disable$/# @torch.compiler.disable  # [starvla] removed so dynamo can trace the op/" "${F}"
            echo "[setup] fla ${F}:${LN} @torch.compiler.disable commented out (.bak kept)"
        else
            grep -q "# @torch.compiler.disable" "${F}" \
                && echo "[setup] fla edit already applied" \
                || { echo "[setup] ERROR: @torch.compiler.disable not found in ${F} - fla version changed, check manually"; exit 1; }
        fi
        python -c "import torch, flash_attn, causal_conv1d, triton; print(\"[setup] torch\", torch.__version__, \"nccl\", torch.cuda.nccl.version(), \"flash_attn\", flash_attn.__version__, \"triton\", triton.__version__)"
    '
fi

# ---------------------------------------------------------------- summary
cat <<EOF

[setup] node ready. Ready-to-paste commands (see docs/qwenpi_pending_experiments.md for the A/B queue):

  # committed baseline (torch 2.6 ok, ~424 ms on viking):
  ${NCCL_HINT}RUN_ID=sanity PROFILE_RANKS=none MAX_STEPS=40 \\
    EXTRA_TRAIN_ARGS='--datasets.vla_data.preprocess_in_collate true --datasets.vla_data.num_workers 8 --datasets.vla_data.collate_pad_to 192 --datasets.vla_data.collate_host_batch true --framework.qwenvl.attn_implementation mixed --framework.qwenvl.compile_language_model false --framework.action_model.compile_dit true --framework.action_model.compile_mode reduce-overhead --framework.action_model.pad_encoder_seq_to 192' \\
    bash scripts/starvla/profile_qwenpi_zero2_single_node.sh

  # best config (needs TORCH27=1 setup, ~288 ms on viking):
  ${NCCL_HINT}RUN_ID=best PROFILE_RANKS=none MAX_STEPS=40 \\
    STARVLA_DEFER_AG=1 STARVLA_FUSED_TEXT_STACK=1 STARVLA_FLA_TRACE=1 STARVLA_FUSED_GROUP_SIZE=8 \\
    STARVLA_FUSED_COMPILE_MODE=reduce-overhead STARVLA_FUSED_VISION=1 STARVLA_FAST_MM_MERGE=1 \\
    EXTRA_TRAIN_ARGS='--datasets.vla_data.preprocess_in_collate true --datasets.vla_data.num_workers 8 --datasets.vla_data.collate_pad_to 192 --datasets.vla_data.collate_host_batch true --datasets.vla_data.collate_mrope_posids true --framework.qwenvl.attn_implementation mixed --framework.qwenvl.compile_language_model false --framework.action_model.compile_dit true --framework.action_model.compile_mode reduce-overhead --framework.action_model.pad_encoder_seq_to 192' \\
    bash scripts/starvla/profile_qwenpi_zero2_single_node.sh
EOF
