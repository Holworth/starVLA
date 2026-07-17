#!/bin/bash
# [opt2] In-container upgrade: torch 2.7.1 + NCCL 2.26.2 + triton 3.3.1.
#
# Why: measured -13% step time by itself (better NCCL launch path, newer
# triton codegen), and it is a PREREQUISITE for the CUDA-Graph work in the
# next commit — torch 2.6's dynamo cannot trace fla's gated-delta kernels
# (graph breaks per layer), so the fused text-stack graphs only exist on 2.7.
#
# enroot containers are per-SLURM-job: this upgrade is lost on every new
# allocation and must be re-run (idempotent, ~6 min).
#
# Usage (repo root, after `enroot create --name starvla <sqsh>`):
#   bash scripts/starvla/upgrade_container_torch27.sh
set -euo pipefail

ENROOT_NAME=${ENROOT_NAME:-starvla}

ENROOT_MOUNT_HOME=no enroot start --rw "${ENROOT_NAME}" bash -lc '
    set -euo pipefail
    if python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith(\"2.7\") else 1)" 2>/dev/null; then
        echo "[torch27] torch 2.7 already installed in container"
    else
        echo "[torch27] installing torch 2.7.1 stack (~6 min)"
        pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
        pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        pip install --no-cache-dir --no-deps --force-reinstall "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    fi
    # fla: drop the @torch.compiler.disable on chunk_gated_delta_rule so dynamo
    # can trace into it (required by the fused text-stack CUDA graphs later).
    FLA=$(python -c "import fla, os; print(os.path.dirname(fla.__file__))")
    F="${FLA}/ops/gated_delta_rule/chunk.py"
    LN=$(grep -n "^@torch.compiler.disable$" "${F}" | head -1 | cut -d: -f1 || true)
    if [ -n "${LN}" ]; then
        cp "${F}" "${F}.bak"
        sed -i "${LN}s/^@torch.compiler.disable$/# @torch.compiler.disable  # [starvla] removed so dynamo can trace the op/" "${F}"
        echo "[torch27] fla ${F}:${LN} @torch.compiler.disable commented out (.bak kept)"
    else
        grep -q "# @torch.compiler.disable" "${F}" \
            && echo "[torch27] fla edit already applied" \
            || { echo "[torch27] ERROR: @torch.compiler.disable not found in ${F} - fla version changed, check manually"; exit 1; }
    fi
    python -c "import torch, flash_attn, causal_conv1d, triton; print(\"[torch27] torch\", torch.__version__, \"nccl\", torch.cuda.nccl.version(), \"flash_attn\", flash_attn.__version__, \"triton\", triton.__version__)"
'
