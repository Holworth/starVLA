#!/usr/bin/env bash
# Build a portable enroot container for starVLA (Qwen3.5 backbone + π0-style
# flow action head, QwenPI). Produces a self-contained .sqsh that can be copied
# to other machines and run.
#
# Base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel — matches starVLA's verified
# Qwen3.5 env (torch 2.6.0+cu124, nvcc 12.4 for building flash-attn). CUDA 12.4
# runs on driver >=525 (this host: 560.35.03 OK).
#
# Usage:
#   containers/starvla-profile/build.sh            # full build + export
#   STEP=import|install|export ...                 # run a single step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_URI="${BASE_URI:-docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
BASE_SQSH="${BASE_SQSH:-${SCRIPT_DIR}/starvla-base.sqsh}"
ENROOT_NAME="${ENROOT_NAME:-starvla}"
OUT_SQSH="${OUT_SQSH:-${SCRIPT_DIR}/starvla.sqsh}"

if [[ -f "${PROJECT_ROOT}/starVLA/training/train_starvla.py" ]]; then
    CODE_DIR="${CODE_DIR:-$(dirname "${PROJECT_ROOT}")}"
    STARVLA_SRC="${STARVLA_SRC:-/code/$(basename "${PROJECT_ROOT}")}"
else
    CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/code}"
    STARVLA_SRC="${STARVLA_SRC:-/code/starVLA}"
fi

# Qwen3.5-specific pins (from starVLA requirements.txt notes; verified combo).
FLASH_ATTN_VER="${FLASH_ATTN_VER:-2.7.4.post1}"

STEP="${STEP:-all}"

do_import() {
    [[ -f "${BASE_SQSH}" ]] && { echo "base sqsh exists: ${BASE_SQSH}"; return; }
    echo "=== enroot import ${BASE_URI} ==="
    enroot import -o "${BASE_SQSH}" "${BASE_URI}"
}

do_create() {
    if ! enroot list 2>/dev/null | grep -qx "${ENROOT_NAME}"; then
        echo "=== enroot create ${ENROOT_NAME} ==="
        enroot create --name "${ENROOT_NAME}" "${BASE_SQSH}"
    fi
}

do_install() {
    echo "=== install starVLA + Qwen3.5 deps inside ${ENROOT_NAME} ==="
    ENROOT_MOUNT_HOME=no enroot start --rw \
        --mount "${CODE_DIR}:/code" \
        "${ENROOT_NAME}" bash -lc '
set -e
cd '"${STARVLA_SRC}"'
echo "[base] $(python -c "import torch;print(torch.__version__, torch.version.cuda)")"
echo "[nvcc] $(nvcc --version | grep release || true)"
pip install --no-input -r requirements.txt
# Qwen3.5 env overrides (FLA-based backbone needs these exact versions).
pip install --no-input transformers==5.3.0 flash-linear-attention==0.3.2 \
    causal_conv1d==1.5.0.post8 triton==3.2.0
# flash-attn must build against the container nvcc/torch.
pip install --no-input flash-attn=='"${FLASH_ATTN_VER}"' --no-build-isolation
pip install --no-input -e .
echo "=== verify: import QwenPI framework + key deps ==="
python -c "
import torch, transformers, deepspeed, accelerate
print(\"torch\", torch.__version__, \"cuda_avail\", torch.cuda.is_available())
print(\"transformers\", transformers.__version__)
import fla; print(\"FLA ok\", fla.__version__)
from starVLA.model.framework.VLM4A.QwenPI import *  # noqa
print(\"QwenPI import OK\")
"
'
}

do_export() {
    echo "=== enroot export -> ${OUT_SQSH} (portable) ==="
    rm -f "${OUT_SQSH}"
    enroot export -o "${OUT_SQSH}" "${ENROOT_NAME}"
    ls -lh "${OUT_SQSH}"
    echo "Copy ${OUT_SQSH} to another machine and: enroot create --name starvla <sqsh>"
}

case "${STEP}" in
    import) do_import ;;
    create) do_create ;;
    install) do_create; do_install ;;
    export) do_export ;;
    all) do_import; do_create; do_install; do_export ;;
    *) echo "unknown STEP=${STEP}"; exit 1 ;;
esac
echo "[build.sh] step '${STEP}' done."
