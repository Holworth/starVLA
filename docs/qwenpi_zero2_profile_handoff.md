# QwenPI ZeRO-2 Nsight Profile Handoff

> **Update 2026-07-03:** the 2-step trace described here is dominated by
> step-1 Triton compilation (the 44.6% `FillFunctor<int>` entry is 99.9% a
> warmup artifact). For steady-state results (`MAX_STEPS=100`) and the
> current optimization roadmap, see
> `docs/qwenpi_zero2_h200_steady_state_profile.md`.

This document is for reproducing and inspecting the QwenPI Nsight Systems
profile without access to the original profiling host.

## What This Captures

Target profile:

```text
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
  qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143.rank0.nsys-rep
```

Run shape:

| Item | Value |
|---|---|
| Framework | starVLA `QwenPI` |
| Model | Qwen3.5-4B VLM backbone + QwenPI action head |
| Training mode | full unfreeze, no LoRA |
| Distributed setup | single node, 8x H20, `accelerate` + DeepSpeed ZeRO-2 |
| Profiling | 8 ranks train, only rank 0 is wrapped by `nsys` |
| Nsight Systems | 2026.3.1 CLI |

`PROFILE_RANKS=0` means only rank 0 is profiled. It is not a single-GPU
training run.

## Remote Sources

Code and small profile harness files are in this GitHub branch:

```bash
export GITHUB_REPO=https://github.com/chenchaoxu7575/starVLA.git
export GITHUB_REF=qwenpi-zero2-profile-handoff-20260630
export PROFILE_HARNESS_COMMIT=887616d9e2fa2bfccb2d31ca1ba9159b890a9dcc
```

Large artifacts are in this Hugging Face dataset repo:

```bash
export HF_REPO_ID=chenchaoxNV/qwenpi-zero2-profile-artifacts
export HF_REPO_TYPE=dataset
```

The GitHub branch is a starVLA fork branch. It includes upstream starVLA source
plus these handoff additions:

```text
scripts/starvla/
containers/starvla-profile/build.sh
results/qwenpi_zero2_rank0_fulltrace_repro_README.md
results/qwenpi_zero2_profile_handoff_core.sha256
```

The Hugging Face repo contains the large files:

```text
containers/starvla-profile/starvla.sqsh
tools/nsight-systems/
data/starvla_libero/
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
results/qwenpi_zero2_profile_handoff_core.sha256
```

## Prerequisites

Target machine:

- Linux host with NVIDIA GPUs.
- NVIDIA driver compatible with the CUDA 12.4 PyTorch container.
- `enroot`.
- `git`.
- `huggingface-cli` or `hf`.
- Access to the private HF artifact dataset if it remains private.

The original run used:

| Item | Value |
|---|---|
| GPU | 8x NVIDIA H20, 97,871 MiB |
| Driver | 560.35.03 |
| enroot | 3.5.0 |

If the target machine does not have 8 comparable GPUs, the run can still be
modified, but the resulting profile is not equivalent to the delivered artifact.

## Quick Start

Choose a clean work directory:

```bash
export WORKDIR=$PWD/qwenpi_zero2_profile_handoff
mkdir -p "$WORKDIR"
cd "$WORKDIR"
```

Clone the handoff branch directly into `$WORKDIR`:

```bash
git clone "$GITHUB_REPO" .
git checkout "$GITHUB_REF"
git merge-base --is-ancestor "$PROFILE_HARNESS_COMMIT" HEAD
```

The `merge-base` command should exit with code 0. That verifies the checkout
contains the profile harness commit.

Download the required large artifacts:

```bash
huggingface-cli login

huggingface-cli download "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --local-dir "$WORKDIR" \
  --include \
    "containers/starvla-profile/starvla.sqsh" \
    "tools/nsight-systems/**" \
    "data/starvla_libero/**" \
    "profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/**" \
    "results/qwenpi_zero2_profile_handoff_core.sha256"
```

Download the Qwen checkpoint. If it is not mirrored in the artifact repo, use
the official Hugging Face model:

```bash
mkdir -p model/pretrained model/huggingface profiles

huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir model/pretrained/Qwen3.5-4B \
  --local-dir-use-symlinks False
```

Check the expected files:

```bash
test -f containers/starvla-profile/starvla.sqsh
test -x scripts/starvla/profile_qwenpi_zero2_single_node.sh
test -f starVLA/training/train_starvla.py
test -f model/pretrained/Qwen3.5-4B/config.json
test -d data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot
test -x tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
```

Verify the core artifact checksum:

```bash
sha256sum -c results/qwenpi_zero2_profile_handoff_core.sha256
```

The checksum covers the enroot image, the Nsight Systems 2026.3.1 `.deb`, and
the primary `.nsys-rep`. It does not cover every file in the dataset directory.

## Create The Enroot Container

Create an enroot container from the delivered image:

```bash
enroot create --name starvla containers/starvla-profile/starvla.sqsh
```

If `starvla` already exists, choose another name:

```bash
enroot create --name starvla-qwenpi containers/starvla-profile/starvla.sqsh
export ENROOT_NAME=starvla-qwenpi
```

Sanity-check the container and Nsight CLI:

```bash
WORKDIR_PARENT="$(dirname "$WORKDIR")"
WORKDIR_NAME="$(basename "$WORKDIR")"

ENROOT_MOUNT_HOME=no enroot start --rw \
  --env WORKDIR_NAME="$WORKDIR_NAME" \
  --mount "$WORKDIR_PARENT:/code" \
  --mount "$WORKDIR/model:/model" \
  --mount "$WORKDIR/scripts:/scripts" \
  --mount "$WORKDIR/tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli:/opt/nvidia/nsight-systems" \
  "${ENROOT_NAME:-starvla}" bash -lc '
    cd "/code/${WORKDIR_NAME}"
    python --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    /opt/nvidia/nsight-systems/2026.3.1/target-linux-x64/nsys --version
  '
```

Expected key versions:

```text
Python 3.11.11
torch 2.6.0+cu124
torch cuda 12.4
NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0
```

## Rerun The Profile

Run the same shape as the delivered profile:

```bash
cd "$WORKDIR"

bash scripts/starvla/profile_qwenpi_zero2_single_node.sh \
  RUN_ID=qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143_rerun \
  PROFILE_RANKS=0 \
  NSYS_CAPTURE_MODE=none \
  MAX_STEPS=2 \
  PROFILE_START_STEP=1 \
  PROFILE_END_STEP=1
```

Use `ENROOT_NAME=...` if the container name is not `starvla`.

The script sets:

| Item | Value |
|---|---|
| accelerate config | `starVLA/config/deepseeds/deepspeed_zero2.yaml` |
| framework | `--framework.name QwenPI` |
| backbone | `--framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B` |
| data root | `--datasets.vla_data.data_root_dir /sv_data` |
| data mix | `--datasets.vla_data.data_mix libero_goal` |
| per-rank batch | `--datasets.vla_data.per_device_batch_size 8` |
| unfreeze | `--trainer.freeze_modules ''` |
| max steps | `--trainer.max_train_steps 2` |
| nsys trace | `cuda,nvtx,cublas,cudnn,osrt` |
| capture mode | `none`, full trace from rank 0 process launch |

Output files are written under:

```text
profiles/<RUN_ID>/
  <RUN_ID>.command.log
  <RUN_ID>.cmd
  <RUN_ID>.rank0.nsys-rep
  <RUN_ID>.rank0.sqlite
  <RUN_ID>.rank0.stats.txt
  <RUN_ID>.rank0.nvtx_sum.txt
  <RUN_ID>.rank0.cuda_api_sum.txt
  <RUN_ID>.rank0.cuda_gpu_kern_sum.txt
```

Successful logs should end with:

```text
accelerate_exit_status=0
rank_reports=1
launcher_exit_status=0
```

## Inspect The Delivered Profile

Open the delivered `.nsys-rep` in Nsight Systems GUI:

```text
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
  qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143.rank0.nsys-rep
```

Regenerate CLI stats:

```bash
cd "$WORKDIR"

NSYS=tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
RUN_ID=qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143
REP=profiles/${RUN_ID}/${RUN_ID}.rank0.nsys-rep

$NSYS stats --force-export=true "$REP" \
  > "profiles/${RUN_ID}/${RUN_ID}.rank0.stats.txt"

$NSYS stats --force-export=true --report nvtx_sum "$REP" \
  > "profiles/${RUN_ID}/${RUN_ID}.rank0.nvtx_sum.txt"

$NSYS stats --force-export=true --report cuda_api_sum "$REP" \
  > "profiles/${RUN_ID}/${RUN_ID}.rank0.cuda_api_sum.txt"

$NSYS stats --force-export=true --report cuda_gpu_kern_sum "$REP" \
  > "profiles/${RUN_ID}/${RUN_ID}.rank0.cuda_gpu_kern_sum.txt"
```

The delivered report has CUDA kernel data. Top `cuda_gpu_kern_sum` entries:

| Time % | Kernel |
|---:|---|
| 44.6 | `at::native::vectorized_elementwise_kernel<..., FillFunctor<int>, ...>` |
| 7.6 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| 7.4 | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` |
| 4.1 | `merge_16x16_to_64x64_inverse_kernel` |
| 3.7 | `chunk_bwd_kernel_dqkwg` |
| 2.3 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |

Visible NVTX ranges:

| Range | Instances | Total |
|---|---:|---:|
| `profile_window` | 1 | 260.39s |
| `train_step_1` | 1 | 260.39s |
| `DeepSpeedEngine.forward` | 2 | 165.00s |
| `model_forward` | 2 | 165.00s |
| `DeepSpeedEngine.backward` | 2 | 151.06s |
| `train_step_2` | 1 | 56.18s |

## Environment Versions

Container versions:

| Package | Version |
|---|---|
| Python | 3.11.11 |
| CUDA runtime banner | 12.4.1 |
| nvcc | 12.4, V12.4.131 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| transformers | 5.3.0 |
| accelerate | 1.5.2 |
| deepspeed | 0.16.9 |
| flash-attn | 2.7.4.post1 |
| flash-linear-attention | 0.3.2 |
| causal-conv1d | 1.5.0.post8 |
| triton | 3.2.0 |
| einops | 0.8.2 |
| numpy | 1.26.4 |
| huggingface-hub | 1.19.0 |
| datasets | 5.0.0 |
| starVLA | 1.0.1 |

Nsight path:

```text
tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
```

Nsight version:

```text
NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0
```

## Build The Enroot Image From Scratch

The delivered `starvla.sqsh` is preferred for exact reproduction. To rebuild:

```bash
cd "$WORKDIR"
containers/starvla-profile/build.sh
```

Base image:

```text
docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

The build script imports the base image with enroot, installs the starVLA Python
environment, installs starVLA editable, and exports:

```text
containers/starvla-profile/starvla.sqsh
```

Rebuilding requires network access to the container registry, PyPI, and GitHub.
It also builds `flash-attn` inside the container.

## Profiling Host Change (2026-07-05)

All optimization-round numbers and the `qwenpi_milestone1` / `qwenpi_milestone2`
profiles were captured on `viking-cr-196` (H200 SXM, 8-GPU all-to-all NVLink,
NCCL 24 channels, AR busbw 472 GB/s). That node entered SLURM `draining` state
and current allocations land on `h200-nvl` boxes (e.g. `4u8g-gen-0029`) with a
**split 4+4 topology**: GPU0-3 and GPU4-7 are NVLink cliques (NV6), the two
quads are joined only by cross-socket PCIe (`SYS` in `nvidia-smi topo -m`).

Consequences on the h200-nvl boxes:

1. **Default NCCL hangs.** NCCL connects cross-quad rings via P2P/CUMEM, the
   DMA writes silently never deliver, and the very first 4-byte allreduce
   blocks until the 30-min watchdog kills the job. Set `NCCL_P2P_LEVEL=NVL`
   (forwarded by the profile script) so P2P stays inside each quad and
   cross-quad traffic falls back to SHM.
2. **Comm is ~12x slower.** Microbench AR busbw is 38.5 GB/s vs 472 GB/s on
   viking-cr-196, so step times are NOT comparable with the report/milestone
   numbers; `qwenpi_milestone3` (captured on `4u8g-gen-0029`) is the baseline
   for this node class only.

## Known Notes

1. The delivered usable artifact uses `NSYS_CAPTURE_MODE=none`, meaning a full
   rank-0 trace. Earlier `NSYS_CAPTURE_MODE=cuda` step-window attempts completed
   training but did not produce CUDA kernel data for this DeepSpeed/QwenPI path.
2. `scripts/starvla/nvtx_patch.py` instruments step, dataloader, and
   model-forward level ranges. It does not yet add fine-grained ranges inside
   the denoise loop.
3. The delivered `cuda_api_sum` does not show `cudaGraph*`,
   `cudaStreamBeginCapture`, `cudaStreamEndCapture`, or `cudaGraphLaunch`.
   Therefore this artifact does not show evidence that the denoise section used
   CUDA Graph.
4. The Qwen checkpoint path used by the run is:

```text
model/pretrained/Qwen3.5-4B
```

That local directory's model card points to `Qwen/Qwen3.5-4B` and declares
`base_model: Qwen/Qwen3.5-4B-Base`.
