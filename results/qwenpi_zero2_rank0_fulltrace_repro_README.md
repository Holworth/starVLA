# QwenPI ZeRO-2 Nsight Profile 远端同步与复现说明

本文档面向**不能访问原始机器**的同事。默认交付方式：

- 小文件/代码通过 GitHub fork 同步。
- 大文件 artifact 通过 Hugging Face repo 同步。
- 接收方在任意 `$WORKDIR` 下拉取并复现。

目标 profile：

```text
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
  qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143.rank0.nsys-rep
```

这个 profile 的含义：

- 框架：starVLA 自己实现的 `QwenPI`。
- 训练方式：Qwen3.5-4B VLM backbone + QwenPI action head，full unfreeze。
- 分布式：单节点 8 卡 H20，`accelerate` + DeepSpeed ZeRO-2。
- profiling：8 个训练 rank 都启动，只有 rank 0 被 `nsys` 包住。
- `PROFILE_RANKS=0` 表示只 profile rank0，不是单卡训练。

## 1. 远端源

把下面两个变量替换成实际地址：

```bash
export GITHUB_REPO=https://github.com/chenchaoxu7575/starVLA.git
export GITHUB_REF=qwenpi-zero2-profile-handoff-20260630
export HF_REPO_ID=chenchaoxNV/qwenpi-zero2-profile-artifacts
export HF_REPO_TYPE=dataset
```

Current HF artifact repo:

```text
https://huggingface.co/datasets/chenchaoxNV/qwenpi-zero2-profile-artifacts
```

The GitHub handoff branch is based on the starVLA fork and includes the local
profile harness in addition to upstream starVLA source.

推荐远端职责：

| 远端 | 内容 | 说明 |
|---|---|---|
| GitHub fork | `code/starVLA` 或等价源码 | 固定到本次 commit |
| GitHub fork | `scripts/starvla/*` | 本次 nsys wrapper、profile entry、NVTX patch |
| GitHub fork | `containers/starvla-profile/build.sh` | 从头构建 enroot 镜像的脚本 |
| GitHub fork | `results/qwenpi_zero2_rank0_fulltrace_repro_README.md` | 本文档 |
| Hugging Face | `containers/starvla-profile/starvla.sqsh` | 已构建 enroot 镜像，约 16G |
| Hugging Face | `tools/nsight-systems/` | Nsight Systems 2026.3.1 CLI，约 796M |
| Hugging Face | `data/starvla_libero/` | LIBERO 数据，约 331M |
| Hugging Face | `profiles/.../` | 已生成 profile artifact，约 306M |
| Hugging Face | `results/qwenpi_zero2_profile_handoff_core.sha256` | 核心 artifact checksum |
| Hugging Face 或官方 HF | `model/pretrained/Qwen3.5-4B/` | 可用官方 `Qwen/Qwen3.5-4B` 下载，也可同步精确本地副本 |

注意：GitHub fork 不能只放上游 starVLA。要复现这份 profile，必须包含
`scripts/starvla/*` 和 `containers/starvla-profile/build.sh` 这类 profile harness。

## 2. Hugging Face Artifact Tree

推荐 Hugging Face dataset repo 内保持与接收方 `$WORKDIR` 一致的相对路径：

```text
containers/starvla-profile/starvla.sqsh
tools/nsight-systems/
data/starvla_libero/
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
results/qwenpi_zero2_profile_handoff_core.sha256

# optional, only if not using official Qwen download:
model/pretrained/Qwen3.5-4B/
```

不要上传这些无关/重复目录：

```text
containers/starvla-profile/starvla-base.sqsh   # 可重建，不是运行必须，约 13G
model/huggingface/                             # 与 local checkpoint 重复，约 8.8G
model/swift_output/                            # ms-swift 旧实验输出，约 17G
code/starVLA/playground/                       # 本机临时/数据目录，约 16G
```

## 3. 生产者：同步到 GitHub fork

在原始工作区准备 GitHub fork。目标是让接收方 clone 后拥有这些小文件：

```text
scripts/starvla/
containers/starvla-profile/build.sh
results/qwenpi_zero2_rank0_fulltrace_repro_README.md
```

同时确保 starVLA 源码固定在本次 commit：

```text
repo:   https://github.com/starVLA/starVLA.git
branch: starVLA_dev
commit: cdf5434438f4449cff85e3588956f7706a5c9cc3
```

如果你的 fork 是完整 handoff repo，建议在 repo 根目录保留本实验目录布局：

```text
code/starVLA/
scripts/starvla/
containers/starvla-profile/build.sh
results/qwenpi_zero2_rank0_fulltrace_repro_README.md
```

如果你的 fork 只是 starVLA fork，则需要把 profile harness 也提交进去，比如：

```text
scripts/starvla/
containers/starvla-profile/build.sh
```

接收方只要 clone 到 `$WORKDIR` 后能看到这些相对路径即可。

## 4. 生产者：同步到 Hugging Face

登录 Hugging Face：

```bash
huggingface-cli login
```

创建 private dataset repo，或者用已有 repo：

```bash
huggingface-cli repo create "$HF_REPO_ID" --type dataset --private
```

从原始机器上传大文件。下面命令假设原始工作区是：

```bash
export SRC_WORKDIR=/home/chenchaox/project/phyai_fd
cd "$SRC_WORKDIR"
```

上传已构建容器：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  containers/starvla-profile/starvla.sqsh \
  containers/starvla-profile/starvla.sqsh \
  --repo-type "$HF_REPO_TYPE"
```

上传 Nsight Systems CLI：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  tools/nsight-systems \
  tools/nsight-systems \
  --repo-type "$HF_REPO_TYPE"
```

上传数据：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  data/starvla_libero \
  data/starvla_libero \
  --repo-type "$HF_REPO_TYPE"
```

上传本次 profile artifact：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143 \
  profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143 \
  --repo-type "$HF_REPO_TYPE"
```

上传 checksum：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  results/qwenpi_zero2_profile_handoff_core.sha256 \
  results/qwenpi_zero2_profile_handoff_core.sha256 \
  --repo-type "$HF_REPO_TYPE"
```

如果需要同步精确的本地 Qwen checkpoint，而不是让接收方从官方 Qwen repo 下载：

```bash
huggingface-cli upload "$HF_REPO_ID" \
  model/pretrained/Qwen3.5-4B \
  model/pretrained/Qwen3.5-4B \
  --repo-type "$HF_REPO_TYPE"
```

核心 checksum 已在本工作区生成：

```text
results/qwenpi_zero2_profile_handoff_core.sha256
```

包含：

```text
containers/starvla-profile/starvla.sqsh
tools/nsight-systems/downloads/NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143.rank0.nsys-rep
```

## 5. 接收方：拉取代码和 artifact

选择任意工作目录：

```bash
export WORKDIR=$PWD/qwenpi_zero2_profile_handoff
mkdir -p "$WORKDIR"
cd "$WORKDIR"
```

拉取 GitHub fork。推荐直接 clone 到 `$WORKDIR` 根目录；这个 branch 的 launcher
支持 starVLA fork 根目录布局：

```bash
git clone "$GITHUB_REPO" .
git checkout "$GITHUB_REF"
```

如果你选择把 starVLA clone 到 `code/starVLA`，也可以，但需要从同一个 branch
把 `scripts/`、`containers/`、`results/` 同步到 `$WORKDIR` 根目录：

```bash
mkdir -p code
git clone "$GITHUB_REPO" code/starVLA
git -C code/starVLA checkout "$GITHUB_REF"
cp -a code/starVLA/scripts code/starVLA/containers code/starVLA/results .
```

从 Hugging Face 下载 artifact 到 `$WORKDIR`，保持相对路径：

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

如果模型也同步在 HF artifact repo：

```bash
huggingface-cli download "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --local-dir "$WORKDIR" \
  --include "model/pretrained/Qwen3.5-4B/**"
```

如果模型不在 artifact repo，则从官方 Qwen repo 下载：

```bash
cd "$WORKDIR"
mkdir -p model/pretrained

huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir model/pretrained/Qwen3.5-4B \
  --local-dir-use-symlinks False
```

创建空 HF cache 目录即可，不需要同步原机器的 `model/huggingface`：

```bash
cd "$WORKDIR"
mkdir -p model/huggingface profiles
```

检查关键文件：

```bash
cd "$WORKDIR"

test -f containers/starvla-profile/starvla.sqsh
test -x scripts/starvla/profile_qwenpi_zero2_single_node.sh
test -f model/pretrained/Qwen3.5-4B/config.json
test -d data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot
test -x tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
```

校验核心 artifact：

```bash
cd "$WORKDIR"
sha256sum -c results/qwenpi_zero2_profile_handoff_core.sha256
```

## 6. 接收方：创建 enroot 容器

目标机器需要有：

- NVIDIA driver，可运行 CUDA 12.4 容器。
- 8 张可用 GPU；原始 profile 使用 8x H20。
- `enroot`。

创建容器：

```bash
cd "$WORKDIR"

enroot create --name starvla containers/starvla-profile/starvla.sqsh
```

如果机器上已经有名为 `starvla` 的 enroot 容器，可以换名字：

```bash
cd "$WORKDIR"

enroot create --name starvla-qwenpi containers/starvla-profile/starvla.sqsh
export ENROOT_NAME=starvla-qwenpi
```

快速验证容器和 Nsight：

```bash
cd "$WORKDIR"

ENROOT_MOUNT_HOME=no enroot start --rw \
  --mount "$WORKDIR/code:/code" \
  --mount "$WORKDIR/model:/model" \
  --mount "$WORKDIR/scripts:/scripts" \
  --mount "$WORKDIR/tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli:/opt/nvidia/nsight-systems" \
  "${ENROOT_NAME:-starvla}" bash -lc '
    cd /code/starVLA
    python --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    /opt/nvidia/nsight-systems/2026.3.1/target-linux-x64/nsys --version
  '
```

预期关键信息：

```text
Python 3.11.11
torch 2.6.0+cu124
torch cuda 12.4
NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0
```

## 7. 从头构建镜像

如果不使用 Hugging Face 上的 `starvla.sqsh`，可以在有网络的机器上从 base image
重建：

```bash
cd "$WORKDIR"

containers/starvla-profile/build.sh
```

构建脚本的 base image：

```text
docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

构建脚本会：

- `enroot import` base image。
- 创建/更新 enroot 容器 `starvla`。
- 在容器中对 `code/starVLA` 执行依赖安装。
- 安装 `starVLA` editable package。
- 导出 `containers/starvla-profile/starvla.sqsh`。

注意：从头构建需要能访问 PyPI/GitHub/容器 registry，并且 flash-attn 会在容器里构建。

## 8. 版本对齐

代码信息：

| 项 | 值 |
|---|---|
| upstream repo | `https://github.com/starVLA/starVLA.git` |
| branch | `starVLA_dev` |
| commit | `cdf5434438f4449cff85e3588956f7706a5c9cc3` |
| commit title | `[chore] Enhance citation details for StarVLA article (#377)` |

原始 profile 的宿主环境：

| 项 | 值 |
|---|---|
| GPU | 8x NVIDIA H20, 97,871 MiB |
| driver | 560.35.03 |
| enroot | 3.5.0 |

容器内版本：

| package | version |
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

Nsight Systems：

```text
tools/nsight-systems/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0
```

本地 checkpoint：

```text
model/pretrained/Qwen3.5-4B
```

该目录的 `README.md` 指向 HF repo `Qwen/Qwen3.5-4B`，并声明
`base_model: Qwen/Qwen3.5-4B-Base`。

## 9. 运行复现实验

在接收方机器上执行：

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

如果容器名不是 `starvla`：

```bash
cd "$WORKDIR"

ENROOT_NAME=starvla-qwenpi \
bash scripts/starvla/profile_qwenpi_zero2_single_node.sh \
  RUN_ID=qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143_rerun \
  PROFILE_RANKS=0 \
  NSYS_CAPTURE_MODE=none \
  MAX_STEPS=2 \
  PROFILE_START_STEP=1 \
  PROFILE_END_STEP=1
```

脚本实际执行的训练配置：

| 项 | 值 |
|---|---|
| accelerate config | `starVLA/config/deepseeds/deepspeed_zero2.yaml` |
| framework override | `--framework.name QwenPI` |
| backbone override | `--framework.qwenvl.base_vlm /model/pretrained/Qwen3.5-4B` |
| data root override | `--datasets.vla_data.data_root_dir /sv_data` |
| data mix override | `--datasets.vla_data.data_mix libero_goal` |
| per-rank batch | `--datasets.vla_data.per_device_batch_size 8` |
| full unfreeze | `--trainer.freeze_modules ''` |
| max steps | `--trainer.max_train_steps 2` |
| nsys trace | `cuda,nvtx,cublas,cudnn,osrt` |
| capture mode | `none`，从 rank0 进程启动开始 full trace |

输出目录：

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

成功退出时日志末尾应包含：

```text
accelerate_exit_status=0
rank_reports=1
launcher_exit_status=0
```

## 10. 查看已有 `.nsys-rep`

如果只需要查看这次已经生成的 profile，不需要重新训练。

用 Nsight Systems GUI 打开：

```text
profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143/
  qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143.rank0.nsys-rep
```

用 CLI 重新导出 stats：

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

原始 artifact 中确认有 CUDA kernel 数据。`cuda_gpu_kern_sum` 的 top entries：

| Time % | Kernel |
|---:|---|
| 44.6 | `at::native::vectorized_elementwise_kernel<..., FillFunctor<int>, ...>` |
| 7.6 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| 7.4 | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` |
| 4.1 | `merge_16x16_to_64x64_inverse_kernel` |
| 3.7 | `chunk_bwd_kernel_dqkwg` |
| 2.3 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |

原始 artifact 中可见的 NVTX ranges：

| Range | Instances | Total |
|---|---:|---:|
| `profile_window` | 1 | 260.39s |
| `train_step_1` | 1 | 260.39s |
| `DeepSpeedEngine.forward` | 2 | 165.00s |
| `model_forward` | 2 | 165.00s |
| `DeepSpeedEngine.backward` | 2 | 151.06s |
| `train_step_2` | 1 | 56.18s |

## 11. 离线 tar fallback

如果 GitHub/Hugging Face 不可用，可以从原始机器打包：

```bash
cd /home/chenchaox/project/phyai_fd

tar --zstd -cf /tmp/qwenpi_zero2_profile_handoff_offline_20260630.tar.zst \
  --exclude='code/starVLA/playground' \
  results/qwenpi_zero2_rank0_fulltrace_repro_README.md \
  results/qwenpi_zero2_profile_handoff_core.sha256 \
  scripts/starvla \
  containers/starvla-profile/build.sh \
  containers/starvla-profile/starvla.sqsh \
  tools/nsight-systems \
  code/starVLA \
  model/pretrained/Qwen3.5-4B \
  data/starvla_libero \
  profiles/qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143
```

接收方解包后从第 5 节的检查步骤继续。

## 12. 已知注意事项

1. 这份可用 artifact 使用 `NSYS_CAPTURE_MODE=none`，即 full trace rank0。
   之前尝试过 `NSYS_CAPTURE_MODE=cuda` 的 step-window 方式，训练能完成，但对这个
   DeepSpeed/QwenPI 路径生成的 `.nsys-rep` 没有 CUDA kernel 数据。
2. `scripts/starvla/nvtx_patch.py` 只打到了 step、dataloader、model_forward
   级别；还没有对 denoise loop 内部做细粒度 NVTX。
3. 这次 `cuda_api_sum` 中没有看到 `cudaGraph*`、`cudaStreamBeginCapture`、
   `cudaStreamEndCapture`、`cudaGraphLaunch`，因此原始 profile 里没有证据表明
   denoise 部分启用了 CUDA Graph。
4. 如果目标机器不是 8 卡，不能直接复现同样的 ZeRO-2 run shape；需要改
   `GPUS`、batch 和 DeepSpeed/accelerate 配置，得到的 profile 不再和本 artifact
   完全等价。
