# QwenPI ZeRO-2 @ 8×H200：80 → 151 samples/s 优化全记录 与 NCCL 瓶颈分析报告

日期：2026-07-04。硬件：viking-cr-196，8× NVIDIA H200 141GB（NVSwitch NV18，
450 GB/s/方向），SLURM + enroot。任务形态：starVLA QwenPI（Qwen3.5-4B VLM +
2.96B 参数 layer-wise cross-DiT action head，共 7.524B 全解冻），DeepSpeed
ZeRO-2 bf16，per-GPU batch 8（global 64），seq ~181-192，2 输入图像。

计量口径：无 nsys 的 100-step 训练，steps 21–100 的 wall-clock 回归
（`tools/bench_report.py`）。samples/s = 64 / step_time。

---

## 一、结果总览

| 阶段 | ms/step | samples/node/s | 相对提升 |
|---|---:|---:|---:|
| 原始（DiT 在 fp32 autocast 下） | ~800 | **~80** | — |
| + commit `a5e6f3b`（DiT 转 bf16） | 569.1 | 112.5 | +41% |
| + 本次优化战役（10 项，见下） | **423.8** | **151.0** | +89% 累计 |
| （bs16 数据点，global batch 128） | 560.0 | **228.6** | +186% 累计 |

端到端 MFU：~7% → 12.4%（bs8）/ 18.7%（bs16）。注意：计算 kernel 的
**瞬时 MFU 已达 ~40%**（129 ms 计算完成 ~52 TFLOP），端到端数字被通信与
空隙稀释——这一区分是理解第三节 NCCL 瓶颈的关键。

---

## 二、全部修改清单：11 项优化瀑布

每行均为该项落地后的独立 100-step 实测（wall-clock 回归，steps 21–100）；
"较前一项提升"为 samples/s 的相对增幅。

| # | 优化项 | ms/step | samples/s | Δ ms | 较前一项提升 |
|---|---|---:|---:|---:|---:|
| 0 | 原始状态（DiT 在 fp32 autocast 下运行） | ~800 | ~80 | — | — |
| 1 | **DiT transformer 转 bf16**（commit `a5e6f3b`） | 569.1 | 112.5 | ~−231 | **+40.6%** |
| 2 | DeepSpeed ThroughputTimer 去同步 + `loss.item()` 门控 | 543.1 | 117.8 | −26.0 | +4.7% |
| 3 | HF 预处理移入 DataLoader worker（collate） | 536.7 | 119.2 | −6.4* | +1.2% |
| 4 | torch.compile DiT + encoder pad 192（含 lm_head 跳过、attn 配置修复） | 523.3 | 122.3 | −13.4 | +2.6% |
| 5 | mixed attention（vision FA2 varlen 批处理，text 保持 SDPA） | 497.6 | 128.6 | −25.7 | +5.2% |
| 6 | torch.compile VLM 文本栈 32 层（`layer.forward` 绑定方法） | 481.7 | 132.9 | −15.9 | +3.3% |
| 7 | HostPinnedBatch + 专用 copy stream H2D | 469.1 | 136.4 | −12.6 | +2.6% |
| 8 | 延迟 action-head 参数组 AllGather | 467.4 | 136.9 | −1.7 | +0.4% |
| 9 | compile 模式升级 max-autotune | 459.0 | 139.4 | −8.4 | +1.8% |
| 10 | **CUDA Graphs**（`reduce-overhead`，替换第 9 项的模式） | 432.9 | 147.9 | −26.1 | **+6.1%** |
| 11 | `reduce_bucket_size` 5e8 → 1e9 | **423.8** | **151.0** | −9.1 | +2.1% |
| — | （bs16 数据点，global batch 128，改训练语义） | 560.0 | 228.6 | — | +51.4% vs #11 |

\* 第 3 项的真实收益 ~40ms（forward 内 CPU 前导消除），当时被 accelerate
`send_to_device` 阻塞在优化器尾部上的等待（+52ms `timing/data`）部分抵消，
第 7 项最终将其兑现。

每一项修改的 loss 轨迹均与基线逐一比对（step-100 loss 0.60–0.68 区间内，
差异与 dataloader 顺序变化一致），语义不变。除 2/4 中的纯收益子项外全部
opt-in（flag/env 门控），默认训练路径不受影响。

### 2.1 各项机制与涉及文件

**#1 DiT fp32 → bf16（`LayerwiseFM_ActionHeader.py`，commit `a5e6f3b`）**
QwenPI 框架有意用 `torch.autocast(dtype=float32)` 包住整个 flow-matching
action head（噪声/速度插值和 loss 归约需要 fp32）。副作用是 2.96B 参数、
32 层 DiT 的全部 Linear/Attention matmul 被压到 **fp32 SIMT/CUTLASS
路径**——当时的 nsys 显示稳态 GPU 时间的 ~44–50% 消耗在 fp32 GEMM 上。
修改：不动外层 fp32 语义，只在 `self.model(...)`（DiT transformer 调用）
外再套一层 bf16 autocast；LayerNorm 依 autocast 白名单留在 fp32，
TimestepEncoder 自带 dtype 转换，数值安全。`forward` /
`predict_action` / `predict_action_realtime` 三个入口同步修改。
这是全部 11 项中单项收益最大的一笔（+40.6%），也印证了本报告的主线：
让计算走 tensor core、让 CPU 与通信别挡路。

**#2 每步同步消除（`ds_config.yaml`、`train_starvla.py`）**
`timers.throughput.synchronized=false` 消除 DeepSpeed ThroughputTimer
每步 53.8ms 的 `cudaDeviceSynchronize`；loss 张量保持在 GPU，仅在
logging 步 `.item()`。

**#3 预处理下移（`lerobot_datasets.py` `QwenVLPreprocessCollate`、
`dataloader/__init__.py`、`QwenPI.py` 预 collate 分支）**
图像 resize/normalize/patchify + chat-template tokenize 原先在
`model.forward` 内的主线程执行（GPU 空转 41ms/step）；移入 DataLoader
worker，pixel_values 在 collate 内转 bf16，actions 堆叠为张量。

**#4 DiT 编译 + 静态形状（`LayerwiseFM_ActionHeader.py` `compile_dit`、
`QwenPI.py` `pad_encoder_seq_to`）** encoder 序列 pad 到固定 192（pad 位
被 mask，数值不变）使 DiT 形状全静态；同时跳过无用的 248K-vocab lm_head
logits（`logits_to_keep=1`，省 2.6ms + 719MB）并修复 `QWen3_5.py` 硬编码
sdpa 覆盖配置的 bug。

**#5 mixed attention（`QWen3_5.py`）** vision tower 走 FA2 varlen——
16 张图一次 batched kernel，替代逐图 16 次串行 SDPA；文本栈保持 SDPA
（seq~190 下全量 FA2 的 unpad/repad 开销反亏 ~10ms）。

**#6 VLM 文本栈编译（`QWen3_5.py` `compile_language_model`）**
关键发现：transformers 5.3 用 `isinstance(module, DecoderLayer)` 匹配的
forward hook 收集 hidden states，`torch.compile(module)` 替换为
OptimizedModule 后 hook 失配、hidden_states 被静默截断（QwenPI 需要全部
32 层）。解法：编译**绑定方法** `layer.forward`——模块身份不变，hook 在
编译区外照常触发。无需升级 transformers。

**#7 HostPinnedBatch（`lerobot_datasets.py`、`QwenPI.py`）**
collate 返回一个 accelerate `send_to_device` 会放行（无 `.to`、非
Mapping）但 DataLoader pin 线程仍会 pin（自定义 `pin_memory()`）的容器；
`QwenPI.forward` 在专用 copy stream 上做 H2D，与 ZeRO-2 优化器尾部
（allgather+adam ~53ms）重叠，`timing/data` 从 52ms 归零。

**#8 延迟 AllGather（`scripts/starvla/ds_defer_allgather_patch.py`，
`STARVLA_DEFER_AG=1`）** DiT 参数组（2.98B）的 post-step AllGather 改
async 发射，`wait()` 推迟到 head forward 入口，藏进 VLM forward；按
numel 选组（最大组=backbone 保持同步）防止误延。

**#9→#10 max-autotune → CUDA Graphs（`compile_mode`）**
`reduce-overhead`（inductor cudagraph）与 DeepSpeed ZeRO-2 的
hook/捕获冲突实测未发生；launch 从 ~11k/step 降到 ~5k + 176 次
`cudaGraphLaunch`；**顺带根治了间歇性 260–375ms 的 host stall**（问题
enqueue 路径被 graph replay 取代）。一次性成本：新容器首次 step1≈190s
JIT + step2≈176s 捕获。

**#11 梯度桶调优（`ds_config.yaml`）** AllReduce 15 桶（1GB）→ 8 桶
（2GB），busbw 更高、排队更少；2e9（4 桶）实测回退，1e9 为甜点。

### 2.2 实测否决的方向（避免重复投入）

NCCL_PROTO/NVLS/channels/buffsize/CGA 全套（见第三节，线速墙）、
`use_multi_rank_bucket_allreduce=false`（通信主导格局下单独实测回退
67.9ms；早期一次与 NVLS 混合的测试亦回退）、
全量 FA2（+10ms）、accelerate `non_blocking` dataloader（+60ms，读回排队
问题）、vision tower compile（flash-attn 算子 SymInt 签名 vs HF 张量
max_seqlen，dynamo 硬错误）、手写算子（瓶颈在编排不在 kernel）。

---

## 三、nsys 性能分析：为什么 NCCL 现在是瓶颈

三份 rank0 全量 trace（均为 MAX_STEPS=100，稳态取 steps 21–100）：

```text
profiles/qwenpi_zero2_h200_rank0_step100_fulltrace_20260703/   基线（优化前）
profiles/qwenpi_milestone1/                                    第 6 项后（compile，无 cudagraph）
profiles/qwenpi_milestone2/                                    第 10 项后（cudagraph，当前最优-1）
```

### 3.1 核心机制：计算被压缩，通信是常数

ZeRO-2 全解冻下，每步每 rank 的通信量只由**参数量**决定，与任何计算
优化无关：梯度 AllReduce 15.05 GB（bf16 × 7.524B）+ post-step 参数
AllGather 15.05 GB ≈ **30 GB/step 恒定**。战役中计算侧被三轮压缩，
通信侧纹丝不动——NCCL 占比机械上升：

| 指标（nsys 稳态，ms/step） | 基线 trace | milestone2 | 变化 |
|---|---:|---:|---|
| 步 wall（traced） | 693.6 | 444.9 | −36% |
| 计算 kernel busy | 250.4 | **129.1** | **−48%** |
| NCCL kernel busy | 117.2 | 127.0 | ≈ 常数 |
| NCCL 占（计算+NCCL）时间比例 | 32% | **50%** | 瓶颈易主 |
| backward 段 wall | 367.5 | 242.1 | −34% |
| forward 段 wall | 238.7 | 173.2 | −27% |
| kernel launch 次数 | ~12,500 | ~5,000 + 176 graph | −60% |

milestone2 中：AllReduce 82.2 ms/step（15×5.48ms，捕获时桶为 5e8）+
AllGather 44.8 ms（2×22.7ms）= NCCL busy 127 ms，其中与计算 kernel
无重叠的"暴露"部分 101 ms（22.8% of wall）。backward 段 NCCL 占其
wall 的 33.9%，forward 段 24.6%（延迟的 DiT 组 AllGather 藏在这里）。

### 3.2 暴露的两个具体机制

**（a）backward 短于通信流水的自然跨度。** backward wall 已被压到
242 ms，而 NCCL 流上串行的梯度 AllReduce 需要 82 ms 且首个桶要等
~30% 的 backward 后才有梯度可发——最后几个桶完成时 backward 计算早已
结束，**尾部桶完全没有重叠对象**。计算越快，可供藏通信的"画布"越小：
这是压缩计算的直接代价，不是失误。

**（b）rank 间到达抖动。** 隔离微基准中 1GB 桶 AllReduce 仅 3.81 ms
（460 GB/s busbw），训练内同尺寸桶中位 5.48 ms（+44%）：AllReduce
kernel 必须等最慢 rank 到达，8 个 SPMD 进程的每桶抖动直接计入 kernel
时长。消灭巨型 stall（第 9 项）后，残余抖动为毫秒级、SPMD 固有。

### 3.3 算法侧已到硬件墙（Round-7 扫参，`scripts/starvla/nccl_microbench.py`）

与训练同形的集合通信微基准，9 组 NCCL 配置（2026-07-04 实测；复跑需先
重建容器：`enroot create --name starvla containers/starvla-profile/starvla.sqsh`
—— enroot 数据目录随 SLURM job 轮换清空）：

| 配置 | AllReduce 2GB busbw | AllGather 9.08GB busbw |
|---|---:|---:|
| baseline（NCCL 2.21.5 auto） | **472.4 GB/s** | 358.1 GB/s |
| NCCL_ALGO=NVLS | 472.7（零差异） | 357.4 |
| MIN_NCHANNELS 24/32 | ≈ | 358.5 / 365.3 |
| BUFFSIZE 8M/16M、CGA 2 | 零差异或更差 | — |

AllReduce busbw **472 GB/s 已超过 450 GB/s 标称线速**——auto-tuner
即最优，无任何算法/协议/缓冲旋钮可动（chan32 的隔离 +2% 在训练内因
SM 争抢回退，实测 429.2 vs 423.8）。由此得**通信 wire 下限**：

```text
AllReduce  8 桶 × 7.41 ms          = 59.3 ms
AllGather  22.2 + 14.6             = 36.8 ms
──────────────────────────────────────────────
wire floor                          ≈ 96 ms/step   （bs8, ZeRO-2 全解冻 7.52B）
```

当前 423.8 ms 中约 23% 是不可压缩的通信物理时间。

### 3.4 Amdahl 收束

bs8 下的工程极限估算：计算 129 ms（瞬时 MFU 已 ~40%）+ 通信不可重叠
部分（59–96 ms，取决于重叠工程）+ 边界/残余 ≈ **250–300 ms/step
（213–256 samples/s）**。继续降低通信占比只剩结构性手段：

1. **加大 batch**（已验证：bs16 → 228.6 samples/s，通信/参数量不变而
   计算翻倍，通信占比自动稀释）；
2. 梯度累积（通信按优化器步摊薄，改训练语义）；
3. 部分冻结 / DiT head 瘦身（2.96B 参数占 39% 通信量，仅服务 40 个
   query token——模型级决定）；
4. 梯度压缩（当前栈不支持）。

---

## 四、附录

- 逐轮实验记录与全部 A/B 数据：`docs/qwenpi_zero2_h200_optimization_log.md`
- 稳态分析方法与基线剖析：`docs/qwenpi_zero2_h200_steady_state_profile.md`
- 分析工具：`tools/analyze_steady_state.py`、`tools/analyze_gaps.py`、
  `tools/bench_report.py`；通信体检：`scripts/starvla/nccl_microbench.py` +
  `nccl_sweep.sh`
- 最优配置复现命令：见 optimization log Round 5/6（`bench_U` 配置）
- 已知限制：transformers 5.3 的 hidden-states capture 机制阻止文本栈
  合并为更少的编译图（graph copy-in ~45ms CPU/step 的根源），是升级
  transformers 如今唯一的实质动机。
