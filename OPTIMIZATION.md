# QwenPI ZeRO-2 训练优化总览(OPTIMIZATION.md)

任务:starVLA QwenPI(Qwen3.5-4B VLM:24 线性注意力层 + 8 全注意力层,hidden 2560
+ 1024/24 层 vision tower + 2.96B 层间交叉 DiT 动作头;共 7.52B 参数全解冻;
DeepSpeed ZeRO-2 bf16,bs8/卡,seq pad 192,双 256² 图)在 8×H200 上优化训练吞吐,
目标 Agibot 的 200 samples/node/s。

**结果:800 → 288.5 ms/步(80 → ~222 samples/node/s,2.8×),超目标 11%。**

- 全部数字测于 viking H200 SXM(全 NVLink)节点;h200-nvl 4+4 拆分拓扑机型的数字
  不可与之互比(见 `docs/qwenpi_pending_experiments.md`)。
- 每项优化在代码中以 `[OPT #N]` 注释标注;完整实验阶梯见
  `docs/qwenpi_zero2_h200_optimization_log.md`。
- 采纳分档:**T0 零版本偏离**(#1-#12,415.9 ms / 154 samples/s)、
  **T1 需 torch 2.7.1 栈**(+#13-#17,288.5 ms / 222 samples/s)。

六类优化的吞吐乘数连乘 1.40 × 1.04 × 1.10 × 1.48 × 1.03 × 1.13 ≈ 2.77×,
与端到端一致。

## 优化项总览

| 类 | # | 优化项 | Δms | 档位 |
|---|---|---|---|---|
| **A 编译/CUDA Graph**(-128.6,25%) | 4a | DiT 编译 + pad192 静态形状 | -10.8 | T0 |
| | 6 | VLM 32 层逐层编译(绑定方法) | -15.9 | T0 |
| | 9 | max-autotune 编译模式 | -8.4 | T0 |
| | 10 | CUDA Graphs(reduce-overhead) | -26.1 | T0 |
| | 14 | 文本栈 8 层一组融合(FLA_TRACE) | -22.7 | T1 |
| | 15 | ViT 整塔单图 | **-44.7** | T1 |
| **B 流水线重叠**(-20.7,4%) | 3 | HF 预处理进 DataLoader worker | -6.4 | T0 |
| | 7 | HostPinnedBatch + 侧流 H2D | -12.6 | T0 |
| | 8 | 延迟参数 allgather | -1.7 | T0 |
| | 18 | 梯度装桶侧流化 | 0(中性,默认关) | 门控实验 |
| **C 主机同步消除**(-42.4,8%) | 2 | DS timer 去同步 + loss 读回门控 | -26.0 | T0 |
| | 12 | MRoPE 位置索引预计算 | -6.6 | T0 |
| | 17 | 免同步多模态 merge | -9.8 | T1 |
| **D 计算削减/选择**(-259.2,51%) | 1 | DiT fp32 → bf16 | **-230.9** | T0 |
| | 4b | 跳过 lm_head 大 GEMM(logits_to_keep=1) | -2.6 | T0 |
| | 5 | mixed attention(vision FA2 / 文本 SDPA) | -25.7 | T0 |
| **E 参数/通信调优**(-11.3,2%) | 11/16 | reduce_bucket_size → 1.5e9(方差 17.7→3.4) | -11.3 | T0 |
| | 专项 | 拆分拓扑机型 NCCL_P2P_LEVEL=NVL + MIN_NCHANNELS=16 | 该机型 -29% | 阶梯外 |
| **F 环境升级**(-48.2,9%) | 13 | torch 2.7.1 + NCCL 2.26 + triton 3.3 | -48.2 | T1 |

档位:**T0** = 零版本偏离,可直接合入(合计 415.9 ms / 154 samples/s);
**T1** = 依赖 torch 2.7.1 栈,需客户确认版本偏离(合计 288.5 ms / 222 samples/s);
**门控实验** = 已提交但默认关闭的否定/待验证结果。以下按类详述。

---

## A. 编译 / CUDA Graph 类 —— 攻击 kernel 发射开销

合计 **-128.6 ms,占总节省 25%,吞吐乘数 ×1.40**。

初始 profile 显示 GPU 空转 ~46%、每步数千次 `cudaLaunchKernel`、大量 <100µs 小
kernel——典型 overhead-bound(GPU 等 CPU 逐个下发)。本类的演进链:先能编(#4a/#6)
→ 编得好(#9/#10)→ 编得大(#14/#15)。

### #4a DiT 编译 + 静态形状(-10.8 ms)

`--framework.action_model.compile_dit true` 对 2.96B DiT 整体 `torch.compile`;
配套 `--framework.action_model.pad_encoder_seq_to 192` 和 collate 侧
`--datasets.vla_data.collate_pad_to 192`,把变长序列钉死到固定 192,消除
dynamo 重编译。静态形状是全类的前提:没有它,后面所有 CUDA Graph 都无法复用。

限制:pad 到 192 假设指令 + 图像 token 不超过 192(collate 目前无 truncation
护栏,评审已标记);改 batch size / 分辨率需要重新预热编译。

### #6 VLM 32 层逐层编译——绑定方法方式(-15.9 ms)

对文本栈每层做 `layer.forward = torch.compile(layer.forward)`(编译**绑定方法**),
而不是 `torch.compile(layer)`(包裹模块)。关键原因:transformers 5.3 通过
isinstance 匹配的 forward hook 收集 `output_hidden_states`(动作头需要逐层隐状态),
`torch.compile(module)` 会把模块换成 `OptimizedModule`,isinstance 匹配失效、
hook 静默不触发。编译绑定方法保留模块身份,hook 照常工作。

### #9 max-autotune 编译模式(-8.4 ms)

编译模式从默认提升到 `max-autotune(-no-cudagraphs)`:inductor 对逐点/规约 kernel
做模板搜索。注意:后续单卡微基准证明 max-autotune 的 **triton GEMM 模板在本模型
形状上打不过 cuBLAS**(45→70µs),收益全部来自 GEMM 周边 elementwise 的融合。
新容器首次编译 ~19 分钟(inductor 缓存在容器内持久)。

### #10 CUDA Graphs(reduce-overhead)(-26.1 ms)

编译模式换 `reduce-overhead`:编译区整体录制为 CUDA Graph,回放时发射成本≈0。
除均值收益外,**消除了偶发的 260-375ms 主机停顿**(最大 gap 从 375ms 降到 26ms)。

限制:要求区域内地址/形状完全静态;与 ZeRO-2 hook 的内存池交互脆弱(见 #14 的
崩溃史);bs 改变需重录。

### #14 文本栈 8 层一组融合(FLA_TRACE)(-22.7 ms)

把 32 层文本栈按 8 层一组编成 4 张大 CUDA Graph(`STARVLA_FUSED_TEXT_STACK=1
STARVLA_FUSED_GROUP_SIZE=8 STARVLA_FLA_TRACE=1`),消掉逐层编译遗留的 92+45 ms/步
图间胶水。前提是让 flash-linear-attention 可被 dynamo 完整追踪:

1. 手工注释掉 `fla/ops/gated_delta_rule/chunk.py:220` 的 `@torch.compiler.disable`
   (容器内改动,setup_node.sh 自动执行,.bak 保留);
2. 常量折叠 `check_shared_mem` / `get_multiprocessor_count`(host 侧设备属性查询,
   dynamo 无法追踪,但对固定硬件是常量);
3. 线性注意力层强制 `causal_conv1d_fn=None`,走 torch 原生 conv 回退(自定义算子
   以非连续 out= 调用无法入图;数值等价,代价 ~10.5ms/步,包回 custom_op 是待办)。

历史教训:在此之前 fused stack 在 torch 2.6 和 2.7 上都因 fla graph break ×
cudagraph_trees × ZeRO-2 hook 崩溃(`_cuda_setCheckpointPoolState`);FLA_TRACE
使每组成为无断点单图后问题消失。**整栈单图反而更慢**(-55ms):单体 backward 把
所有梯度桶推迟到图尾,杀死通信重叠——组大小≈桶数量是甜点。

### #15 ViT 整塔单图(-44.7 ms,单项最大编译类收益)

24 个 vision block + FA2 varlen 编成**一张** reduce-overhead 图
(`STARVLA_FUSED_VISION=1`)。历史上"vision 不可编译"的两个判词都是 host 侧常量:

1. HF 把 `max_seqlen` 算成 GPU 0-维张量,flash-attn 自定义算子要 SymInt →
   按 `grid_thw` 字节缓存 host int,盖章到 attention 模块上;
2. transformers 的 `lazy_import_flash_attention` 在调用时 importlib → 预热后
   折叠成字典查找。

依赖 torch 2.7.1 + flash_attn 2.8.0.post2(2.6 + 2.7.4 组合不行)。

---

## B. 流水线重叠类 —— 攻击引擎空转(计算/拷贝/通信引擎互等)

合计 **-20.7 ms,占 4%,×1.04**。

### #3 HF 预处理进 DataLoader worker(-6.4 ms)

`--datasets.vla_data.preprocess_in_collate true --datasets.vla_data.num_workers 8`:
图像预处理、tokenization、chat template 从训练循环移到 collate(worker 进程),
与上一步 GPU 计算重叠。限制:改变了 batch 的类型约定(list[dict] → dict),
eval/predict 路径尚未适配(评审已标记)。

### #7 HostPinnedBatch + 侧流 H2D(-12.6 ms)

`--datasets.vla_data.collate_host_batch true`:collate 产出 pinned-memory 批,
在专用 CUDA stream 上异步 H2D,与上一步的优化器尾部重叠。milestone4 分流账本里
该侧流每步仅 0.3ms 且完全被遮盖——重叠做到位的样子。

### #8 延迟参数 allgather(-1.7 ms)

`STARVLA_DEFER_AG=1`:ZeRO-2 优化器步后的参数重收(allgather)按参数组拆分——
VLM 主干(4.54B)组同步等待,动作头(2.98B)组延迟到动作头 forward 处再等,
让 AG 与 VLM forward 重叠。限制:等待点只补在动作头 forward,eval/存档路径
存在竞态(评审已标记,须在 predict/save 前补 wait)。

### #18 梯度装桶侧流化(0 ms,中性,默认关)

`STARVLA_GRAD_COPY_STREAM=1`:把 ZeRO-2 contiguous_gradients 的逐参数 D2D 拷贝
推迟并用 `torch._foreach_copy_` 批量在侧流执行。实测中性——根因是每步 445 次
python hook(~25µs/次)才是瓶颈,不是拷贝本身。保留为文档化的否定结果,
并由此催生 A2(compiled autograd,在途)。

---

## C. 主机同步消除类 —— 攻击 CPU↔GPU 往返串行化

合计 **-42.4 ms,占 8%,×1.10**。共同指纹:几十字节的 D2H + 每步几百次往返。

### #2 DeepSpeed timer 去同步 + loss 读回门控(-26.0 ms)

ds_config `timers.throughput.synchronized=false`(默认每步 `cuda.synchronize`
计时);`loss.item()` 等日志读回按 `LOGGING_FREQ` 门控,不再每步同步。

### #12 MRoPE 位置索引预计算(-6.6 ms)

`--datasets.vla_data.collate_mrope_posids true`:HF 的 `compute_3d_position_ids`
每步在 GPU 路径上做每图 3 次 `.item()` + 每行 `.tolist()`(实测 13ms 主机受限)。
改为在 collate(CPU worker)用**无权重模型 shim**(`Qwen3_5Model.__new__` + config,
不加载权重)精确复刻该计算,产出 (3,B,T) `position_ids` 随批下发;按样本精确键
(token 类型行、mask 行、grid 字节)缓存。正确性:与 HF 批量计算逐位一致的单测 +
`STARVLA_CHECK_POSIDS=1` 8 rank 在体校验。限制:带 `model_type=='qwen3_5'` 门——
Qwen2.5-VL 的 mrope 语义不同(time_interval=4),误用会静默出错。

### #17 免同步多模态 merge(-9.8 ms)

`STARVLA_FAST_MM_MERGE=1`。kernel 级 nsys 揪出 vision 图与首个文本图之间 16ms
的空隙 = ~150 次十几字节的 D2H 往返:`get_placeholder_mask` 的占位符计数校验
(DeviceSelect + `bool()`)、`_update_linear_attn_mask` 的 `torch.all`、图内 MRoPE。
全部换成免校验/直通路径。代价:占位符计数不匹配从报错变成静默错位散射——
新数据管线先用 stock 路径跑一次校验再开启。

---

## D. 计算削减 / 选择类 —— 攻击不必要的计算

合计 **-259.2 ms,占 51%,×1.48**。占总节省一半,全部来自"删错的",不是"算得快"。

### #1 DiT fp32 → bf16(-230.9 ms,全程最大单项)

2.96B 的 DiT 动作头原先整体跑 fp32(前向+反向,且 repeated_diffusion_steps=2
双倍执行)。以 `torch.autocast("cuda", dtype=torch.bfloat16)` 包裹 DiT 调用:
字节减半 + H200 tensor core bf16 吞吐约为 fp32 的 2 倍以上。梯度/优化器状态
仍按 DeepSpeed bf16 方案管理,loss 曲线与 fp32 基线一致。

### #4b 跳过 lm_head 大 GEMM(-2.6 ms)

QwenPI 的动作损失只消费 VLM 的**逐层隐状态**(`output_hidden_states=True`),
从不使用语言模型 logits;但 HF forward 默认对全序列算 lm_head——一个
[B×192, 2560] × [2560, ~152k 词表] 的大 GEMM + 后续 float 转换,结果直接被丢弃。
在接口层注入 `logits_to_keep=1`,lm_head 只算最后 1 个位置(HF 无"完全跳过"开关,
1 行是最小合法值),砍掉 192/193 的词表投影计算。限制:该注入在共享接口上是
无条件的,改变了兄弟框架(LangForce)拿全序列 logits 的约定——评审已标记,
需改成 QwenPI 专属。

### #5 mixed attention:vision FA2 / 文本 SDPA(-25.7 ms)

`--framework.qwenvl.attn_implementation mixed`,展开为
`{"vision_config": "flash_attention_2", "text_config": "sdpa"}`:

- **vision tower 用 FA2**:图像 patch 序列走 varlen 路径,FA2 明显快;
- **文本栈用 SDPA**:反直觉的实测结论——seq~190 时 FA2 的 unpad/repad 开销
  超过其内核收益,全开 FA2 反而整体**更慢**;SDPA(cutlass 内存高效后端)在
  短序列上胜出。

教训:注意力实现的选择是形状函数,不是"新即是好"。限制:无 flash-attn 的环境
下 dict 形式绕过了现有的 sdpa 回退逻辑(评审已标记)。

---

## E. 参数 / 通信调优类 —— 攻击配置与硬件的错配

合计 **-11.3 ms,占 2%,×1.03**(另有拆分拓扑机型专项,不计入 viking 阶梯)。

### #11/#16 reduce_bucket_size 5e8 → 1e9 → 1.5e9(-11.3 ms,方差 17.7 → 3.4 ms)

ZeRO-2 的梯度按桶 allreduce。5e8 产生过多小 AR;1e9 收 -9.1ms;**1.5e9 的依据是
读 DeepSpeed `stage_1_and_2.py` 源码**:contiguous_gradients + overlap_comm 下
ipg 缓冲是 2 × bucket 的双缓冲乒乓,1.5e9 使 DiT 的梯度恰好占满两个缓冲,
均值 -2ms 但**步时方差从 ±17.7 收敛到 ±3.4 ms**(A/B 测量的信噪比价值大于均值)。
限制:显存代价 2×1.5e9×2B = 6GB/卡;当前改在共享 ds_config 上会影响仓库其他
配方(评审已标记,应移为 QwenPI 配方级覆盖)。已否定:2e9 更差;NVLS、
use_multi_rank_bucket_allreduce=false 均回退。

### (专项)拆分拓扑机型 NCCL 调优(该机型 -29%,不计入主阶梯)

h200-nvl 4+4 机型(quad 内 NVLink、quad 间 SYS):必须 `NCCL_P2P_LEVEL=NVL`,
否则首个集合通信静默挂死(30 分钟 watchdog SIGABRT);跨 quad 走 SHM 传输时
默认 4 通道严重饥饿,**`NCCL_MIN_NCHANNELS=16` 单变量 -29%**(865→618ms);
32 通道收益收敛,加大 BUFFSIZE 无贡献,`allreduce:tree` 中性(全局 `Tree`
会崩:Broadcast 无 Tree 实现)。

---

## F. 环境升级类 —— 软件栈本身的红利

合计 **-48.2 ms,占 9%,×1.13**。

### #13 torch 2.7.1 + NCCL 2.26.2 + triton 3.3.1(-48.2 ms)

纯升级、零代码改动就有 -13%(415.9→367.7):triton 3.3 为 fla 生成 TMA kernel、
inductor 代码生成改进、NCCL 2.21.5→2.26.2。配套:flash_attn 2.8.0.post2 与
causal-conv1d 1.5.2 的 cxx11abiTRUE-cp311 预编译 wheel(精确 URL 固化在
`scripts/starvla/setup_node.sh`,`TORCH27=1` 一键安装)。

**这是唯一偏离 Agibot 版本钉的类**,同时是 #14/#15(融合大图)的解锁前提——
torch 2.7 偏离锁着 -55ms 直接收益 + 融合图的约 -67ms,合计约 -120ms,
是与客户谈版本升级时的核心筹码。容器随 SLURM job 消失,升级需每 job 重做
(setup_node.sh 约 10 分钟无人值守)。

---

## MFU(Model FLOPs Utilization)现状

**FLOPs/步/卡**(两个独立来源交叉验证,相差 6%):

| 方法 | 前向 | 训练总量(×3:fwd 1 + bwd 2) |
|---|---|---|
| 解析(逐组件 6PT:文本 33.1T + vision 7.7T + DiT 16.1T) | 19.0T | 56.9 TFLOPs |
| DeepSpeed flops profiler 实测 | 17.85T | 53.6 TFLOPs |

按 H200 SXM bf16 dense 峰值 989.5 TFLOPS:

| 配置 | 步时 | MFU |
|---|---|---|
| 初始状态 | ~800 ms | ~7% |
| T0 零偏离(#1-#12) | 415.9 ms | ~13% |
| **T1 最优(#1-#17)** | **288.5 ms** | **~19%(18.8-19.9%)** |
| bs16(旧栈参考点) | 573.3 ms | ~19% |

两因子分解:MFU = kernel 效率(算的时候多快,~27% of peak)× 计算时间占用率
(多少墙钟在算,~70%)。**同形状(bs8/seq192/全解冻)的物理天花板 ≈ 35-40%**:
消掉全部裸露通信与空转到 ~27%,其余受限于小 M GEMM 形状(单卡实测:FFN 在
M=1536 已达机器可持续吞吐的 100%,受功率墙而非形状墙;持续负载下 SM 降频至
~76% boost)。与 LLM 预训练的 40-60% 不可直接对标——那是"大 token 量形状红利"
(计算通信比高两个数量级 + 大 M GEMM),对 VLA 的批量/序列语义不成立。
**对外汇报建议以 samples/node/s 为主指标(222 vs 目标 200),MFU 附形状说明。**

墙钟收支表(milestone4 step 40,分流账本,`tools/stream_budget.py`):

```
墙钟 307.4 ms(nsys 下;干净运行 288.5)
├─ 计算侧流忙        213.3 ms  69.4%
├─ 裸露通信(NCCL 忙 105.6,其中 43.7 被计算遮盖)  61.9 ms  20.1%
└─ 真空转            32.2 ms  10.5%
```

剩余路线:工程收尾(A2 compiled autograd ~11ms、A6 DiT 去冗余 ~3-4ms、
fla conv custom_op ~5ms)→ ~250ms 地板;突破地板需语义级(bs16、1-bit Adam、
fp8、冻结 vision),实验卡见 `docs/qwenpi_pending_experiments.md`。
