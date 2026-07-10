# QwenPI 待验证实验队列(按机会窗口大小分档)

机器紧张时的原则:**想法不需要"等到机器"才有进度——把每个想法固化成一张
可执行的实验卡(假设/命令/时长/判定规则),窗口一出现就消耗队列**。
新节点先跑 `bash scripts/starvla/setup_node.sh`(需 torch 2.7 的实验加
`TORCH27=1`),脚本会自动检测 4+4 拓扑并打印该节点必需的 NCCL 环境。

方法论要点:**吞吐类 A/B 不需要 viking**——同一节点上跑 A 和 B,相对增益
在任何 8×H200 上都成立(NVL 4+4 节点的绝对值不可与 viking 对比,但 Δ% 可信)。
判定一律用 40 步、掐头 5 步的 p50 步时;同节点噪声 ±2 ms(bucket 1.5e9 之后
方差 3.4 ms,见优化日志 Round 8)。

基线参照(viking, bs8):committed 基线 423.8 ms;best config 288.5 ms。

## 第 0 档:零 GPU(随时可做)

| 卡片 | 内容 | 产出 |
|---|---|---|
| Z1 | 离线写好各实验补丁(下表 A2/A3 的代码),按既有模式 env-flag 门控 | 窗口期零编码 |
| Z2 | 继续挖掘 milestone2/3/4 sqlite(本地 + HF 都有)——还没做的:backward 逐图时间分解、优化器阶段逐 kernel 预算 | 新瓶颈假设 |
| Z3 | 语义级提案文档化(bs16/1-bit Adam/fp8/冻结 vision),给 Agibot 附收敛验证协议 | 客户可自行验证 |

## 第 1 档:任意 8×H200,10-20 分钟/张(同节点 A/B,共约 2 小时清空)

优先级从高到低;A = 对照组(best config),B = 加一个变量。

| 卡片 | 假设 | 做法 | 预期 | 判定 |
|---|---|---|---|---|
| A1 **bs16 新栈吞吐** | 旧栈 bs16 已超线性(228.6 samples/s),新栈(#12-#17)下应更高 | best config + `--datasets.vla_data.per_device_batch_size 16`,注意 pad/graph 需重编译,先 MAX_STEPS=40 试跑 | >260 samples/s | 只报吞吐,收敛归第 2 档 |
| A2 **compiled autograd** | 445 次/步的 python hook 突发(~11 ms)是 #18 测得中性的根因;compiled autograd 把 hook 编进反向图 | 补丁:`torch._dynamo.config.compiled_autograd=True` 或 backward 包 `compiled_autograd.enable()`;需 torch 2.7;与 DS ZeRO-2 hook 兼容性是主要风险 | -8~-11 ms | p50 差 >4 ms 且 loss 曲线逐位一致 |
| A3 **QKV/gate-up 横向融合** | M=1536 的小 GEMM 效率 ~57%,加载时 concat 权重减少 kernel 数、增大单 GEMM | 加载后权重手术 + forward 改写,fused 组内替换 | -5~-15 ms | 同上 |
| A4 **max-autotune 全栈** | 此前仅 DiT 用了 max-autotune;fused 文本/vision 组用的 reduce-overhead | `STARVLA_FUSED_COMPILE_MODE=max-autotune`(纯 env,零代码;编译时间 +10 min) | -3~-8 ms | 同上 |
| A5 **#18 复测 on A2** | grad-copy 侧流在 hook 开销消失后可能从中性变正 | A2 之上 `STARVLA_GRAD_COPY_STREAM=1` | -3~-6 ms | 同上 |

## 第 2 档:多日训练资源(收敛验证——本就该跑在 Agibot 训练集群上)

这一档**从来不适合我们的 profiling 窗口**,机器紧张不改变其归属:我们交付
实验卡,客户在正式训练集群上跑全程收敛曲线。

| 卡片 | 内容 | 解锁 |
|---|---|---|
| C1 | bs16(全局 128)全程收敛对比(可能需 lr 重标定);临界批量判定 | +30% 吞吐(A1 的数) |
| C2 | 1-bit Adam(DeepSpeed 原生配置)收敛对比 | 预估 -40 ms(通信墙) |
| C3 | 冻结 vision tower 收敛对比(VLA 常见配置) | ~-50 ms 计算 + 0.3B 梯度通信 |

## 已消耗窗口:2026-07-06/07,smc-522ga-0029(H200 NVL 4+4,torch 2.7 栈)

本机通信是关键路径(跨 quad 走 SHM,ring 里 20/92 条链路),因此:计算侧卡
(A2/A4)在此测不出信号,通信侧卡信号极强。40 步、掐头 5 步 p50,方差 ±5 ms:

| run | 配置(均含 best config + NCCL_P2P_LEVEL=NVL) | p50 | samples/node/s |
|---|---|---|---|
| nvl_base | bs8 | 865 ms | 74 |
| nvl_tree | + `NCCL_ALGO='allreduce:tree'` | 858 ms | 中性(注:全局 `NCCL_ALGO=Tree` 会崩,Broadcast 无 Tree 实现) |
| nvl_chan16only | + `NCCL_MIN_NCHANNELS=16` | **618 ms** | 104(**-29%**,SHM 默认 4 通道饥饿) |
| nvl_chan | + 16ch + `NCCL_BUFFSIZE=16MB` | 626 ms | buffsize 无贡献 |
| nvl_chan32 | + 32ch | 612 ms | 收益已收敛,16 是甜点 |
| nvl_bs16 | bs16 | 925 ms | 138(batch 翻倍仅 +7% 步时;**无 OOM**,A1 显存风险排除) |
| nvl_bs16_chan | bs16 + 16ch | **687 ms** | **186** |

结论:(1) 拆分拓扑机型的标配环境 = `NCCL_P2P_LEVEL=NVL NCCL_MIN_NCHANNELS=16`;
(2) A1 的吞吐半张卡已消耗(无 OOM + 通信摊薄实证),收敛半张归第 2 档 C1;
(3) A2/A4 等 viking 窗口。viking 全 NVLink 通道数是否同样欠配值得一测
(那边通信本就基本被藏住,预期增益小,但零成本)。

### 单卡 GEMM 微基准(2026-07-07,拓扑无关,scratchpad/gemm_fusion_bench.py)

GEMM 效率是单卡属性,不需要全 NVLink 机器即可验证。bf16、cuBLAS(nvjet)、
文本栈精确形状(hidden 2560, QKV N=2048/512/512, MLP N=9216):

| 形状 | 分开 | 融合 | 结论 |
|---|---|---|---|
| QKV @ M=1536 | 44.8µs (54.5%) | 34.8µs (70.1%) | **1.29×,真实但小** |
| gate+up @ M=1536 | 227.4µs (64.4%) | 240.1µs (61.0%) | **融合反而变慢**(N 已够大) |
| QKV @ M=3072 (bs16) | 94.9µs (51.4%) | 79.5µs (61.4%) | 1.19× |
| 参考 M=16384 | — | 63.1% | **K=2560 封顶效率 ~65%**,大 M 也救不了 |
| QKV max-autotune | 70.1µs | 62.4µs | **triton 打不过 cuBLAS,负收益** |

- **A3 降级**:只有 QKV 值得融(且代码核实:vision QKV 与 fla in_proj_qkv 已原生融合,
  分开的只有 8 个全注意力层 → 真实收益 ~0.3ms,低于噪声,不做);
- **A4 GEMM 半张卡否决**:autotune 的 triton GEMM 模板在这些形状全面落后 nvjet;
  A4 若有收益只能来自 GEMM 周边 elementwise 的融合,预期下修至 0-3ms;
- ~~"K=2560 封顶效率 ~65%"~~ **已被 2026-07-07 的复核推翻**(单点外推 + 分母用错:
  本机是 H200 NVL,dense bf16 纸面 ~835 不是 SXM 的 989.5),见下节。

### GEMM 天花板复核(2026-07-07,单卡 H200 NVL,workflow 三线验证)

| 证据 | 数据 | 结论 |
|---|---|---|
| 机器可持续上限(8192³) | **622 TFLOPS**,SM 降频至 1350MHz,功耗 592/600W 顶格 | **功率墙**解释纸面→实测的全部差距(622/835≈74.5% ≈ 1350/1785 boost) |
| K 扫描 @ M=N=8192 | K=1280→10240: 561→621;K=2560 = 594 (95.5%) | **K 惩罚仅 ~5%,"K 封顶"理论作废** |
| FFN-up 真形状 M 曲线 | M=320: 408 (66%) / 640: 527 (85%) / **1536: 642 (≥100% ceiling)** | **bs8 的 FFN 已贴功率墙跑满,kernel 层面零余量** |
| QKV fused M=1536 | 587 (94%);分开也有 ~87% | 融合话题就此关闭 |

**FFN 不能再优化的正确依据** = 它在真实形状(M=1536)下达到了实测机器可持续吞吐的
100%——不是"形状封顶",是硅片功率墙。唯一换挡杠杆是 fp8(每字节吞吐翻倍)。

**DiT"无计可施"修正**:M=640 的 GEMM 在 85% ceiling(M 是 kernel 层面唯一约束),
但代码复核(cross_attention_dit.py / LayerwiseFM_ActionHeader.py)发现结构性冗余,
合计 ~3-4ms/step,新增第 1 档卡片 **A6(DiT 去冗余包)**:
1. repeated_diffusion_steps=2 的两份 encoder 完全相同 → cross-attn K/V 算了两遍
   (encoder 流 M=3072 一半重复),去重 ~2-3ms/step,最大单项;
2. action_decoder 对 40 个 query token 全算只留 8 个 → 5× 冗余(先切片再解码);
3. 32 个 16 行的 adaLN GEMM(输入相同的 temb)→ 可批成 1 个堆叠 GEMM;
4. SELF 块 q/k/v 同源可融(16 块),CROSS 块 k/v 同源可融(16 块);
5. proj_out_1/2 死参数 ~13M,白吃 optimizer + 梯度通信(return_pre_output=True 恒真)。

## 记录约定

每张卡跑完把结果(节点、配置、p50/p90、Δ)追加到
`docs/qwenpi_zero2_h200_optimization_log.md`;profile 需要留档时用
`RUN_ID=<卡片号>` 并上传 HF `qihankang/startVLA_profile`(保持 profiles/
目录结构,不删已有数据)。
