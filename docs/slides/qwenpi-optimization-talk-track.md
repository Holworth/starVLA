# QwenPI Training Optimization — 演讲稿与 Q&A(逐页)

> 与 `qwenpi-optimization.pptx` 的 speaker notes 同步;每页含讲稿与「客户可能的提问 → 建议答法」。
> 弹药索引:所有 nsys trace 在 HF `qihankang/startVLA_profile`;DeepSpeed 机制源码行号在 P12 notes;否证清单在 P14 notes。

## P1 封面

**讲稿**:各位好。今天汇报我们在 starVLA QwenPI 训练上的性能优化工作:8 张 H200 上,把每步训练时间从 800 毫秒压到 306 毫秒。整个过程的方法论只有一句话——先用 nsys 找到问题,再针对问题做修复,每一步都有同机 A/B 数据。今天所有数字,大家都可以拿我们公开的 nsys trace 复核。

- **Q: 测试环境?** A: 单节点 8×H200 SXM、全 NVLink,DeepSpeed ZeRO-2 全参数微调,LIBERO 数据;软件栈第 4 页详述。

## P2 Agenda

**讲稿**:流程很简单:先给结果,然后花一分钟对齐模型和负载,重点在中间两段——诊断出了哪些问题、每个问题怎么修;最后讲还剩什么、下一步怎么走。中间任何一页欢迎随时打断。

## P3 Results at a Glance

**讲稿**:四个数:步时 800→306 毫秒,2.6 倍;吞吐从交接时的 80 到 209 samples/node/s,过了 200 的目标线;MFU 7.2%→18.8%;batch 24 时 309,目标的 1.5 倍。口径特别说明:「步时」是**步间周期**——包含步尾参数 AllGather 的排空,不是只算前反向窗口;并用 1000 步无 profiler 干净跑验证,稳态 303–313ms。

- **Q: 之前说过 249,为什么现在 306?** A: 我们自己发现并修正了口径:免同步日志之后,in-step 计时器漏掉了步间 ~57ms 的 AllGather 排空。249 是计时器窗口,306 是真实周期——你们用 nsys 量到的就是 306。宁可难看,必须经得起复核。
- **Q: 209 离 200 太近?** A: bs24 是 309;且 bs8 最大的单块损耗就是那 57ms AllGather 暴露,最后一页有重叠方案,修复后预期回到 ~250。
- **Q: MFU 18.8% 太低?** A: 分母是纸面 989.5 TFLOPS;这台机器 GEMM 持续上限实测 622(功耗墙),kernel 层已接近打满(见 P14);bs24 时 27.7%。

## P4 Model & Workload

**讲稿**:对齐负载。7.52B 全参训练:Qwen3.5-VL 4B 主干——decoder 是 32 层混合结构(全注意力 + gated-delta-net 交替),这个细节在 CUDA Graph 页会变得重要;动作专家是 2.96B cross-attention DiT,深宽随 VLM(32 块、hidden 2560),逐层交叉注意 VLM hidden states。右表:ZeRO-2、bs 8–24、文本定长 192。

- **Q: 为什么 ZeRO-2 不是 ZeRO-3/FSDP?** A: 与你们对齐的基线选型,本次不改训练语义。但今天 ZeRO-2 无法重叠参数 AllGather 已是第一大结构损耗——最后一页正是建议评估 FSDP 或改造 ZeRO-2。
- **Q: 定长 192 对真实数据成立吗?** A: pad 长度是配置项,当前分布覆盖良好、padding 开销实测很小;更长指令按长度分桶、每桶各捕一张图,方案现成。
- **Q: transformers 为什么停在 5.3?** A: 有意不动,与你们环境钉板对齐;所有优化都绕开了升级需求。

## P5 Baseline Diagnosis

**讲稿**:基线的一个真实步,807ms,全按 **GPU 执行时钟**画。上两行:阶段与模块——前向 309 里 Qwen3.5 148、DiT 124;反向里 DiT 234、Qwen3.5 175。下面是真实 kernel 泳道。五个问题各配修复与收益:fp32 动作头、CPU 在 forward 里预处理、每步同步点、发射瓶颈、通信配置。本页是全 deck 的地图。

- **Q: CPU 侧 forward 只有 225ms,为何画 309?** A: eager 下 CPU 发射超前:forward 函数 225ms 返回,GPU 队列里的 DiT 前向执行到 309。所以统一 GPU 时钟、kernel 按发射窗口归属(correlationId 可复算),两行边界才能对齐。
- **Q: Qwen3.5 反向为何不是前向两倍?** A: GPU busy 恰是 2.46×(纯 GEMM 1.94×),符合理论;墙钟只有 1.18× 是前向发射饥饿——占空比 34% vs 70%。这是 P10 的主题。
- **Q: 13,868 个 kernel 哪来的?** A: eager 逐算子发射,kernel 中位 8µs、79% 小于 20µs,发射成本成为主导。

## P6 Optimization Roadmap

**讲稿**:柱子是每级落地后的真实步时(周期口径)。六级:bf16 动作头 −214 最大;流水线与同步 −104(内含环境升级 −37);混合注意力 −32;CUDA Graph −122;通信调优 −22。累计 2.6×。每根柱子都是同机独立 60 步实测。

- **Q: 为什么不标每项增量?** A: 柱差即增量;刻意展示绝对时间,那才是你们感受到的速度。
- **Q: 能只挑几项吗?** A: 大多独立,但有依赖:Graph 需要定长 shape 与 torch 2.7.1;混合注意力需要 flash-attn 2.8。可给依赖图分阶段合入。

## P7 Fix 1 · bf16 Action Head

**讲稿**:最大单项。左图:基线 37% 的 GPU 计算是 fp32 GEMM,全部来自 2.96B 的动作头。先澄清一个事实:权重存储本来就是 bf16——DeepSpeed bf16 引擎把全部可训练参数按 bf16 存储,fp32 只存在于优化器的 master 副本里。基线的 fp32 计算来自代码里一个作用域过宽的 autocast(float32) 上下文,它把整个动作专家的前向连同 2.96B transformer 的矩阵乘一起拖进了 fp32。我们的修复就是把这个作用域收窄。

**精度边界总结(可直接向客户宣读)**:
- **转成 bf16 的只有一处**:DiT transformer 主体的那一次调用——32 个 block 的 Linear/attention/MLP 矩阵乘,前向与反向(反向经由保存的 bf16 激活自动继承);GEMM 在 tensor core 上以 fp32 累加。
- **保留 fp32 的全部部分**:①flow-matching 标量数学(噪声采样、(1−t)·noise+t·actions 插值、velocity 目标、timestep 采样与离散化);②action/state encoder、位置编码、action_decoder;③MSE loss;④DiT 内部 LayerNorm(autocast 白名单强制);⑤TimestepEncoder(显式 cast 到自身参数精度);⑥优化器——Adam 在 fp32 master 权重上更新,grad-norm 走 fp32 AllReduce。
- **结果:训练 loss 没有可测的退化**。1000 步、仅 dtype 不同的对照:末段 100 步均值 0.1473 vs 0.1473(四位小数一致);末段逐步差 std 0.014,仅为单条曲线自身步间波动(0.080)的 0.4 倍——精度差异淹没在采样噪声之下。

**混合精度训练的可行性(理论 + 实证)**:
1. **数值理论**:bf16 与 fp32 指数位相同(8 bit)→ 动态范围一致,没有 fp16 的上/下溢与 loss-scaling 需求;尾数少(7 bit)的影响由三道防线兜住——tensor core 的 bf16 GEMM **以 fp32 累加**、优化器在 **fp32 master 权重**上更新(微小更新量不会被低精度舍入吞掉)、归约/归一化/loss 由 autocast 白名单强制 fp32。
2. **业界实证**:千亿级模型的标准训练精度就是 bf16——BLOOM-176B 在 OPT 的 fp16 不稳定教训后明确选择 bf16;Llama 3 系列以 bf16 训练;Megatron/DeepSpeed 的官方配方即 bf16 计算 + fp32 master。开源 VLA 实现(如 OpenVLA)同样默认 bf16 autocast 微调。
3. **本仓库实证**:同一次训练里 4.5B 的 VLM 主干从第一天起就在 bf16 下运行;动作头切 bf16 后的千步对照见上文统计。

**参考文献**:
- Micikevicius et al., *Mixed Precision Training*, ICLR 2018 (arXiv:1710.03740) — 混合精度训练奠基:低精度计算 + fp32 master 权重,训练质量与 fp32 持平。
- Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (arXiv:1905.12322) — bf16 无需 loss scaling 即可匹配 fp32 收敛。
- BigScience, *BLOOM* (arXiv:2211.05100) 与 Meta *OPT* (arXiv:2205.01068) 训练日志 — fp16 大模型不稳定的实录与 bf16 选型依据。
- Meta, *The Llama 3 Herd of Models* (arXiv:2407.21783) — bf16 作为当代大模型的默认训练精度。
- PyTorch AMP 官方文档(pytorch.org/docs/stable/amp.html)— autocast 嵌套语义与 fp32 白名单算子表;NVIDIA *Train With Mixed Precision* 指南(docs.nvidia.com)— tensor core fp32 累加;DeepSpeed bf16 配置文档。
- Kim et al., *OpenVLA* (arXiv:2406.09246) — VLA 领域 bf16 微调的公开参照。

- **Q: loss 一致 ≠ 任务成功率一致?** A: 同意——任务级评测要在你们的评测器上做,这是建议的联合验证项。我们的证据:千步损失轨迹统计不可区分 + 数值敏感处全部保留 fp32,与业界 VLA/diffusion 训练通行做法一致。
- **Q: 为什么当初是 fp32?收窄合理吗?** A: 权重存储本就是 bf16,外层 fp32 从未保护过权重精度——基线花两倍 GEMM 代价买到的精度只作用于激活。收窄用的是 torch.autocast 的标准嵌套机制(内层覆盖外层、白名单算子仍走 fp32),与 VLM 主干的既有跑法一致,是 PyTorch AMP/Megatron/DeepSpeed 的标准配方。
- **Q: 早期(前200步)两条曲线好像有偏差?** A: 扩散时间步/噪声未跨 run 固定种子,陡降段 loss 对抽到的 t 高度敏感;进入平稳段偏移严格归零(步401-600 与 901-1000 的均值差都是 0.0000)。需要逐位级验证可以加固定采样种子的模式。

## P8 Sync Points(问题)

**讲稿**:前向计算流上四条虚线,都是 GPU 等 CPU 的往返:①图像解码/tokenize 在 forward 里,每步开头 GPU 干等 ~37ms;②ViT 的 RoPE/位置编码逐步在 CPU 算;③ViT→decoder 的 MRoPE 位置索引也在 CPU;④loss 读回 + 计时器每步两次拉停发射流水线。时间线标出它们在真实步里的位置。

- **Q: 这些是 HF 的 bug 吗?** A: 是通用性设计,不是 bug——但训练热循环输入形状固定,这些就成了纯开销,可预计算/缓存。

## P9 Removing the Sync Points

**讲稿**:四个修复:预处理进 8 个 worker 与上一步 GPU 重叠;锁页内存 + 专用拷贝流;免同步日志;MRoPE 按图像网格缓存。加混合注意力,周期 549→450。

- **Q: 标题 −67,之前不是 −117?** A: 117 是 in-step 计时器口径;把同步点移出窗口的同时,原本藏在窗口里的 AllGather 排空也被移出,周期口径净赚 67。两个口径我们都主动讲清。
- **Q: 8 个 worker 抢 CPU?数据顺序变吗?** A: data_times 稳定 1–2ms,CPU 有余;采样器与顺序不变,仅预处理位置变,有逐位验证。

## P10 Launch-Bound Forward

**讲稿**:CUDA Graph 的动机。同样的数学:基线 Qwen3.5 前向 148ms 墙钟,GPU 真正在算的只有 50ms——2/3 在等发射:3,929 次 launch,每 op Python 调度 30–60µs,kernel 中位 8µs。优化后 6 次 graph launch,墙钟 65ms,2.3×。

- **Q: 57% 占空比还不高?** A: 剩余是图间 eager 胶水与本页窗口效应;整步 GPU 空闲 205→43ms,继续压胶水收益递减,优先级给了 AllGather。
- **Q: 反向为什么不受害?** A: autograd 在 C++ 发射,占空比本就 70%。

## P11 CUDA Graphs — Why Four, Not One

**讲稿**:vision 一张图、decoder 四张、DiT compile。为什么四张不是一张?右图:DS 的梯度 AllReduce 靠 hook,hook 只能在图与图之间触发。四张图=每组反向重放完即发通信,与计算重叠;一张图=所有通信挤到反向结束后串行——同样的计算与字节,红线(步结束)更晚。

- **Q: 模型改了图要重做?** A: 启动时自动捕获,分组 runner 对层列表通用;换注意力结构需回归。保留逐层 compile 回退。
- **Q: 为什么 8 层一组?** A: 实测甜点,发射节省在 8 已饱和,更大只增编译时长与故障面。
- **Q: 一张大图真的不行?** A: 技术上也过不去:线性注意力库 graph-break 把单图切成共享显存池的分区,与 ZeRO-2 hook 分配冲突,2.6/2.7.1 都崩。细节会后展开。

## P12 ZeRO-2 Comm Tuning: Two Knobs

**讲稿**:两个旋钮两个问题。旋钮一 reduce_scatter=false:这里有个反直觉的事实——DeepSpeed 默认的 reduce_scatter=true **底层其实仍是整桶 all_reduce**(不是真 reduce-scatter),只是多付了每桶的 flatten(诊断页那 22ms CatArrayBatchedCopy)和 copy-back;梯度本就连续,关掉后一次 all_reduce 直达,**字节完全不变、纯省两轮拷贝**(−13ms)。旋钮二桶大小:DS 只有**两个**梯度缓冲,每次桶满计算流必须等上次 reduction 排空。真实对照:5e8 反向被打断 15 次/47ms;1.5e9 只有 8 次/27ms。两旋钮正交同向。

**reduce_scatter 的源码事实(应对追问)**:DeepSpeed 0.16.9 里 reduce_scatter=true(默认 use_multi_rank_bucket_allreduce=True,构造默认)→ allreduce_and_scatter → allreduce_bucket,其中 `rank=None` 时调 **dist.all_reduce(L1521)**——所以默认 true 路径是**完整 all_reduce**(两段全做,2(N−1)/N·G 字节)加重排,而非承诺的半量 reduce-scatter。真正的 reduce-scatter 只在 use_multi_rank=false 子路径(→ dist.reduce 逐分片),我们测过在单机更慢。

- **Q(最尖锐): all_reduce 不就是 reduce-scatter + all-gather 吗?那 reduce_scatter=true 应该省一半字节?** A: 理论完全对——纯 reduce-scatter 跳过 all-gather 段、只搬一半字节,正是 ZeRO-2 该用的。但陷阱是 DeepSpeed 默认的 reduce_scatter=true 底层调的是 dist.all_reduce,**根本没做真 reduce-scatter**,搬满量字节还加重排,所以关掉它是纯赚、不是权衡。真 reduce-scatter 在另一条子路径,我们测过在单机反而更慢:NVLink 上通信与反向计算重叠、非带宽瓶颈(478 GB/s 已到顶但被藏住),砍一半字节几乎不缩短暴露时间,且它碎成很多小 dist.reduce 又照样 flatten。
- **Q: 那多节点上呢?** A: 多节点通信暴露且带宽受限,砍一半字节才真正值钱——但正确做法是上一个**真正实现** reduce-scatter 的路径/新版 DS,不是把开关拨回默认 true(那只是 all_reduce+重排)。桶大小也要按拓扑重调;均为 15 分钟级实验,可以一起跑。
- **Q: 1.5e9 通用吗?** A: 不通用——按梯度元素计数、取决于参数布局(DiT 那段 3e9 连续梯度),所以做成了模型级配置。
- **Q: DS 为什么不修双缓冲?** A: 上游显存/流水深度取舍;我们的异步 gather 改造会动同一份代码,可一并评估。

## P13 Batch Scaling

**讲稿**:8→24,吞吐 209→309、MFU 27.7%——每步固定开销被摊薄。甜点 24;32 回落,已排除 OOM 与数据加载,初步指向显存压力。边界:bs24=全局 192,收敛需你们数据+评测器验证后才可采纳。

- **Q: lr 怎么调?收敛谁验证?** A: 线性缩放+warmup 起步;验证在你们侧,我们陪跑。此前推荐配置仍是 bs8。
- **Q: bs32 为什么塌?** A: 现场不猜;trace 在手,可作后续项深挖。

## P14 What's Next

**讲稿**:优化后的完整一步:计算结束后 44ms 参数 AllGather 完全裸露,占 18%——当前最大单块。kernel 分布已平坦:GEMM 95ms 且顶在 622 TFLOPS 功耗墙上,kernel 层无可榨,剩下都是结构性机会。三个方向按规模:重叠 AllGather ~50ms(FSDP 或 ZeRO-2 异步 gather);fp8 破功耗墙;梯度压缩拿回步内 ~20ms。2.6× 已落袋,bs8 通往 250+ 的路在 AllGather 上。

- **Q: FSDP 迁移成本收益?** A: 收益上限=实测 44–57ms。FSDP 动 checkpoint 与配置、面大而彻底;ZeRO-2 补丁集中一个文件、面小。建议一周级 PoC 把两条路的数字跑出来再定。
- **Q: fp8 收敛风险?** A: TE 路线 per-tensor scaling + 与 bf16 相同的验证流程(千步 loss → 任务评测)。是唯一能抬 GEMM 上限的杠杆。
- **Q: 哪些方向你们试过被否了?** A: 有数据的否证清单:全 FA2 慢 10ms;NCCL 调参无效(478 GB/s 已线速);单张大图不可行;FlashQLA 天花板 10.5ms。清单可发你们,避免重复踩坑。
