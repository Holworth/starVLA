# QwenPI ZeRO-2 Steady-State Profile on 8x H200

Date: 2026-07-03. Host: viking-cr-196 (8x NVIDIA H200 141GB, NVSwitch NV18,
driver 595.58.03). This document supersedes the performance conclusions drawn
from the 2-step full trace in `docs/qwenpi_zero2_profile_handoff.md`.

> **Follow-up:** the optimizations proposed in §6 were implemented and
> measured the same day — see `docs/qwenpi_zero2_h200_optimization_log.md`
> (569 → 498 ms/step; includes measured rejections of the NCCL_PROTO/NVLS
> hypotheses and the torch.compile-VLM blocker).

## TL;DR

1. The dominant `FillFunctor<int>` elementwise kernel in the original 2-step
   trace (44.6% of kernel time) is a **step-1 compilation artifact**: 99.9% of
   its instances (100.0% of its time) sit inside `train_step_1`, which runs
   Triton compile + autotune for the flash-linear-attention (gated delta rule)
   kernels. At steady state it is 1 instance/step, ~0 ms. Any profile of this
   job must be read from steady-state steps (>= ~20).
2. A `MAX_STEPS=100` reproduction was run on this machine
   (`profiles/qwenpi_zero2_h200_rank0_step100_fulltrace_20260703/`), matching
   the 2026-07-01 runs to <2% on every steady-state metric.
3. The real steady-state problem is **GPU idle ~51%** of each step. The job is
   launch/CPU-bound and comm-exposed, not compute-bound: current MFU is
   ~10.8%; the compute floor at 40% MFU would be ~140 ms/step vs the current
   528 ms/step no-nsys baseline. There is roughly **3.5-4x headroom** on this
   hardware, and a concrete, verified optimization roadmap below.

## 1. Why the 2-step trace was misleading

The handoff artifact (`qwenpi_zero2_rank0_fulltrace_2step_nsys2026_20260628_2143`,
8x H20) ran only 2 steps. Step 1 includes Triton compilation and autotuning of
the `fla` gated-delta-rule kernels; autotune benchmarking launches hundreds of
thousands of buffer fills. Measured on the 07-01 H200 100-step trace:

| Metric | Whole trace | Inside step 1 | Steady (steps 21-100) |
|---|---:|---:|---:|
| `FillFunctor<int>` instances | 154,900 | 154,801 (99.9%) | 1.0 /step |
| `FillFunctor<int>` time | 12.35 s | 12.35 s (100.0%) | ~0.00 ms/step |

Step-1 wall time is 103.1 s with a cold Triton cache (fresh container,
2026-07-03 run) and 24.8 s with a warm cache (07-01 run, reused container).
Steps 2+ drop to ~0.6-0.7 s immediately. With `MAX_STEPS=2` the warmup
dominated the whole trace; with `MAX_STEPS=100` (a realistic training config)
steady state dominates and the elementwise kernel disappears from the picture.

## 2. Reproduction record

Command (after `enroot create --name starvla containers/starvla-profile/starvla.sqsh`;
note the enroot data dir is per-SLURM-job on this cluster, so the container
must be recreated when the job changes):

```bash
bash scripts/starvla/profile_qwenpi_zero2_single_node.sh \
  RUN_ID=qwenpi_zero2_h200_rank0_step100_fulltrace_20260703 \
  PROFILE_RANKS=0 NSYS_CAPTURE_MODE=none \
  MAX_STEPS=100 PROFILE_START_STEP=1 PROFILE_END_STEP=1
```

Run shape: 8 ranks train (accelerate + DeepSpeed ZeRO-2, bf16, per-GPU batch 8,
grad accum 1, full unfreeze), rank 0 wrapped by nsys 2026.3.1 full trace
(`cuda,nvtx,cublas,cudnn,osrt`). Training completed 100/100 steps,
`accelerate_exit_status=0`.

Artifacts:

```text
profiles/qwenpi_zero2_h200_rank0_step100_fulltrace_20260703/
  *.rank0.nsys-rep (189 MB)   *.rank0.sqlite (445 MB)
  *.rank0.stats.txt  *.rank0.nvtx_sum.txt  *.rank0.cuda_api_sum.txt
  *.rank0.cuda_gpu_kern_sum.txt  *.command.log
```

Reference runs for comparison:

| Run | Steady mean/med step | Notes |
|---|---|---|
| `...step100_fulltrace_20260703` | 690 / 661 ms | this doc's primary trace |
| `...nsys_step_100_profile_20260701_215926` | 644 / 622 ms | same config, 07-01 |
| `...step_100_profile_20260701_223452` (no nsys) | ~528 ms (1.89 it/s) | true baseline |
| `...nccl_simple_step_100_profile` | max_steps=10, NCCL_PROTO=Simple | proto A/B (see 5.4) |

Per-kernel steady-state metrics agree between the 07-01 and 07-03 traces to
<2%. nsys inflates step wall by ~1.25-1.3x (528 -> 660-690 ms); scale in-trace
savings by ~0.75-0.8 when projecting to production.

Analysis tooling (host python3, run against a **local copy** of the sqlite):

```bash
python3 tools/analyze_steady_state.py <rank0.sqlite>   # warmup vs steady kernel mix, busy/idle
python3 tools/analyze_gaps.py <rank0.sqlite>           # gap histogram, per-phase busy, API load
```

## 3. Model / job shape (verified against checkpoint + trace)

| Item | Value |
|---|---|
| Trainable params | 7.524B total = VLM 4.567B + DiT action head 2.957B (39%) |
| Qwen3.5-4B text stack | 32 layers (24 gated-delta-rule linear-attn + 8 full-attn), hidden 2560, vocab 248,320 |
| Vision tower | 0.334B, 2 views x 256 patch tokens per sample (224px input upscaled to 256px by processor min-pixels) |
| VLM seq len | 179-181 tokens (batch 8/GPU -> 1448 tokens/GPU/step) |
| DiT head | 32 blocks (16 cross + 16 self interleaved), inner dim 2560, 40 heads; batch 16 (8 x `repeated_diffusion_steps=2`), 40 query tokens (32 future + 8 action; `action_horizon: 8` from the LIBERO yaml) |
| Gradient checkpointing | NOT active (yaml requests it; kernel counts show exactly 1 fwd per layer) |
| FLOPs | ~53.5-54 TFLOP/step/GPU -> current MFU ~10.8% of H200 bf16 peak |

## 4. Steady-state results (steps 21-100 of the 07-03 trace)

Headline (per step, nsys scale, 693.6 ms wall):

| Bucket | ms/step | % of wall |
|---|---:|---:|
| GPU busy (kernels + memcpy/memset) | 336.6 | 48.5% |
| — compute kernels | 250.4 | 36.1% |
| — NCCL kernels (union) | 117.2 | 16.9% (exposed: 73 ms, 10.6%) |
| **GPU idle** | **357.0** | **51.5%** |

Phase split:

| Phase | wall ms/step | GPU busy | of which NCCL |
|---|---:|---:|---:|
| `DeepSpeedEngine.forward` | 238.7 | 27.8% | 0% |
| `DeepSpeedEngine.backward` | 367.5 | 54.6% | 20.0% |
| rest (optimizer + step boundary) | 83.9 | 83.0% | 52.3% |
| `dataloader` | 0.2 | — | — |

Idle decomposition (the money table):

| Idle source | ms/step | Mechanism (verified, see §5) |
|---|---:|---|
| Micro-gaps < 500 µs | ~213 | ~12,500 kernel launches/step, eager-mode dispatch; DiT head contributes ~2,810 launches / ~58 ms, VLM ~9,550 launches / ~188 ms |
| Step-boundary serial section | ~45 (every step) | 53.8 ms `cudaDeviceSynchronize` (DeepSpeed ThroughputTimer) + ~10 ms python step tail + 39 ms HF processor preprocessing on the main thread before the first forward kernel |
| Episodic giant stalls | ~33 (window avg) | host-side stall inside `ncclAllReduce()` *enqueue* of gradient bucket 0; 8 stalls of 262-375 ms + 13 of 52-70 ms clustered in steps 58-93 |

Top steady kernels (ms/step): grad AllReduce bf16 73.4 (15/step, med 4.73 ms),
param AllGather 43.8 (2/step, med 21.9 ms), elementwise family ~68 (6,900
inst/step), bf16 sm90 GEMMs ~99 (GEMMs internally run at ~57% of peak — the
kernels themselves are fine), ZeRO-2 grad-flatten cats 21.5, fused Adam 9.2.

## 5. Verified root causes

Each item below was established from the trace/code and independently
re-verified (adversarial re-derivation of the numbers).

### 5.1 Step boundary: DeepSpeed timer sync + in-forward CPU preprocessing

- `ds_config.yaml` has no `timers` section, so DeepSpeed's ThroughputTimer
  defaults to `synchronized=true`: `engine.step()` ends with a
  `cudaDeviceSynchronize` costing **53.8 ms/step**, draining the optimizer
  tail (43.8 ms AllGather + 9.2 ms fused Adam) before the CPU may continue.
- `build_qwenvl_inputs` (starVLA/model/modules/vlm/QWen3_5.py) runs the full
  HF processor — image resize/rescale/normalize/patchify + chat-template
  tokenization — on the main thread **inside `model.forward`**, every step:
  mean 41.1 ms/step with zero kernels in flight. The DataLoader already runs 4
  workers with `pin_memory=True`, `prefetch_factor=2`, but its `collate_fn` is
  identity and returns raw PIL images + strings, so the workers do ~nothing
  (dataloader NVTX: 0.2 ms/step).
- `pixel_values` (25.2 MB/step fp32) is uploaded as a *pageable* blocking copy
  (3.0 ms CPU + 2.9 ms GPU). The processor min-pixels setting upscales
  224x224 inputs to 256x256 (+31% vision tokens).

### 5.2 Launch-bound eager execution (~213 ms/step)

- Forward alone: 4,889 launches, 134.6 ms of micro-gap idle after the first
  kernel; CPU is in the Python dispatcher between launches (launch API time
  itself is only 19.4 ms).
- DiT head: 47.1 ms fwd wall with only 16.3 ms busy (1,026 launches); ~29-30
  kernels per cross block (7 GEMMs + attention + 4 dropouts + 2 LN + ~15
  pointwise incl. fp32 AdaLN modulation and redundant fp32->bf16 casts before
  each of q/k/v). Head shapes are static (pad encoder seq 181->192 to remove
  the last dynamic dim) -> torch.compile/CUDA-graph friendly.
- VLM forward: 226 tiny D2H reads + 236 `cudaStreamSynchronize` per step
  (`.item()`-style seq-len/mask reads in the Qwen3.5 hybrid-attention / fla
  glue code) — cheap today (~4 ms) but they pin the CPU to the GPU timeline
  and block any run-ahead / graph capture.
- ViT processes 16 images sequentially (384 attention calls/step instead of a
  batched 24).
- `QWen3_5.py:62` hard-codes `attn_implementation="sdpa"`, silently overriding
  the config's `flash_attention_2` (trace shows mem-efficient SDPA kernels).

### 5.3 Communication (117 ms/step busy, ~73 ms exposed)

- Volumes are inherent to full unfreeze + ZeRO-2: grads 15.05 GB/step bf16
  (15 buckets = ceil(7.524e9 / 5e8-elem bucket)), post-step param AllGather
  15.05 GB/step (2 ops).
- DeepSpeed 0.16.9 with `use_multi_rank_bucket_allreduce=true` implements the
  "reduce-scatter" as a fused **AllReduce**, moving 2x the bytes of a true
  reduce-scatter (observed: zero ReduceScatter kernels in the trace).
- `overlap_comm` works (buckets stream through backward, spacing med 17.7 ms),
  but ~29 ms/step of AllReduce is still exposed because the backward compute
  stream is only 54.6% busy (launch-bound) — fixing launch overhead converts
  exposed comm into overlapped comm for free. The 43.8 ms AllGather is 100%
  exposed (structural: ZeRO-2 gathers params after Adam, before next forward).
- Effective busbw: AllReduce 371 GB/s (82% of 450 GB/s NVLink line rate),
  AllGather 300 GB/s (67%). Bandwidth headroom is real but modest (~25-30
  ms/step busy).

### 5.4 Two debunked hypotheses (do not chase these)

- **"RING_LL protocol is mis-selected for large messages"** — false. The
  `_RING_LL` kernel-name suffix is an NCCL 2.19+/2.21 entry-point
  consolidation artifact. Measured 371 GB/s busbw is physically impossible
  under LL (~225 GB/s cap), and the existing `NCCL_PROTO=Simple` A/B run
  changed AllGather time by 0.3%. `NCCL_PROTO` tuning is a dead end here.
- **"Giant backward stalls are rank-0 waiting for straggler ranks"** — false.
  The 262-375 ms stalls are host-side stalls **inside the `ncclAllReduce`
  enqueue call** for bucket 0 (NVTX spans the whole gap; the launching thread
  makes zero CUDA calls until the gap's final ~40 µs; the thread is invisible
  to OSRT during the stall, unlike its normal traced condvar waits). NCCL
  kernel durations on stall steps are unchanged (<1.5%), meaning all 8 ranks
  arrive nearly simultaneously — a common-mode host phenomenon (suspect:
  kernel-level page-fault/reclaim or NCCL-internal spin), episodic over steps
  58-93 in this trace. Root cause still open (see §7).

## 6. Optimization roadmap

Savings quoted at nsys scale (multiply by ~0.75-0.8 for production estimate).
Baseline: 690 ms/step traced / 528 ms/step production.

### Tier 1 — config & small patches (fast, high confidence, ~90-100 ms)

| # | Change | Saving (ms/step) | Notes |
|---|---|---:|---|
| 1 | `ds_config.yaml`: add `"timers": {"throughput": {"synchronized": false}}` | 30-45 | kills the per-step 53.8 ms device sync; no semantics change |
| 2 | Move `build_qwenvl_inputs` into DataLoader `collate_fn` (workers+pin_memory already configured; keep old path for `predict_action`) | ~40 | removes the 41 ms in-forward CPU preamble; also shrinks per-rank arrival jitter |
| 3 | `.to(device, non_blocking=True)` + pinned collated tensors for the 25 MB `pixel_values`; cast to bf16 in collate | 4-6 | pageable->pinned; halves bytes |
| 4 | Skip lm_head logits: call the base model or pass `logits_to_keep=1` (QwenPI only consumes `hidden_states`) | 2.5-3 | also frees a 719 MB bf16 logits allocation |
| 5 | Fix `attn_implementation` hardcode to honor `flash_attention_2` (QWen3_5.py:62) | small | correctness-of-config; FA2 vs SDPA on 8 full-attn layers |
| 6 | Gate `loss.item()` / tqdm / logger behind `logging_frequency` | 2-8 | matters after #1 removes the big sync |

### Tier 2 — launch-overhead & comm (medium effort, ~80-120 ms)

| # | Change | Saving (ms/step) | Notes |
|---|---|---:|---|
| 7 | `torch.compile` the DiT (`self.model`, `dynamic=False`, max-autotune), pad encoder seq to fixed 192; stage 2: `reduce-overhead` (cudagraphs) | 30-60 | head is 2,810 launches / 58 ms micro-gap; shapes verified static; scoped away from fla Triton kernels |
| 8 | Remove the 226 per-step D2H `.item()` syncs in the VLM/fla glue (precompute seq-lens once per batch on CPU) | 10-30 | prerequisite for run-ahead & any graph capture of forward |
| 9 | `"use_multi_rank_bucket_allreduce": false` in ds_config | 8-15 | true per-partition reduce halves grad wire bytes (15->7.5 GB) |
| 10 | Verify/enable NVLS: persist `NCCL_DEBUG_FILE` to a mounted path, run nccl-tests in-container, try `NCCL_ALGO=NVLS` | 10-18 | AllGather 43.8 -> ~30-33 expected if NVLS engages |
| 11 | Avoid the x2 repeat of 32 hidden-state layers via K/V-shared cross-attention (fold repeat into query dim) | 4-7 | 712 MB/step of copies + ~130 launches |
| 12 | Widen the bf16 autocast to cover action encoder/decoder; hoist AdaLN modulation casts | 2-4 | kills fp32 SIMT GEMMs + ~200 cast launches |

### Tier 3 — structural (larger projects)

- **Batch the ViT across images** (16 sequential -> 1 varlen call per layer):
  large share of the VLM's 3,863-launch forward.
- **torch.compile / graph the VLM forward** once #8 removes the syncs.
- **Hunt the episodic enqueue stalls** (~33 ms/step in this window): needs a
  non-nsys run with `perf sched` / major-fault counters; check THP/compaction,
  NUMA balancing, and CPU pinning of the 8 ranks + dataloader workers.
- **Overlap the param AllGather with next forward** (custom DeepSpeed surgery
  or ZeRO-3-style prefetch): up to ~30 ms structural exposure.
- **(Model-owner decision)** The DiT head is 2.96B params (39% of trainable
  params, comm volume, and a third of FLOPs) serving 40 query tokens. Halving
  depth/width would cut compute+comm+launches massively but changes the model.

### Expected trajectory

Tier 1 alone: ~528 -> ~440-460 ms/step production (~2.2 it/s). Tier 1+2:
~330-380 ms/step (~2.7-3.2 it/s). Hardware floor at 40% MFU is ~140 ms/step;
closing further requires Tier 3 (graphed VLM, comm restructuring).

## 7. Open questions

1. Root cause of the episodic `ncclAllReduce` enqueue stalls (§5.4) — host
   kernel-level investigation pending; reproduces intermittently (16 of 80
   steps in this trace, all at bucket 0).
2. NVLS engagement on this platform (NCCL 2.21.5): debug log was lost to the
   container's /tmp; rerun with `NCCL_DEBUG_FILE` on a mounted path.
3. Exact mechanism of the 2 equal-size AllGathers (param groups are 4.57B vs
   2.96B, yet the two ops take identical time) — DeepSpeed allgather sharding
   detail, harmless but unexplained.
4. The single 358 ms forward stall (step 39) shares the silent-CPU signature
   of the backward stalls but sits between two elementwise launches on the
   main thread — same host phenomenon, different location.

## Appendix: method notes

- Steady window: steps 21-100 via NVTX `train_step_N` ranges; busy = interval
  union of kernels + memcpy + memset; exposed NCCL = NCCL union minus overlap
  with compute union.
- All quantitative claims in §5 were independently re-derived by a second
  analysis pass (adversarial verification); corrections found were <5% except
  where noted in the text.
- nsys overhead: compare `model_times` in the command logs (0.51-0.53 s
  no-nsys vs 0.64-0.69 s traced).
