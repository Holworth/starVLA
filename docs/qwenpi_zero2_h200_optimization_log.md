# QwenPI ZeRO-2 Optimization Log — 8x H200 (2026-07-03)

Follow-up to `docs/qwenpi_zero2_h200_steady_state_profile.md`. Goal: close the
gap to a reference implementation reported at **200 samples/node/s** (same
shape: bf16 ZeRO-2, full unfreeze, bs 8/GPU, seq ~190, 2 input images,
1 node of 8x H200 → 200 samples/s = 320 ms/step at global batch 64).

## Headline result

| Metric | Baseline (as-shipped) | Optimized | Reference |
|---|---:|---:|---:|
| steady step time (steps 21-100, wall) | 569.1 ms | **497.6 ms** | 320 ms |
| throughput | 112.5 samples/s | **128.6 samples/s** | 200 |
| `timing/model` (typical fast step) | ~0.51-0.55 s | ~0.37-0.42 s | — |

All runs: 100 steps, no nsys, loss trajectory verified unchanged
(step-100 `action_dit_loss` 0.6352 baseline vs 0.6231-0.6265 optimized runs —
differences consistent with dataloader order changes only).

Measurement: `tools/bench_report.py <run>.command.log` — least-squares slope
of tqdm elapsed over steps 21-100 (robust to the 1 s timestamp resolution);
`timing/model`+`timing/data` means shown as a cross-check.

## What is enabled in the optimized configuration

Benchmark command (`bench_FINAL_optimized`):

```bash
bash scripts/starvla/profile_qwenpi_zero2_single_node.sh \
  RUN_ID=... PROFILE_RANKS=none NSYS_CAPTURE_MODE=none \
  MAX_STEPS=100 PROFILE_START_STEP=1 PROFILE_END_STEP=1 LOGGING_FREQ=10 \
  EXTRA_TRAIN_ARGS='--datasets.vla_data.preprocess_in_collate true \
    --datasets.vla_data.num_workers 8 \
    --framework.qwenvl.attn_implementation mixed \
    --framework.action_model.compile_dit true \
    --framework.action_model.pad_encoder_seq_to 192'
```

1. **DeepSpeed ThroughputTimer no longer synchronizes**
   (`ds_config.yaml: timers.throughput.synchronized=false`). Removes a 53.8
   ms/step `cudaDeviceSynchronize` at the end of `engine.step()`.
2. **`loss.item()` gated behind `logging_frequency`**
   (`train_starvla.py`): the loss stays a GPU tensor except on logging steps.
3. **HF preprocessing moved into DataLoader workers**
   (`QwenVLPreprocessCollate` in `starVLA/dataloader/lerobot_datasets.py`,
   opt-in via `datasets.vla_data.preprocess_in_collate`): the processor
   (image resize/normalize/patchify + chat-template tokenization) runs in
   workers; pixel_values cast to bf16 in collate (halves H2D bytes);
   actions stacked to a tensor. Removes the 41 ms/step CPU preamble that ran
   inside `model.forward` with the GPU idle. `Qwen_PI.forward` accepts the
   pre-collated dict batch (legacy list path unchanged).
4. **lm_head logits skipped** (`QWen3_5.py`: `logits_to_keep=1` when no
   labels): the 248,320-vocab projection (2.6 ms + 719 MB bf16 per step) was
   computed and thrown away every step.
5. **`attn_implementation` config honored** (removed a hard-coded `"sdpa"`
   override) and a new `mixed` mode: **FA2 for the vision tower** (varlen
   kernel processes all 16 images in one call per layer instead of 16
   sequential calls) + **SDPA for the text stack** (full FA2 measured ~10
   ms/step slower at seq~190 due to unpad/repad overhead). `mixed` is worth
   ~50 ms/step of model time vs full sdpa.
6. **torch.compile on the DiT action head**
   (`framework.action_model.compile_dit=true`) with the encoder sequence
   padded to a fixed 192 (`pad_encoder_seq_to=192`, masked positions, exact
   numerics): the head was 2,810 launches/step; compile removes most of its
   ~58 ms/step micro-gap idle.

## What was tried and rejected (measured, not speculation)

| Attempt | Result | Why |
|---|---|---|
| `NCCL_PROTO=Simple` | no-op | RING_LL kernel name is an NCCL 2.21 symbol artifact; auto-selection already at 371 GB/s busbw |
| `NCCL_ALGO=NVLS` (NVLS confirmed available in nccl_debug logs) | **regression** (523→561 ms) | forced NVLS beats Ring on neither the 1 GB grad buckets nor the 7.5 GB allgathers here |
| `use_multi_rank_bucket_allreduce=false` | regression (part of the same run) | 8 small per-partition reduces per bucket are latency-bound vs one fused allreduce |
| full `flash_attention_2` (text+vision) | +10 ms model time | unpad/repad + per-layer cu_seqlens work dominates at seq~190 |
| `accelerate dataloader non_blocking=true` | regression (model +60 ms) | H2D lands on default stream behind the optimizer tail; first `.tolist()` readback in the vision tower then blocks longer |
| `torch.compile` of the VLM text stack (whole-model AND per-layer) | **fails** | transformers 5.3's `output_hidden_states` capture stops collecting once layers are wrapped in `OptimizedModule`; QwenPI needs all 32 layer states. `compile_language_model` now raises with an explanatory error. Re-attempt after transformers upgrade or with a custom forward-hook collector. |
| 16 dataloader workers (vs 8) | no change | the residual ~50 ms `timing/data` is not worker throughput (see below) |

## Where the remaining 178 ms/step lives (497.6 vs 320 target)

1. **VLM eager launch overhead, ~120-160 ms**: the 32-layer hybrid text stack
   + fla glue still issues ~4-8k kernels/step with 10-50 µs gaps. The fix is
   compiling the text stack — currently blocked by the transformers 5.3
   hidden-states capture issue above. This is the single biggest known lever.
2. **ZeRO-2 optimizer tail, ~53 ms serial GPU**: param AllGather (43.8 ms,
   2x 7.5 GB @ 300 GB/s) + fused Adam (9.2 ms). Next forward depends on the
   gathered params, so this cannot overlap compute; it surfaces as the
   constant ~50 ms `timing/data` (accelerate's `send_to_device` for the next
   batch queues behind it on the default stream). Reducing it requires
   overlapping the gather with the start of forward (custom DeepSpeed
   surgery / ZeRO-3-style prefetch) or a faster collective.
3. **Episodic host stalls, ~20-40 ms amortized**: 260-375 ms silent stalls in
   the `ncclAllReduce` enqueue path on ~10-20% of steps (documented in the
   profile doc §5.4); root cause open (suspect kernel-level page-fault /
   reclaim; needs `perf` on a non-nsys run).
4. **Exposed grad allreduce, ~29 ms**: shrinks automatically as backward
   compute densifies (item 1).

A realistic path to ~320 ms/step: fix item 1 (compile the text stack after a
transformers upgrade, or hand-collect hidden states via hooks) ≈ -100-140 ms,
plus item 2/3 ≈ -40-60 ms. The hardware floor at 40% MFU remains ~140 ms/step.

## Per-batch experiment history

| Run | Config | wall ms/step | samples/s |
|---|---|---:|---:|
| `bench_baseline_20260703` | as-shipped | 569.1 | 112.5 |
| `bench_A_timer_loss_20260703` | +timer sync off, loss gating | 543.1 | 117.8 |
| `bench_B_collate_20260703` | +collate preprocessing (4 workers) | 536.7 | 119.2 |
| `bench_C_vlm_20260703` | +8 workers, lm_head skip, full FA2 | 539.3 | 118.7 |
| `bench_E_compile_20260703` | C + sdpa + compile DiT + pad192 | 523.3 | 122.3 |
| `bench_F_comm_20260703` | E + NVLS + multi-rank-off (reverted) | 560.7 | 114.1 |
| `bench_FINAL_optimized` | E + mixed attention | 497.6 | 128.6 |
| `bench_I_compile_fwd` | + VLM text-stack compile (see update below) | **481.7** | **132.9** |

## Update (same day): VLM text-stack compile UNBLOCKED on transformers 5.3

Root cause of the bench_G failures found by reading
`transformers/utils/output_capturing.py`: hidden states are collected by
forward hooks installed via `isinstance(module, Qwen3_5DecoderLayer)`
matching. `torch.compile(module)` replaces layers with `OptimizedModule`,
the isinstance match fails, no hooks are installed, and `hidden_states`
comes back truncated.

Fix (no transformers upgrade needed): compile the **bound method**, not the
module — `layer.forward = torch.compile(layer.forward, dynamic=False)` —
which preserves module class identity; the capture hooks fire at the
`__call__` level, outside the compiled region
(`transformers`' `CompileableContextVar` is explicitly designed for this).
Implemented behind `framework.qwenvl.compile_language_model=true` in
`QWen3_5.py`; coexists with `compile_dit`. Pair with
`--datasets.vla_data.collate_pad_to 192` for static shapes.

Result (`bench_I_compile_fwd`, 100 steps): 481.7 ms/step wall,
132.9 samples/s; loss trajectory unchanged; step-1 compile cost ~190 s
(inductor cache persists in the container, subsequent runs cheaper).
The 200-samples/s reference (Agibot) is confirmed to use exactly this pair
of tricks — "dataloader 掩盖 + element-wise kernel torch compile 加速",
with the same first-rounds-JIT-then-stable behavior we observe.

## Round 3 (same day): pipeline overlap + autotune — 459.0 ms / 139.4 samples/s

| Run | Change | wall ms/step | samples/s |
|---|---|---:|---:|
| `bench_J2_hostbatch` | `HostPinnedBatch` + side-stream H2D | 469.1 | 136.4 |
| `bench_L_deferag` | + deferred action-head param allgather | 467.4 | 136.9 |
| `bench_N_maxautotune` | + compile mode `max-autotune-no-cudagraphs` | **459.0** | **139.4** |

1. **`HostPinnedBatch`** (`lerobot_datasets.py`, opt-in
   `datasets.vla_data.collate_host_batch`): the collated batch is wrapped in
   an object that accelerate's `send_to_device` ignores (no `.to`, not a
   Mapping) but the DataLoader pin thread still pins (custom `pin_memory()`).
   `Qwen_PI.forward` moves the tensors on a dedicated copy stream, so the
   H2D overlaps the ZeRO-2 optimizer tail instead of blocking `next()` on it
   (`timing/data` 52 → 0 ms; net wall −12.6 ms — part of the wait relocates
   to the first `.tolist()` readback inside the HF vision path, which is
   unavoidable without patching transformers: any early GPU readback drains
   the default stream through the tail).
2. **Deferred param allgather** (`scripts/starvla/ds_defer_allgather_patch.py`,
   opt-in `STARVLA_DEFER_AG=1`): ZeRO-2's post-step allgather of the
   action-head param group (2.98B params) is launched async and waited at
   the head's forward entry, hiding it behind the VLM forward. Group
   selection is by numel (largest group = backbone stays synchronous).
   Verified: loss trajectory unchanged; net −1.7 ms (the async gather
   contends with forward kernels for SMs, eroding the theoretical ~20 ms).
3. **`max-autotune-no-cudagraphs`** compile mode for the VLM layers + DiT
   (`framework.qwenvl.compile_mode` / `framework.action_model.compile_mode`):
   −8.4 ms steady-state; cost: step-1 JIT grows to ~19 min per fresh
   container (inductor cache persists inside the container afterwards).
   Long training jobs amortize this trivially; short smoke-tests should use
   the default mode.

Blocked this round (documented in code):
- **Vision-tower compile**: HF 5.3 computes `max_seqlen` as a GPU tensor
  inside the block; flash-attn's custom op requires SymInt → dynamo hard
  error (`compile_vision` flag raises with explanation).
- **grid_thw on CPU** (to eliminate readbacks): the vision path derives
  `cu_seqlens` from it and FA2 requires that on CUDA → crash; keeping grid
  on GPU costs one tail-drain readback per step.

Remaining (unchanged from §"Where the remaining ms live", now ~140 ms to
target): residual VLM launch/graph-break overhead (fla ops break each
linear-attn layer into segments), the serial optimizer tail minus what the
defer hides, and the episodic host stalls (root cause still open; needs a
`perf` session).

## Round 4 (same day): bs=16 datapoint — **223.3 samples/s, above the 200 target**

`bench_O_bs16` (same flags as bench_L, default compile mode, `BS=16`):
573.3 ms/step wall at global batch 128 → **223.3 samples/node/s**
(vs 139.4 at bs 8). Fixed costs (launch overhead, optimizer tail, step
boundary) amortize over 2x samples and GEMMs run closer to roofline.
Loss trajectory healthy. Matches the customer's own H20 scaling table
pattern (43.5 samples/s @ bs8 → 50.2 @ bs16 on H20).

IMPORTANT caveat: bs16 changes the global batch (64 → 128) and therefore
the optimization trajectory — this is a systems datapoint, not a free
lunch; needs training-quality sign-off, and the bs convention of the
Agibot 200-samples/s figure must be confirmed (their spec says
batch_size=8, presumably per-GPU, in which case our apples-to-apples
number is 139.4 and the remaining gap at bs8 stands).

Also this round:
- fla 0.3.2's `chunk_gated_delta_rule` carries an explicit
  `@torch.compiler.disable` (fla/ops/gated_delta_rule/chunk.py:220) — the
  per-layer graph breaks are upstream-intentional; forcing through is
  high-risk/low-reward (~5-15 ms estimate). Custom-kernel work is NOT
  currently justified: in-kernel GEMM efficiency is already ~57% of peak,
  fla/FA2 are hand-written kernels, and the deficit is orchestration
  (idle/serialization), not kernel quality.
- Episodic-stall hypothesis test: /proc/vmstat sampled at 2 Hz during a
  full run — allocstall/compact_stall/pgmajfault/thp_fault_fallback deltas
  all ZERO → memory-reclaim/THP/page-fault causes ruled out.
  `perf_event_paranoid=-1` on this host, so a `perf sched`/cycles session
  on a stall-exhibiting run is the next diagnostic (stalls did not occur
  in this round's runs).

## Round 5 (same day): CUDA Graphs via inductor `reduce-overhead` — 432.9 ms / 147.9 samples/s

Feasibility analysis conclusion, validated empirically:
- **Whole-step manual capture (`torch.cuda.graph`)**: NOT feasible on this
  stack — HF 5.3 has per-forward CPU readbacks (grid_thw `.tolist()` etc.,
  capture-fatal) and DeepSpeed ZeRO-2 interleaves Python bookkeeping and
  bucket collectives through backward. Weeks-scale rewrite.
- **Regional graphs via `torch.compile(mode="reduce-overhead")`**: WORKS.
  The feared ZeRO-2 grad-hook-during-capture conflict did not materialize
  (inductor cudagraph-trees tolerated the interleaved bucket allreduces on
  both the DiT and the 32 text-layer graphs). Loss trajectories unchanged.
  - DiT only (`bench_P`): neutral (head already fully fused by compile).
  - VLM layers + DiT (`bench_Q` probe, `bench_R` 100-step):
    **432.9 ms/step, 147.9 samples/s** — the biggest single-change gain of
    the campaign (−26 ms vs max-autotune best).
  - One-time cost: step 1 ≈ 190 s inductor JIT + step 2 ≈ 176 s graph
    capture, per fresh container. Trivially amortized by real training runs.
  - Preconditions already in place from earlier rounds: fixed shapes
    (collate_pad_to 192 / pad_encoder_seq_to 192) and no `.item()` in the
    compiled regions. A sample exceeding 192 text tokens would trigger
    recompile+recapture (not observed in libero_goal).
- **Custom kernels remain unjustified** (see Round 4): the win here came
  from launch elimination, not faster kernels.

Best config now: Round-3 flags with `compile_mode reduce-overhead` for both
`framework.qwenvl.compile_mode` and `framework.action_model.compile_mode`.

## Round 6: milestone2 profile, comm re-tuning — 423.8 ms / 151.0 samples/s

`profiles/qwenpi_milestone2/` (rank0 full nsys trace of the cudagraph config)
shows the post-cudagraph step structure: total kernel time 369→257 ms/step,
cudaLaunchKernel 11k→5k/step (176 cudaGraphLaunch), forward wall 239→173 ms,
backward 368→242 ms, **and the episodic 260-375 ms host stalls are GONE**
(max gap now 26 ms — the problematic ncclAllReduce enqueue path was replaced
by graph replay). The regime flipped to comm-heavy: NCCL busy 127 ms/step
(AllReduce 82 + AllGather 45), nominal exposure 101 ms; new cost: cudagraph
input copy-in (~45 ms/step CPU in cudaMemcpyAsync calls, 1955/step).

A/Bs on the new regime (100 steps each):

| Change | wall ms/step | verdict |
|---|---:|---|
| `use_multi_rank_bucket_allreduce=false` | 500.8 | REJECTED (again; 120 small reduces are latency-bound — permanently closed) |
| `reduce_bucket_size` 5e8 → **1e9** (15 → 8 buckets) | **423.8** | **ADOPTED** (higher busbw, less queueing) |
| `reduce_bucket_size` 2e9 (4 buckets) | 428.5 | rejected (tail exposure grows) |

bs16 with the cudagraph config (`bench_S`, bucket 5e8): 560.0 ms/step at
global batch 128 = **228.6 samples/node/s** (MFU ~18.7%); expect slightly
more with the 1e9 bucket.

## Scoreboard (bs 8/GPU, global batch 64, no nsys, steps 21-100)

| Milestone | ms/step | samples/s | MFU |
|---|---:|---:|---:|
| as-shipped baseline | 569.1 | 112.5 | 9.5% |
| compile + pipeline rounds (bench_N) | 459.0 | 139.4 | 11.4% |
| + CUDA Graphs (bench_R) | 432.9 | 147.9 | 12.1% |
| + reduce_bucket 1e9 (bench_U) | **423.8** | **151.0** | **12.4%** |
| bs16 + cudagraphs (bench_S, global batch 128) | 560.0 | **228.6** | ~18.7% |
| Agibot reference (bs convention TBC) | 320 | 200 | ~16% |

## Round 7: NCCL algorithm sweep — communication is AT THE HARDWARE WALL

Method: microbenchmark matching the exact training collective shapes
(`scripts/starvla/nccl_microbench.py` + `nccl_sweep.sh`, torchrun 8-rank
inside the container, ~40 s per config), winner validated in training.

| Config | AR 2GB busbw | AG 9.08GB busbw |
|---|---:|---:|
| baseline (NCCL auto) | **472.4 GB/s** | 358.1 GB/s |
| NCCL_ALGO=NVLS | 472.7 | 357.4 |
| NCCL_MIN_NCHANNELS=24 / 32 | 472.6 / 471.7 | 358.5 / **365.3** |
| NCCL_BUFFSIZE 8M / 16M | ±0 | ±0 |
| NCCL_CGA_CLUSTER_SIZE=2 | ±0 | 345 (worse) |

Conclusions (all permanently settled):
1. **AllReduce runs at 472 GB/s busbw — above the 450 GB/s nominal NVLink
   line rate.** NCCL 2.21.5's auto-tuner is already optimal; NVLS is
   identical; no protocol/algorithm/buffer knob moves it. Isolated 2 GB
   bucket = 7.4 ms → 8-bucket wire floor 59 ms/step; AllGathers 22.2 + 14.6
   = 37 ms. **Total comm wire floor ≈ 96 ms/step** for ZeRO-2 full-unfreeze
   7.52B at global batch 64 — this is physics, not tuning.
2. The in-training AR inflation (7.4 → ~9-10 ms/bucket) is **inter-rank
   arrival skew** (kernel waits for the slowest rank), not bandwidth.
3. The only in-training positive (chan32, +2% AG isolated) REGRESSED in the
   full run (429.2 vs 423.8 ms — SM contention during overlap eats it).
   Not adopted; leave NCCL fully on auto.
4. Further comm reduction is STRUCTURAL only: gradient accumulation
   (halves comm per optimizer step; changes semantics), gradient
   compression (unsupported on this stack), partial freezing (model
   change), or larger batch (amortizes; see bs16 row).

Remaining ~104 ms to the 320 ms target (from the milestone2 decomposition):
1. Exposed AllReduce tail (~50-70 ms after the bucket fix): backward is now
   shorter than the comm pipeline; options: NVLS-for-AllReduce-only retest
   (untested alone in this regime), NCCL channel tuning, or grad-accum 2
   (halves comm per optimizer step; changes semantics).
2. cudagraph input copy-in (~45 ms/step CPU): inherent to 33 separate
   compiled regions; would shrink if the text stack could compile as fewer
   graphs (blocked by the transformers 5.3 hidden-states capture design).
3. 100-500 µs gap pool (~54 ms): between graph replays / eager glue.
4. The recurring ~22 ms backward gap (constant size, most steps) — likely
   the first-bucket comm wait; visible in the milestone2 timeline for GUI
   inspection.

Note on metrics: `timing/model` improved far more than wall (549→~400 ms
typical) because two costs merely moved out of `_train_step` and re-surfaced
at the step boundary (`timing/data` ≈ 50 ms = the optimizer tail the CPU now
waits on inside `next(dataloader)`); wall-clock regression over steps 21-100
is the only honest end-to-end number.

## Round 8 (2026-07-05, h200-nvl 4u8g-gen-0029): torch 2.7 exploration & fused-stack groundwork

Node change: viking-cr-196 went SLURM-draining; this round ran on an H200 NVL
box with a split 4+4 NVLink topology (comm 12x slower, `NCCL_P2P_LEVEL=NVL`
mandatory — see the handoff doc). Numbers from this round are NOT comparable
with viking; its lasting outputs are mechanisms, not ms:

- torch 2.7.1 + NCCL 2.26 stack: -13% on that box (1751→1520 ms).
- `fused_text_stack_patch.py`: 32 text layers compiled in groups. Blocked on
  stock fla by `@torch.compiler.disable` graph breaks x cudagraph_trees
  allocator checkpointing x ZeRO-2 hook allocations (crashes on torch 2.6 AND
  2.7.1; fla-core 0.5.1 still ships the disable). STARVLA_FLA_TRACE=1 fixes
  it: remove the decorator (container edit) + constant-fold
  `check_shared_mem`/`get_multiprocessor_count` → break-free single graph per
  group, verified fullgraph fwd+bwd.
- Whole-stack single graph measured WORSE than groups: a monolithic backward
  graph releases all 7.5B gradients only at graph end, killing ZeRO-2
  comm/compute overlap. Groups of 8 (4 graphs) are the sweet spot.

## Round 9 (2026-07-06, viking-prod-260): validation + fused vision — **~298 ms / 211-218 samples/s at bs8, customer 200 target crossed**

Fresh viking-class node (H200 SXM, full NVLink). All numbers = timing/model
mean over steps 21-60, 60-step runs, run-to-run spread ±9 ms.

| run | config | ms/step | samples/s |
|---|---|---|---|
| A | committed best (torch 2.6, items 1-11) | 422.5 | 151.5 — reproduces 423.8/151.0 |
| B | A + [OPT #12] MRoPE precompute | 415.9 | 153.9 |
| C | [OPT #13] torch 2.7.1 stack | 367.7 | 174.1 |
| D | C + [OPT #14] fused text stack g8 | 336.1 / 353.8 | ~185 |
| F | D + [OPT #16] reduce_bucket 1.5e9 | 341.3 / 344.7 | ~186 (variance 17.7→3.4 ms) |
| G | F + [OPT #15] fused vision tower | **302.3 / 294.2** | **211.7 / 217.6** |

Item details and LIMITATIONS:

- **[OPT #12] MRoPE position-id precompute** (`collate_mrope_posids`):
  compute the (3,B,T) 3D position ids on CPU in DataLoader workers via a
  weightless Qwen3_5Model shim + exact-key per-sample cache (keys:
  mm_token_type_ids row, attention_mask row, grid bytes — HF never reads
  token values), ship them in qwen_inputs; HF skips its per-step
  compute_3d_position_ids (python loop + 3x .item()/image + .tolist()/row).
  -6.6 ms on the per-layer stack; NEUTRAL on the fused stack (the phase
  overlaps better there). Verified bitwise vs HF (unit + in-vivo
  STARVLA_CHECK_POSIDS across 8 ranks). Limitations: Qwen3.5-only
  (model_type-gated — Qwen2.5-VL has time_interval=4 mrope semantics and
  would silently corrupt), requires preprocess_in_collate, works on the
  customer-pinned torch 2.6 stack (only deviation-free item this round).
- **[OPT #13] torch 2.7.1 stack**: -55 ms (-13%) even on full NVLink (triton
  3.3 TMA fla kernels + inductor). Limitations: DEVIATES from the
  Agibot-pinned versions — torch 2.6.0→2.7.1, flash_attn
  2.7.4.post1→2.8.0.post2 (ABI), causal-conv1d 1.5.0.post8→1.5.2 (ABI) —
  needs customer sign-off; container-level change, not in this repo.
- **[OPT #14] fused text stack, groups of 8** (`STARVLA_FUSED_TEXT_STACK=1
  STARVLA_FLA_TRACE=1 STARVLA_FUSED_GROUP_SIZE=8`, compile_language_model
  false): -22 ms of inter-graph glue (hooks, per-graph input copies, eager
  fla launches). Limitations: requires #13 + the container fla edit
  (untested on torch 2.6); training path only (cache/generation falls back
  to stock HF); STATIC SHAPES required — collate_pad_to fixed,
  per-GPU batch size constant through the run (any constant bs works, 8 and
  16 both fine; what breaks CUDA Graphs is shape VARIATION, not a specific
  bs — avoid partial last batches / dynamic seq); group size trades glue vs
  ZeRO-2 overlap (1 graph = worst, 8 = sweet spot, ~matches VLM grad bucket
  count); gradient checkpointing path falls back.
- **[OPT #15] fused vision tower** (`STARVLA_FUSED_VISION=1`): 24 ViT blocks
  + FA2 varlen in ONE reduce-overhead graph. Unblocked by two host-constant
  folds: (a) max_seqlen (GPU 0-dim tensor vs flash custom-op SymInt) → cached
  host int per image_grid_thw; (b) transformers lazy_import_flash_attention
  (importlib at call time) → prewarmed dict. -45 ms, the single biggest item
  this round. Limitations: verified on torch 2.7.1 + flash_attn 2.8.0.post2
  ONLY (the old 2.6 + 2.7.4 stack hard-fails on the flash custom op); fixed
  camera resolutions assumed — each distinct image_grid_thw costs one
  recompile + graph re-capture (fine for a handful of camera setups, wrong
  workload shape for arbitrary-resolution data); vision
  output_hidden_states/attentions falls back to eager; grid tensor takes one
  tiny D2H per step for the cache key (replaces the .item() sync the eager
  flash path already paid).
- **[OPT #16] reduce_bucket_size 1e9→1.5e9**: DiT head (2.985B params, FIRST
  gradients out in backward) now fills exactly 2 ipg buckets = DeepSpeed's
  two ping-pong overlap buffers (`stage_1_and_2.py` swaps `ipg_index`
  between 0/1); at 1e9 the third DiT bucket could stall autograd waiting for
  a free buffer on skew-heavy steps. Mean-neutral (343.0 vs 345.0) but
  variance collapsed (range 17.7→3.4 ms). Limitation: the value is COUPLED
  to the action-head size — if the DiT param count changes, retune so the
  head's gradients fill ≤2 buckets.

Profiles: `profiles/milestone3` (config F) and `profiles/milestone4`
(config G) on HF `qihankang/startVLA_profile`, both captured with the new
`NSYS_CUDA_GRAPH_TRACE=node` knob (per-kernel visibility inside CUDA
graphs; analysis note — with the default `graph` mode, in-graph kernels
live in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, not the KERNEL table).

Cumulative: original ~800 ms (80 samples/s) → committed 423.8 (151) →
**~298 ms (211-218 samples/s), 2.7x, bs8 spec-compliant, target 200 met.**
Remaining levers: NCCL wire floor (~96 ms), DiT/optimizer segment,
preamble/merger eager glue.

## Files touched

- `starVLA/config/deepseeds/ds_config.yaml` — timers section
- `starVLA/training/train_starvla.py` — loss.item gating
- `starVLA/dataloader/lerobot_datasets.py` — `QwenVLPreprocessCollate`
  (opt-in; `pad_to`, bf16 pixels, `keep_examples`, timing probe)
- `starVLA/dataloader/__init__.py` — collate wiring from config
- `starVLA/model/framework/VLM4A/QwenPI.py` — pre-collated batch path +
  `pad_encoder_seq_to`
- `starVLA/model/modules/vlm/QWen3_5.py` — attn config honored + `mixed`,
  `logits_to_keep`, compile_language_model guard
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` —
  `compile_dit` flag
- `scripts/starvla/profile_qwenpi_zero2_single_node.sh` — `LOGGING_FREQ`,
  `EXTRA_TRAIN_ARGS`, NCCL env passthrough, persistent `NCCL_DEBUG_FILE`
- `tools/bench_report.py`, `tools/analyze_steady_state.py`,
  `tools/analyze_gaps.py` — measurement tooling

All opt-in flags default off, so the default training path is unchanged
except items 1, 2, 4 (pure wins) and 5 (config now honored — note the LIBERO
yaml requests `flash_attention_2`, which is ~10 ms/step slower than `sdpa`
at this seq length; pass `--framework.qwenvl.attn_implementation mixed` or
`sdpa` explicitly for best speed).
