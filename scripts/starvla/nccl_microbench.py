# NCCL microbenchmark matching the QwenPI ZeRO-2 collective shapes:
#   - AllReduce: 1e9-elem bf16 buckets (2 GB) — the reduce_bucket_size=1e9 shape
#   - AllGather: group0 params (4.539e9 elems bf16 -> each rank contributes
#     567.4M elems, output 9.08 GB) and group1 (2.985e9 elems)
# Prints per-op time and busbw so NCCL env combos can be swept quickly
# without a full training run. Launch: torchrun --nproc_per_node=8 nccl_microbench.py
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)

WARMUP, ITERS = 5, 20


def bench(fn, bytes_moved_algo, label):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    busbw = bytes_moved_algo / (ms / 1e3) / 1e9
    if rank == 0:
        print(f"{label:<44} {ms:8.2f} ms   busbw {busbw:7.1f} GB/s", flush=True)


# AllReduce, 2 GB bf16 bucket (reduce_bucket_size=1e9)
n = 1_000_000_000
buf = torch.ones(n, dtype=torch.bfloat16, device=dev)
ar_bytes = 2 * (world - 1) / world * n * 2
bench(lambda: dist.all_reduce(buf), ar_bytes, f"allreduce bf16 {n/1e9:.1f}G elems (2GB)")

# AllReduce, 1 GB bucket for reference (old reduce_bucket_size=5e8)
n2 = 500_000_000
buf2 = torch.ones(n2, dtype=torch.bfloat16, device=dev)
bench(lambda: dist.all_reduce(buf2), 2 * (world - 1) / world * n2 * 2, f"allreduce bf16 {n2/1e9:.1f}G elems (1GB)")
del buf2

# AllGather group0 (VLM 4.539e9 params)
g0 = 4_539_265_536
part0 = g0 // world
inp0 = torch.ones(part0, dtype=torch.bfloat16, device=dev)
out0 = torch.empty(part0 * world, dtype=torch.bfloat16, device=dev)
ag0_bytes = (world - 1) / world * part0 * world * 2
bench(lambda: dist.all_gather_into_tensor(out0, inp0), ag0_bytes, "allgather group0 (9.08GB out)")
del inp0, out0

# AllGather group1 (DiT 2.985e9 params)
g1 = 2_984_958_512
part1 = (g1 // world // 512) * 512
inp1 = torch.ones(part1, dtype=torch.bfloat16, device=dev)
out1 = torch.empty(part1 * world, dtype=torch.bfloat16, device=dev)
bench(lambda: dist.all_gather_into_tensor(out1, inp1), (world - 1) / world * part1 * world * 2, "allgather group1 (5.97GB out)")

dist.destroy_process_group()
