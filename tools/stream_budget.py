import sqlite3
from collections import defaultdict
import sys
# Per-stream kernel accounting + wall-clock ledger for one train step.
# Usage: python tools/stream_budget.py <rank0.sqlite> <step_no>
# Rationale: streams overlap, so kernel times must NEVER be summed across
# streams into one percentage pool — hidden comm costs zero wall time.
# Ledger identity: wall = compute-side busy + exposed comm + true idle.
c = sqlite3.connect(sys.argv[1])
s0, s1 = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?", (f'train_step_{sys.argv[2]}',)).fetchone()
wall = (s1-s0)/1e6

def family(n):
    if n.startswith('ncclDevKernel'): return 'NCCL ' + ('AllReduce' if 'AllReduce' in n else 'AllGather' if 'AllGather' in n else 'other')
    if n.startswith('nvjet'): return 'GEMM nvjet_*'
    if 'fmha' in n or n.startswith(('flash_fwd','flash_bwd')): return 'attention (FA2/SDPA)'
    if n.startswith(('chunk_','recompute_w_u','prepare_wy','bwd_prepare','fwd_prepare','merge_16x16')): return 'fla (triton)'
    if n.startswith('conv_depthwise2d'): return 'fla causal-conv'
    if n.startswith('triton_'): return 'inductor fused'
    if n.startswith('multi_tensor_apply'): return 'fused adam'
    if 'CatArrayBatchedCopy' in n: return 'cat/flatten'
    if 'elementwise' in n or n.startswith(('reduce_kernel','vectorized','unrolled')): return 'eager elemwise'
    return 'other'

evs = []   # (stream, start, end, family)
rows = c.execute("""SELECT k.streamId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    JOIN StringIds s ON k.shortName=s.id
                    WHERE k.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=?)""",(s0,s1)).fetchall()
for sid, a, b, n in rows: evs.append((sid, a, b, family(n)))
for kind, lab in ((1,'memcpy H2D'),(2,'memcpy D2H'),(8,'memcpy D2D')):
    for sid, a, b in c.execute("""SELECT streamId, start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=?
        AND correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=?)""",(kind,s0,s1)):
        evs.append((sid, a, b, lab))

def union(iv):
    if not iv: return 0.0, []
    iv = sorted(iv); out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return sum(b-a for a, b in out)/1e6, out

streams = defaultdict(list)
for sid, a, b, f in evs: streams[sid].append((a, b, f))

print(f"step wall (nsys 下): {wall:.1f} ms\n")
print(f"{'stream':>7s} {'角色':20s} {'busy ms':>8s} {'%wall':>6s} {'kernels':>8s}  top families(流内占比)")
roles = {}
srt = sorted(streams.items(), key=lambda kv: -sum(b-a for a,b,_ in kv[1]))
comp_iv, comm_iv = [], []
for sid, lst in srt:
    fam = defaultdict(float); n = len(lst)
    for a, b, f in lst: fam[f] += (b-a)/1e6
    busy, _ = union([(a,b) for a,b,_ in lst])
    nccl = sum(v for k, v in fam.items() if k.startswith('NCCL'))
    role = ('NCCL 通信流' if nccl/max(busy,1e-9) > 0.5 else
            'H2D 侧流' if fam.get('memcpy H2D',0)/max(busy,1e-9) > 0.5 else
            '主计算流' if busy > 50 else '辅助流')
    roles[sid] = role
    if role == 'NCCL 通信流': comm_iv += [(a,b) for a,b,_ in lst]
    else: comp_iv += [(a,b) for a,b,_ in lst]
    top = ', '.join(f"{k} {v:.1f}ms({100*v/busy:.0f}%)" for k, v in sorted(fam.items(), key=lambda kv:-kv[1])[:3])
    print(f"{sid:7d} {role:18s} {busy:8.1f} {100*busy/wall:5.1f}% {n:8d}  {top}")

comp_busy, comp_merged = union(comp_iv)
comm_busy, _ = union(comm_iv)
# exposed comm: comm intervals minus compute-union
def subtract(iv, cover):
    total = 0
    for a, b in sorted(iv):
        cur = a
        for ca, cb in cover:
            if cb <= cur: continue
            if ca >= b: break
            if ca > cur: total += min(ca, b) - cur
            cur = max(cur, cb)
            if cur >= b: break
        if cur < b: total += b - cur
    return total/1e6
exposed = subtract(sorted(comm_iv), comp_merged)
all_busy, _ = union(comp_iv + comm_iv)
print(f"\n=== 重叠/裸露分析(相对墙钟 {wall:.1f} ms)===")
print(f"计算侧流合并 busy : {comp_busy:7.1f} ms  ({100*comp_busy/wall:.1f}% of wall)")
print(f"NCCL 流合并 busy  : {comm_busy:7.1f} ms  ({100*comm_busy/wall:.1f}% of wall)")
print(f"  其中被计算隐藏  : {comm_busy-exposed:7.1f} ms")
print(f"  其中裸露(计算流空转时通信在跑): {exposed:7.1f} ms  ({100*exposed/wall:.1f}% of wall)")
print(f"全设备 union busy : {all_busy:7.1f} ms")
print(f"真空转(所有流都闲): {wall-all_busy:7.1f} ms  ({100*(wall-all_busy)/wall:.1f}% of wall)")
