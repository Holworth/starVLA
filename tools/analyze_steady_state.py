#!/usr/bin/env python3
"""Steady-state analysis of a starVLA QwenPI rank0 nsys sqlite export.

Single pass over the kernel table, everything else in memory. Splits the trace
into warmup (train_step_1) and a steady-state window using NVTX train_step_*
ranges, then reports per-window kernel summaries, NCCL exposed time, GPU busy
fraction, per-step marker-kernel distribution, and NVTX phase totals.

Usage: analyze_steady_state.py <rank0.sqlite> [--steady-start 21] [--steady-end 100]
"""
import argparse
import bisect
import re
import sqlite3
from collections import defaultdict

MS = 1e6  # ns per ms


def union_ivs(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out


def union_len(ivs):
    return sum(e - s for s, e in union_ivs(ivs))


def inter_len(a, b):
    a, b = union_ivs(a), union_ivs(b)
    i = j = tot = 0
    while i < len(a) and j < len(b):
        tot += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def overlap(a_lo, a_hi, b_lo, b_hi):
    return max(0, min(a_hi, b_hi) - max(a_lo, b_lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--steady-start", type=int, default=21)
    ap.add_argument("--steady-end", type=int, default=100)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    c = sqlite3.connect(args.sqlite)
    strings = dict(c.execute("SELECT id, value FROM StringIds"))

    # --- NVTX ranges ---
    steps, phases = {}, defaultdict(list)
    phase_names = {
        "DeepSpeedEngine.forward",
        "DeepSpeedEngine.backward",
        "DeepSpeedEngine.allreduce_gradients",
        "DeepSpeedEngine.step",
        "dataloader",
        "model_forward",
    }
    for start, end, text, textId in c.execute(
        "SELECT start, end, text, textId FROM NVTX_EVENTS"
    ):
        if end is None:
            continue
        name = (text if text else strings.get(textId, "")) or ""
        name = name.lstrip(":")
        m = re.fullmatch(r"train_step_(\d+)", name)
        if m:
            steps[int(m.group(1))] = (start, end)
        elif name in phase_names:
            phases[name].append((start, end))

    if not steps:
        raise SystemExit("no train_step_N NVTX ranges found")
    nsteps = max(steps)
    hi_step = min(args.steady_end, nsteps)
    s_lo, s_hi = steps[args.steady_start][0], steps[hi_step][1]
    n_steady = hi_step - args.steady_start + 1

    print(f"steps found: 1..{nsteps}")
    for k in (1, 2, 3):
        if k in steps:
            print(f"step {k}: {(steps[k][1]-steps[k][0])/MS:8.1f} ms")
    durs = sorted((e - s) / MS for n, (s, e) in steps.items() if args.steady_start <= n <= hi_step)
    print(
        f"steady steps {args.steady_start}..{hi_step} (n={n_steady}): "
        f"mean {sum(durs)/len(durs):.1f} ms  med {durs[len(durs)//2]:.1f} ms  "
        f"min {durs[0]:.1f}  max {durs[-1]:.1f}"
    )

    # --- single pass over kernels ---
    kerns = list(
        c.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL")
    )
    mem_iv = []
    for tab in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            mem_iv += list(c.execute(f"SELECT start, end FROM {tab}"))
        except sqlite3.OperationalError:
            pass

    warm_lo, warm_hi = steps[1]
    warm = defaultdict(lambda: [0, 0])
    steady = defaultdict(lambda: [0, 0])
    nccl_iv, comp_iv = [], []
    for s, e, sid in kerns:
        if warm_lo <= s < warm_hi:
            a = warm[sid]
            a[0] += e - s
            a[1] += 1
        if s_lo <= s < s_hi:
            a = steady[sid]
            a[0] += e - s
            a[1] += 1
            (nccl_iv if strings.get(sid, "").startswith("nccl") else comp_iv).append((s, e))

    wall = s_hi - s_lo
    kbusy = union_len(nccl_iv + comp_iv)
    mem_in = [(s, e) for s, e in mem_iv if s_lo <= s < s_hi]
    abusy = union_len(nccl_iv + comp_iv + mem_in)
    cbusy = union_len(comp_iv)
    nbusy = union_len(nccl_iv)
    exposed = nbusy - inter_len(nccl_iv, comp_iv)
    print(
        f"\nsteady window wall {wall/MS:.0f} ms ({wall/MS/n_steady:.1f} ms/step)\n"
        f"  kernel busy   {kbusy/MS:9.0f} ms ({100*kbusy/wall:5.1f}%)   "
        f"+memcpy/memset {abusy/MS:.0f} ms ({100*abusy/wall:.1f}%)\n"
        f"  compute busy  {cbusy/MS:9.0f} ms ({100*cbusy/wall:5.1f}%)\n"
        f"  nccl busy     {nbusy/MS:9.0f} ms ({100*nbusy/wall:5.1f}%)   "
        f"exposed (not overlapped by compute) {exposed/MS:.0f} ms ({100*exposed/wall:.1f}%)\n"
        f"  gpu idle      {(wall-abusy)/MS:9.0f} ms ({100*(wall-abusy)/wall:5.1f}%)"
    )

    tot_steady = sum(v[0] for v in steady.values())
    tot_warm = sum(v[0] for v in warm.values())
    print(
        f"\n=== top kernels, STEADY steps {args.steady_start}..{hi_step} "
        f"(total kernel time {tot_steady/MS:.0f} ms, {tot_steady/MS/n_steady:.1f} ms/step) ==="
    )
    print(f"{'time%':>6} {'ms/step':>9} {'inst/step':>10} {'avg_us':>8}  name")
    for sid, (t, n) in sorted(steady.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(
            f"{100*t/tot_steady:6.1f} {t/MS/n_steady:9.2f} {n/n_steady:10.1f} "
            f"{t/n/1e3:8.1f}  {strings.get(sid, str(sid))[:95]}"
        )
    print(f"\n=== top kernels, WARMUP step 1 (total kernel time {tot_warm/MS:.0f} ms) ===")
    for sid, (t, n) in sorted(warm.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"{100*t/tot_warm:6.1f} {t/MS:9.1f}ms {n:8d} inst  {strings.get(sid, str(sid))[:95]}")

    # --- marker kernel distribution across steps ---
    marker_sids = [sid for sid in set(list(warm) + list(steady)) if "FillFunctor<int>" in strings.get(sid, "")]
    if marker_sids:
        step_keys = sorted(steps)
        starts = [steps[k][0] for k in step_keys]
        per_step = defaultdict(lambda: [0, 0])
        mset = set(marker_sids)
        for s, e, sid in kerns:
            if sid in mset:
                i = bisect.bisect_right(starts, s) - 1
                if i >= 0 and s < steps[step_keys[i]][1]:
                    a = per_step[step_keys[i]]
                    a[0] += e - s
                    a[1] += 1
        name = strings[marker_sids[0]]
        print(f"\n=== '{name[:70]}' per step (ms | count) ===")
        for sn in step_keys:
            if sn <= 5 or sn % 10 == 0 or sn == nsteps:
                t, n = per_step.get(sn, (0, 0))
                print(f"  step {sn:>3}: {t/MS:8.1f} ms  {n:6d} inst")

    # --- NVTX phase totals within steady window ---
    print("\n=== NVTX phase totals within steady window (CPU-side ranges) ===")
    for name in sorted(phases):
        ivs = phases[name]
        tot = sum(overlap(s, e, s_lo, s_hi) for s, e in ivs)
        cnt = sum(1 for s, e in ivs if s_lo <= s < s_hi)
        if cnt:
            print(f"  {name:<40} {tot/MS/n_steady:8.1f} ms/step  ({cnt} inst)")


if __name__ == "__main__":
    main()
