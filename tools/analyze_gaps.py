#!/usr/bin/env python3
"""Drill-down on GPU idle structure in the steady-state window.

Reports per-phase (forward / backward / rest-of-step) GPU busy fraction, idle
gap histogram, top individual gaps, CUDA runtime API launch load, and NCCL
collective shapes.

Usage: analyze_gaps.py <rank0.sqlite> [--steady-start 21] [--steady-end 100]
"""
import argparse
import bisect
import re
import sqlite3
from collections import defaultdict

MS = 1e6
US = 1e3


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


def clip(ivs, lo, hi):
    out = []
    for s, e in ivs:
        s2, e2 = max(s, lo), min(e, hi)
        if s2 < e2:
            out.append((s2, e2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--steady-start", type=int, default=21)
    ap.add_argument("--steady-end", type=int, default=100)
    args = ap.parse_args()

    c = sqlite3.connect(args.sqlite)
    strings = dict(c.execute("SELECT id, value FROM StringIds"))

    steps, fwd, bwd, dl = {}, [], [], []
    for start, end, text, textId in c.execute("SELECT start, end, text, textId FROM NVTX_EVENTS"):
        if end is None:
            continue
        name = ((text if text else strings.get(textId, "")) or "").lstrip(":")
        m = re.fullmatch(r"train_step_(\d+)", name)
        if m:
            steps[int(m.group(1))] = (start, end)
        elif name == "DeepSpeedEngine.forward":
            fwd.append((start, end))
        elif name == "DeepSpeedEngine.backward":
            bwd.append((start, end))
        elif name == "dataloader":
            dl.append((start, end))

    nsteps = max(steps)
    hi_step = min(args.steady_end, nsteps)
    s_lo, s_hi = steps[args.steady_start][0], steps[hi_step][1]
    n_steady = hi_step - args.steady_start + 1

    kerns = [
        (s, e, sid)
        for s, e, sid in c.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL")
        if s_lo <= s < s_hi
    ]
    gpu_iv = [(s, e) for s, e, _ in kerns]
    for tab in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            gpu_iv += [
                (s, e)
                for s, e in c.execute(f"SELECT start, end FROM {tab}")
                if s_lo <= s < s_hi
            ]
        except sqlite3.OperationalError:
            pass
    busy = union_ivs(gpu_iv)
    nccl_u = union_ivs([(s, e) for s, e, sid in kerns if strings.get(sid, "").startswith("nccl")])
    comp_u = union_ivs([(s, e) for s, e, sid in kerns if not strings.get(sid, "").startswith("nccl")])

    def busy_in(lo, hi, u=busy):
        return sum(e - s for s, e in clip(u, lo, hi))

    # --- per-phase split ---
    print("=== per-phase GPU busy within steady window ===")
    for name, ivs in (("forward", fwd), ("backward", bwd)):
        ivs = [iv for iv in ivs if s_lo <= iv[0] < s_hi]
        wall = sum(e - s for s, e in ivs)
        b = sum(busy_in(s, e) for s, e in ivs)
        cb = sum(busy_in(s, e, comp_u) for s, e in ivs)
        nb = sum(busy_in(s, e, nccl_u) for s, e in ivs)
        print(
            f"  {name:<9} wall {wall/MS/n_steady:7.1f} ms/step | gpu busy {100*b/wall:5.1f}% "
            f"| compute {100*cb/wall:5.1f}% | nccl {100*nb/wall:5.1f}%"
        )
    # rest of step = step minus fwd/bwd/dataloader
    rest_wall = rest_busy = rest_comp = rest_nccl = 0
    sub = union_ivs([iv for iv in fwd + bwd + dl if s_lo <= iv[0] < s_hi])
    for sn in range(args.steady_start, hi_step + 1):
        lo, hi = steps[sn]
        cur = lo
        holes = []
        for s, e in clip(sub, lo, hi):
            if s > cur:
                holes.append((cur, s))
            cur = max(cur, e)
        if cur < hi:
            holes.append((cur, hi))
        for s, e in holes:
            rest_wall += e - s
            rest_busy += busy_in(s, e)
            rest_comp += busy_in(s, e, comp_u)
            rest_nccl += busy_in(s, e, nccl_u)
    print(
        f"  {'rest':<9} wall {rest_wall/MS/n_steady:7.1f} ms/step | gpu busy {100*rest_busy/rest_wall:5.1f}% "
        f"| compute {100*rest_comp/rest_wall:5.1f}% | nccl {100*rest_nccl/rest_wall:5.1f}%"
    )

    # --- gap histogram ---
    gaps = []
    cur = s_lo
    for s, e in busy:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < s_hi:
        gaps.append((cur, s_hi))
    buckets = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 1e18)]
    print("\n=== idle gap histogram (steady window) ===")
    total_idle = sum(e - s for s, e in gaps)
    for lo_us, hi_us in buckets:
        sel = [e - s for s, e in gaps if lo_us * US <= (e - s) < hi_us * US]
        if sel:
            print(
                f"  {lo_us:>6.0f}-{hi_us if hi_us < 1e17 else 99999:>6.0f} us: {len(sel):7d} gaps, "
                f"total {sum(sel)/MS:8.0f} ms ({100*sum(sel)/total_idle:5.1f}% of idle, "
                f"{sum(sel)/MS/n_steady:6.1f} ms/step)"
            )
    print(f"  total idle {total_idle/MS:.0f} ms ({total_idle/MS/n_steady:.1f} ms/step)")

    # --- top 12 individual gaps with phase attribution ---
    def phase_of(t):
        for name, ivs in (("forward", fwd), ("backward", bwd), ("dataloader", dl)):
            for s, e in ivs:
                if s <= t < e:
                    return name
        return "rest(optimizer/step)"

    print("\n=== top individual gaps ===")
    for s, e in sorted(gaps, key=lambda g: g[0] - g[1])[:12]:
        sn = "?"
        for k, (lo, hi) in steps.items():
            if lo <= s < hi:
                sn = k
                break
        print(f"  {(e-s)/MS:8.2f} ms  step {sn:>3}  phase={phase_of(s)}")

    # --- CUDA runtime API load (CPU side) ---
    print("\n=== CUDA runtime API within steady window (CPU side) ===")
    api = defaultdict(lambda: [0, 0])
    for s, e, nid in c.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        if s_lo <= s < s_hi:
            a = api[nid]
            a[0] += e - s
            a[1] += 1
    for nid, (t, n) in sorted(api.items(), key=lambda kv: -kv[1][0])[:12]:
        print(
            f"  {t/MS/n_steady:8.2f} ms/step {n/n_steady:9.1f} calls/step  avg {t/n/US:7.1f} us  "
            f"{strings.get(nid, nid)}"
        )

    # --- NCCL kernel shapes ---
    print("\n=== NCCL kernels in steady window ===")
    nk = defaultdict(list)
    for s, e, sid in kerns:
        nm = strings.get(sid, "")
        if nm.startswith("nccl"):
            nk[nm].append(e - s)
    for nm, ds in sorted(nk.items(), key=lambda kv: -sum(kv[1])):
        ds.sort()
        print(
            f"  {nm[:60]:<62} n/step {len(ds)/n_steady:5.1f}  med {ds[len(ds)//2]/MS:7.2f} ms  "
            f"total {sum(ds)/MS/n_steady:7.1f} ms/step"
        )


if __name__ == "__main__":
    main()
