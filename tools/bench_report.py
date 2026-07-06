#!/usr/bin/env python3
"""Parse a profile-script command.log into steady-state throughput numbers.

Two estimates:
  1. timing/model + timing/data means over steady steps (excludes logging/tqdm gap)
  2. wall-clock between first/last steady INFO timestamps (second resolution)

Usage: bench_report.py <command.log> [--steady-start 21] [--samples-per-step 64]
"""
import argparse
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--steady-start", type=int, default=21)
ap.add_argument("--samples-per-step", type=int, default=64)
args = ap.parse_args()

text = open(args.log, errors="replace").read()
# Rich wraps lines; join continuation whitespace so key/value pairs survive.
flat = re.sub(r"\s+", " ", text)

# Per-step 'timing/model': value  (associate with preceding 'Step N, Loss')
step_model, step_data = {}, {}
for m in re.finditer(
    r"Step (\d+), Loss:.*?'timing/data':\s*([0-9.e-]+),\s*'timing/model':\s*([0-9.e-]+)", flat
):
    n = int(m.group(1))
    step_data[n] = float(m.group(2))
    step_model[n] = float(m.group(3))

if not step_model:
    sys.exit("no per-step timing found (logging_frequency>1?); falling back unsupported")

max_step = max(step_model)
steady = [n for n in sorted(step_model) if args.steady_start <= n <= max_step]
mt = sum(step_model[n] for n in steady) / len(steady)
dt = sum(step_data[n] for n in steady) / len(steady)
est = mt + dt

# Wall-clock: regress tqdm elapsed [MM:SS] over steady steps (one update per
# step; least-squares slope beats 1s timestamp resolution).
tq = {}
for m in re.finditer(r"(\d+)/\d+ \[(\d{2}):(\d{2})<[^\]]*model_times=[^\]]*\]", text):
    n, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tq[n] = mi * 60 + s  # last occurrence per step wins
wall_line = ""
pts = [(n, t) for n, t in sorted(tq.items()) if args.steady_start <= n <= max(tq)]
if len(pts) >= 10:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    nn = len(pts)
    xbar, ybar = sum(xs) / nn, sum(ys) / nn
    slope = sum((x - xbar) * (y - ybar) for x, y in pts) / sum((x - xbar) ** 2 for x in xs)
    wall_line = (
        f"tqdm-regression steps {xs[0]}..{xs[-1]} (n={nn}): {slope*1000:.1f} ms/step -> "
        f"{args.samples_per_step/slope:.1f} samples/s"
    )

print(f"steps parsed: {len(step_model)} (max {max_step}); steady = {args.steady_start}..{max_step} (n={len(steady)})")
print(f"timing/model mean {mt*1000:.1f} ms + data {dt*1000:.1f} ms = {est*1000:.1f} ms/step -> {args.samples_per_step/est:.1f} samples/s")
if wall_line:
    print(wall_line)
