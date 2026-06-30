#!/usr/bin/env bash
# starVLA QwenPI 8卡 bs=8:高频(5Hz)带时间戳采 GPU util,排除 warmup。
# host 侧 nvidia-smi -lms 采样;容器内训练;事后按 train log 的 step 时钟切稳态窗口
# (跳过前 5 步 warmup/编译,只统计 step6→末步)。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/home/chenchaox/project/phyai_fd/data/starvla_libero/util_logs
mkdir -p "$OUT"
WARMUP=5   # 跳过的前 N 步

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 \
  > "$OUT/util_bs8_5hz.csv" 2>/dev/null &
POLLER=$!
"$HERE/run_container.sh" bash /scripts/starvla/train_libero.sh > "$OUT/train_bs8.log" 2>&1 || true
kill "$POLLER" 2>/dev/null || true

python3 - "$OUT/util_bs8_5hz.csv" "$OUT/train_bs8.log" "$WARMUP" <<'PY'
import re,sys,statistics
util_csv,train_log,warmup=sys.argv[1],sys.argv[2],int(sys.argv[3])
def sec(h,m,s): return int(h)*3600+int(m)*60+int(s)
# step -> 时钟秒
step_t={}
for ln in open(train_log,errors='ignore'):
    m=re.search(r'\[(\d\d):(\d\d):(\d\d)\][^|]*Step (\d+)',ln)
    if m: step_t[int(m.group(4))]=sec(*m.group(1,2,3))
if len(step_t)<warmup+2: print("步数不足,无法切窗口"); sys.exit()
steps=sorted(step_t); lo=step_t[steps[warmup]]; hi=step_t[steps[-1]]
# util csv: "YYYY/MM/DD HH:MM:SS.mmm, idx, util, mem"
per={}
for ln in open(util_csv,errors='ignore'):
    p=[x.strip() for x in ln.split(',')]
    if len(p)<4: continue
    t=re.search(r'(\d\d):(\d\d):(\d\d)',p[0])
    if not t or not p[1].isdigit(): continue
    ts=sec(*t.group(1,2,3))
    if lo<=ts<=hi: per.setdefault(int(p[1]),[]).append(int(p[2]))
if not per: print("窗口内无样本"); sys.exit()
gpu=[statistics.mean(v) for v in per.values()]
n=min(len(v) for v in per.values())
print(f"稳态窗口 = step{warmup+1}..{steps[-1]} ({hi-lo}s);每卡样本 ~{n}")
print(f"GPU util(8卡稳态平均) = {statistics.mean(gpu):.1f}%   每卡: {[round(x,1) for x in gpu]}")
PY
echo "util -> $OUT/util_bs8_5hz.csv ; log -> $OUT/train_bs8.log"
