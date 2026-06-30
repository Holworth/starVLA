# Monkeypatch starVLA VLATrainer + QwenPI to emit NVTX ranges and gate nsys
# capture to steady steps [START, END] (skip warmup). Imported by profile_entry.py.
import os
import subprocess
import time

import torch
import starVLA.training.train_starvla as T
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

START = int(os.environ.get("STARVLA_PROFILE_START_STEP", "11"))
END = int(os.environ.get("STARVLA_PROFILE_END_STEP", "13"))
CAPTURE_ENABLED = START > 0 and END >= START
TRIGGER = os.environ.get("STARVLA_PROFILE_TRIGGER", "nvtx").lower()
NSYS_BIN = os.environ.get("STARVLA_NSYS_BIN")
NSYS_SESSION = os.environ.get("STARVLA_NSYS_SESSION")
START_STAGGER_SEC = float(os.environ.get("STARVLA_NSYS_START_STAGGER_SEC", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if TRIGGER == "session" and (not NSYS_BIN or not NSYS_SESSION):
    print("[nvtx_patch] session trigger requested without nsys session; falling back to nvtx", flush=True)
    TRIGGER = "nvtx"

def _nsys_session(command):
    if not NSYS_BIN or not NSYS_SESSION:
        raise RuntimeError("STARVLA_NSYS_BIN and STARVLA_NSYS_SESSION are required for session profiling")
    if command == "start" and START_STAGGER_SEC > 0:
        time.sleep(LOCAL_RANK * START_STAGGER_SEC)
    subprocess.run([NSYS_BIN, command, "--session", NSYS_SESSION], check=True)

def _cuda_profiler(command):
    cudart = torch.cuda.cudart()
    if command == "start":
        err = cudart.cudaProfilerStart()
    else:
        err = cudart.cudaProfilerStop()
    if err != 0:
        raise RuntimeError(f"cudaProfiler{command.title()} failed with CUDA error {err}")

_st = {"i": 0, "on": False}

_orig_step = T.VLATrainer._train_step
def _train_step(self, *a, **k):
    step = _st["i"] + 1
    if CAPTURE_ENABLED and step == START:
        print(f"[nvtx_patch] rank {LOCAL_RANK} profiler start at train_step_{step} (trigger={TRIGGER})", flush=True)
        if TRIGGER == "cuda":
            torch.cuda.synchronize()
            _cuda_profiler("start")
            torch.cuda.nvtx.range_push("profile_window")
        elif TRIGGER == "session":
            torch.cuda.synchronize(); _nsys_session("start")
            torch.cuda.nvtx.range_push("profile_window")
        else:
            torch.cuda.nvtx.range_push("profile_window")
        _st["on"] = True
    torch.cuda.nvtx.range_push(f"train_step_{step}")
    try:
        return _orig_step(self, *a, **k)
    finally:
        torch.cuda.nvtx.range_pop()
        if CAPTURE_ENABLED and step == END and _st["on"]:
            if TRIGGER == "cuda":
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                _cuda_profiler("stop")
            elif TRIGGER == "session":
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize(); _nsys_session("stop")
            else:
                torch.cuda.nvtx.range_pop()
            _st["on"] = False
            print(f"[nvtx_patch] rank {LOCAL_RANK} profiler stop at train_step_{step} (trigger={TRIGGER})", flush=True)
        _st["i"] += 1
T.VLATrainer._train_step = _train_step

_orig_get = T.VLATrainer._get_next_batch
def _get_next_batch(self, *a, **k):
    torch.cuda.nvtx.range_push("dataloader")
    try:
        return _orig_get(self, *a, **k)
    finally:
        torch.cuda.nvtx.range_pop()
T.VLATrainer._get_next_batch = _get_next_batch

_orig_fwd = Qwen_PI.forward
def _forward(self, *a, **k):
    torch.cuda.nvtx.range_push("model_forward")
    try:
        return _orig_fwd(self, *a, **k)
    finally:
        torch.cuda.nvtx.range_pop()
Qwen_PI.forward = _forward

if CAPTURE_ENABLED:
    print("[nvtx_patch] applied (capture train steps %d..%d, trigger=%s)" % (START, END, TRIGGER), flush=True)
else:
    print("[nvtx_patch] applied (capture disabled)", flush=True)
