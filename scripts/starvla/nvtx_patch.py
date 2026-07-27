# Monkeypatch starVLA VLATrainer + QwenPI to emit NVTX ranges and gate nsys
# capture to steady steps [START, END] (skip warmup). Imported by profile_entry.py.
import os
import subprocess
import time

import torch
import starVLA.training.train_starvla as T
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.qwenpi_metadata import env_bool

START = int(os.environ.get("STARVLA_PROFILE_START_STEP", "11"))
END = int(os.environ.get("STARVLA_PROFILE_END_STEP", "13"))
CAPTURE_ENABLED = START > 0 and END >= START
BENCHMARK_WINDOW = env_bool("STARVLA_BENCHMARK_WINDOW")
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

if BENCHMARK_WINDOW and (not CAPTURE_ENABLED or END <= START):
    raise ValueError(
        "STARVLA_BENCHMARK_WINDOW requires a positive profile window with END > START"
    )

_st = {"i": 0, "on": False, "benchmark_start": None}
_emit = {"cm": None}

_orig_step = T.VLATrainer._train_step
def _train_step(self, *a, **k):
    step = _st["i"] + 1
    if BENCHMARK_WINDOW and step == START:
        torch.cuda.synchronize()
        _st["benchmark_start"] = time.perf_counter()
    elif BENCHMARK_WINDOW and step == END:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - _st["benchmark_start"]
        intervals = END - START
        if LOCAL_RANK == 0:
            print(
                "[perf_window] "
                f"start_step={START} end_step={END} intervals={intervals} "
                f"elapsed_s={elapsed:.9f} step_s={elapsed / intervals:.9f}",
                flush=True,
            )
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
        # Opt-in (STARVLA_EMIT_NVTX=1) aten-op shape annotation, scoped
        # EXACTLY to the captured window (steps START..END): every dispatcher
        # op (forward + backward + optimizer) gets an `aten::..., sizes = [...]`
        # NVTX range. Off by default: the per-op dispatcher hook costs real
        # CPU time (~150 ms/step at bs24) and would distort timing-oriented
        # captures. NOTE: ops inside torch.compile / CUDA-Graph regions do
        # not pass through the dispatcher at replay time and get no aten
        # ranges - run the eager config when full GEMM/attention shape
        # coverage is needed.
        if env_bool("STARVLA_EMIT_NVTX"):
            _emit["cm"] = torch.autograd.profiler.emit_nvtx(record_shapes=True)
            _emit["cm"].__enter__()
    torch.cuda.nvtx.range_push(f"train_step_{step}")
    try:
        return _orig_step(self, *a, **k)
    finally:
        torch.cuda.nvtx.range_pop()
        if CAPTURE_ENABLED and step == END and _st["on"]:
            if _emit["cm"] is not None:
                _emit["cm"].__exit__(None, None, None)
                _emit["cm"] = None
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

# Opt-in (STARVLA_NVTX_ACTION_HEAD=1) action-head phase ranges, for module-level
# fwd/bwd attribution. Forward: plain range around the head's forward. Backward:
# identity autograd Functions mark the region — push when grad reaches the DiT
# OUTPUT (backward enters the head), pop once ALL marked INPUTS (hidden_states +
# every per-layer encoder tensor) have received their grads (backward leaves the
# head and the VLM's backward can begin).
if env_bool("STARVLA_NVTX_ACTION_HEAD"):
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
        LayerwiseFlowmatchingActionHead as _LFM,
    )
    from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT as _DiT
    from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface as _QVL

    _orig_qvl_fwd = _QVL.forward
    def _qvl_forward(self, *a, **k):
        torch.cuda.nvtx.range_push("qwenvl_fwd")
        try:
            return _orig_qvl_fwd(self, *a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    _QVL.forward = _qvl_forward

    _orig_lfm_fwd = _LFM.forward
    def _lfm_forward(self, *a, **k):
        torch.cuda.nvtx.range_push("action_head_fwd")
        try:
            return _orig_lfm_fwd(self, *a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    _LFM.forward = _lfm_forward

    class _BwdMark(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t, cb):
            ctx.cb = cb
            return t
        @staticmethod
        def backward(ctx, g):
            ctx.cb()
            return g, None

    _orig_dit_fwd = _DiT.forward
    def _dit_forward(self, hidden_states=None, encoder_hidden_states=None, *a, **k):
        st = {"n": 0, "target": 0, "open": False}
        def _push():
            if not st["open"]:
                torch.cuda.nvtx.range_push("action_head_bwd")
                st["open"] = True
        def _pop():
            st["n"] += 1
            # instant mark per encoder-grad arrival: reveals whether the VLM's
            # layer backward interleaves with the DiT block backward
            torch.cuda.nvtx.mark(f"vl_emb_grad_{st['n']}/{st['target']}")
            if st["n"] >= st["target"] and st["open"]:
                torch.cuda.nvtx.range_pop()
                st["open"] = False
        if torch.is_grad_enabled() and hidden_states is not None and hidden_states.requires_grad:
            hidden_states = _BwdMark.apply(hidden_states, _pop)
            st["target"] += 1
        if torch.is_grad_enabled() and isinstance(encoder_hidden_states, (list, tuple)):
            marked = []
            for e in encoder_hidden_states:
                if torch.is_tensor(e) and e.requires_grad:
                    e = _BwdMark.apply(e, _pop)
                    st["target"] += 1
                marked.append(e)
            encoder_hidden_states = type(encoder_hidden_states)(marked)
        out = _orig_dit_fwd(self, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, *a, **k)
        if torch.is_grad_enabled() and torch.is_tensor(out) and out.requires_grad:
            out = _BwdMark.apply(out, _push)
        return out
    _DiT.forward = _dit_forward
    print("[nvtx_patch] action-head fwd/bwd NVTX ranges enabled", flush=True)

if CAPTURE_ENABLED:
    print("[nvtx_patch] applied (capture train steps %d..%d, trigger=%s)" % (START, END, TRIGGER), flush=True)
else:
    print("[nvtx_patch] applied (capture disabled)", flush=True)
