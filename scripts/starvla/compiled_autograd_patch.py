# [EXP card A2, docs/qwenpi_pending_experiments.md]
# Compiled autograd: capture the backward pass (including DeepSpeed ZeRO-2's
# per-param gradient hooks) with dynamo instead of the eager autograd engine.
# Target: the ~445 python hook invocations/step (~11 ms CPU burst, milestone4
# step-24) that ds_grad_copy_stream_patch (#18) identified as the reason the
# side-stream copy experiment measured neutral.
#
# Requires torch >= 2.7. Experiment status, NOT a shipped optimization:
#   - DS hooks execute dist ops + python bucket bookkeeping; dynamo will
#     graph-break around those — the win, if any, comes from compiling the
#     numeric segments in between and de-pythonizing the hook trampoline.
#   - cudagraph modes (reduce-overhead/max-autotune) are unsafe here because
#     collectives fire inside the captured region; default inductor mode only
#     unless STARVLA_CA_MODE overrides (at your own risk).
#   - Decision rule per experiment card: p50 delta > 4 ms on a same-node A/B
#     AND bitwise-identical loss curve over 40 steps.
# Enabled via STARVLA_COMPILED_AUTOGRAD=1 (imported by profile_entry).
import os

import torch
import torch._dynamo.compiled_autograd as _ca
from deepspeed.runtime.engine import DeepSpeedEngine

_MODE = os.environ.get("STARVLA_CA_MODE", "default")
_compile_kwargs = {"dynamic": False}
if _MODE != "default":
    _compile_kwargs["mode"] = _MODE


def _compiler_fn(gm):
    return torch.compile(gm, **_compile_kwargs)


_orig_backward = DeepSpeedEngine.backward


def _backward(self, loss, *args, **kwargs):
    # torch 2.7 renamed the context manager to _enable (private but stable
    # within the pinned stack; signature (compiler_fn, dynamic=False)).
    with _ca._enable(_compiler_fn):
        return _orig_backward(self, loss, *args, **kwargs)


DeepSpeedEngine.backward = _backward
print(f"[compiled_autograd_patch] applied (mode={_MODE})", flush=True)
