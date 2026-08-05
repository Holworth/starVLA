# Multi-tensor gradient-norm for DeepSpeed ZeRO-1/2 (STARVLA_DS_FAST_GRAD_NORM).
#
# DeepSpeedZeroOptimizer.get_grad_norm_direct (stage_1_and_2.py, deepspeed
# 0.16.9) norms the partition one parameter at a time:
#     for g, p in zip(gradients, params):
#         all_norms.append(torch.linalg.vector_norm(g.data.double().detach(), ...))
# `.double()` materialises an fp64 copy of every bf16 gradient (9x the memory
# traffic) and the python loop serialises the partition into 2 launches per
# tensor. Measured at bs32 on 8xH200 (profiles/qwenpi_bs32_kernelmix, node-level
# trace): 357 tensors -> 714 kernels + 131 memsets, 7.5 ms of GPU work smeared
# over a 20.4 ms window (median ~24 ms across steps) in which no other stream is
# active -- reduction has finished, Adam has not started -- so it is fully
# exposed on the step's critical path.
#
# torch._foreach_norm does the same reduction in a few multi-tensor kernels with
# no dtype conversion. The cross-tensor reduction stays fp64 (357 scalars, free)
# because an fp32 square-sum overflows past ~1.8e19; only the per-tensor pass
# moves to fp32 accumulation, which is what torch.nn.utils.clip_grad_norm_ does
# natively. The norm only feeds the clip scale and the reported
# _global_grad_norm; forward/backward math is untouched, but the optimizer
# update is no longer bit-identical, so checks must be tolerance based.
#
# The phase is entirely eager (graphId NULL) and sits between the grad hooks and
# Adam: no CUDA-graph capture, static shape, or ZeRO reduction path is involved.
#
# STARVLA_DS_FAST_GRAD_NORM=0 restores the stock implementation.
# STARVLA_CHECK_DS_GRAD_NORM=1 also runs the stock path and raises on mismatch.
import inspect
import os

import torch
from deepspeed import comm as dist
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime.utils import inf, is_model_parallel_parameter, mask_nan_or_inf_with_val_inplace
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

RTOL = 1e-3
_EXPECTED_SIG = ["self", "gradients", "params", "norm_type"]
_orig = DeepSpeedZeroOptimizer.get_grad_norm_direct
_check = os.environ.get("STARVLA_CHECK_DS_GRAD_NORM", "").strip().lower() not in ("", "0", "false", "no", "off")


def _fast_get_grad_norm_direct(self, gradients, params, norm_type=2):
    norm_type = float(norm_type)
    if norm_type == inf:  # rare path, left to the stock implementation
        return _orig(self, gradients, params, norm_type)

    # identical filtering to upstream: skip pipeline-replicated params, and count
    # non-model-parallel params on rank 0 only
    grads = [
        g.data for g, p in zip(gradients, params)
        if not (hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated)
        and (is_model_parallel_parameter(p) or self.model_parallel_rank == 0)
        and g is not None and g.numel() > 0
    ]
    if grads:
        norms = torch._foreach_norm(grads, norm_type)  # one multi-tensor pass
        total_norm = torch.stack(norms).double().pow(norm_type).sum().float()
    else:
        total_norm = torch.zeros((), dtype=torch.float32, device=self.device)

    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)
    self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)
    total_norm = total_norm.pow(1.0 / norm_type)
    mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)

    if _check:
        ref = _orig(self, gradients, params, norm_type)
        rel = (total_norm - ref).abs() / ref.clamp(min=1e-12)
        if not bool((rel <= RTOL).all()):
            raise RuntimeError(
                f"[ds_grad_norm_patch] grad-norm mismatch: fast={total_norm.item()} "
                f"stock={ref.item()} rel={rel.item()}"
            )
    return total_norm


def _install():
    sig = list(inspect.signature(_orig).parameters)
    if sig != _EXPECTED_SIG:
        # DeepSpeed changed this private API: keep the stock path rather than risk it
        print(f"[ds_grad_norm_patch] unexpected signature {sig}, patch NOT applied", flush=True)
        return
    DeepSpeedZeroOptimizer.get_grad_norm_direct = _fast_get_grad_norm_direct
    print(f"[ds_grad_norm_patch] foreach grad-norm installed (check={_check})", flush=True)


_install()
