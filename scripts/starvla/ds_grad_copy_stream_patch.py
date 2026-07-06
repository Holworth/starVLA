# [OPT #18, docs/qwenpi_zero2_h200_optimization_log.md Round 9 run I]
# Move ZeRO-2's gradient bucket-fill copies off the compute stream.
#
# Stock DeepSpeed (contiguous_gradients) copies every produced gradient into
# the ipg bucket INLINE on the compute stream: with fused backward graphs all
# of a graph's grads materialize at once, so the hooks fire in a burst and the
# compute stream spends ~12 ms/step executing 800+ small DtoD memcpys while
# the SMs idle (milestone4 step-24: 445 copies / 5.7 GB serialized between the
# DiT backward graph and the next text-stack graph, plus ~6 ms of launch gaps
# from issuing them one by one).
#
# This patch defers the copies: hooks only record (src, dst) pairs and repoint
# param.grad to the bucket view exactly like stock; the pairs are flushed with
# torch._foreach_copy_ on a dedicated copy stream right before the bucket is
# reduced. Ordering (all against stock invariants in stage_1_and_2.py):
#   1. copy_stream.wait_stream(compute) at flush — every source gradient was
#      produced on the compute stream before the flush point;
#   2. compute.wait_stream(copy_stream) before reduce_ipg_grads — DS's
#      average_tensor already does reduction.wait(compute), so the reduction
#      transitively waits for the copies;
#   3. buffer reuse stays safe through DS's existing bidirectional cross-sync
#      in average_tensor (compute.wait(reduction) each bucket): our copies
#      wait compute, which waited the previous reduction of the same buffer;
#   4. sources are kept alive by the pending list until flushed and
#      record_stream(copy_stream)'d so the caching allocator cannot recycle
#      them under the async copy.
# Falls back to stock behavior for non-(contiguous+overlap_comm) configs and
# for extra-large params (which never enter the bucket).
#
# Enabled via STARVLA_GRAD_COPY_STREAM=1 (imported by profile_entry).
import torch
import deepspeed.runtime.zero.stage_1_and_2 as _s12
from deepspeed.accelerator import get_accelerator

_orig_reduce_independent = _s12.DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads
_orig_reduce_ipg = _s12.DeepSpeedZeroOptimizer.reduce_ipg_grads


def _copy_state(self):
    st = getattr(self, "_starvla_grad_copy", None)
    if st is None:
        st = {"stream": get_accelerator().Stream(), "src": [], "dst": [], "foreach": True}
        self._starvla_grad_copy = st
    return st


def _flush_pending_copies(self):
    st = getattr(self, "_starvla_grad_copy", None)
    if not st or not st["src"]:
        return
    cur = get_accelerator().current_stream()
    cs = st["stream"]
    cs.wait_stream(cur)  # (1) sources were produced on the compute stream
    with get_accelerator().stream(cs):
        if st["foreach"]:
            try:
                torch._foreach_copy_(st["dst"], st["src"])
            except (RuntimeError, TypeError):
                st["foreach"] = False
        if not st["foreach"]:
            for d, s in zip(st["dst"], st["src"]):
                d.copy_(s, non_blocking=True)
    for t in st["src"]:
        t.record_stream(cs)  # (4) keep source storage until the copy ran
    st["src"].clear()
    st["dst"].clear()


def _patched_reduce_independent(self, param, i):
    if not (self.contiguous_gradients and self.overlap_comm):
        return _orig_reduce_independent(self, param, i)

    grad_reduc = self.get_gradient_for_reduction(param)
    if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
        self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
        self.reduce_ipg_grads()  # patched: flushes pending copies first
        # Swap ipg_index between 0 and 1 (stock behavior, contiguous+overlap)
        self.ipg_index = 1 - self.ipg_index
        self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

    param_id = self.get_param_id(param)
    assert self.params_already_reduced[param_id] == False, \
        f"The parameter {param_id} has already been reduced. \
        Gradient computed twice for this partition. \
        Multiple gradient reduction is currently not supported"

    if param.numel() > self.reduce_bucket_size:
        self.extra_large_param_to_reduce = param
    else:
        st = _copy_state(self)
        dst = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
        st["src"].append(grad_reduc.view(-1))
        st["dst"].append(dst)
        # Repoint like stock so downstream sees the bucket view; the actual
        # data movement happens at flush time on the copy stream.
        grad_reduc.data = dst.data.view_as(grad_reduc)

    self.elements_in_ipg_bucket += param.numel()

    assert grad_reduc is not None, f"rank {_s12.dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

    self.grads_in_ipg_bucket.append(grad_reduc)
    self.params_in_ipg_bucket.append((i, param.param_idx_in_group, param_id))

    if _s12.is_moe_param(param):
        self.ipg_bucket_has_moe_params = True

    self.report_ipg_memory_usage("End ipg_remove_grads", 0)


def _patched_reduce_ipg(self):
    _flush_pending_copies(self)
    st = getattr(self, "_starvla_grad_copy", None)
    if st is not None:
        # (2) the reduction path syncs against the compute stream; chain our
        # copy stream in front of it.
        get_accelerator().current_stream().wait_stream(st["stream"])
    return _orig_reduce_ipg(self)


_s12.DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads = _patched_reduce_independent
_s12.DeepSpeedZeroOptimizer.reduce_ipg_grads = _patched_reduce_ipg
print("[ds_grad_copy_stream_patch] applied", flush=True)
