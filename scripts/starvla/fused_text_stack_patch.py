# Fused text stack + fused vision tower.
# Fused text-stack compile: put the forward-phase glue between the 32
# per-layer CUDA graphs INTO the compiled region.
#
# REQUIRES the torch 2.7.1 container stack (torch 2.7.1+cu126,
# flash_attn 2.8.0.post2, causal-conv1d 1.5.2) AND, for STARVLA_FLA_TRACE=1,
# the container-side fla edit: comment out @torch.compiler.disable at
# fla/ops/gated_delta_rule/chunk.py:220 (a .bak is kept next to it; both
# handled by scripts/starvla/upgrade_container_torch27.sh). Neither is part
# of the customer-pinned environment.
#
# Background (nsys gap analysis of the per-layer-compiled config): with
# per-layer compiled forwards the GPU still idles
# ~137 ms/step in <500us gaps, almost all of it host-side glue between the 32
# layer graphs — transformers 5.3's output_hidden_states capture hooks
# (utils/output_capturing.py fires a ContextVar-guarded hook at every layer
# __call__), per-layer module dispatch, and per-graph cudagraph input copies
# (~1955 cudaMemcpyAsync/step).
#
# This patch replaces Qwen3_5TextModel.forward for the no-cache training path:
#   - preamble (embed_tokens, masks, mrope) stays eager, identical to HF;
#   - the 32-layer loop + hidden-state collection + final norm run inside ONE
#     torch.compile(dynamic=False, mode=reduce-overhead) function, calling the
#     UNBOUND Qwen3_5DecoderLayer.forward so neither the capture hooks nor any
#     per-layer compiled bound method is involved;
#   - hidden_states output replicates capture_outputs' tie_last semantics
#     exactly: (embeds, layer1..layer31, norm(layer32)) — what QwenPI slices
#     with [-num_layers:].
# Any cached/generation call (past_key_values / use_cache) falls back to the
# original decorated forward, so generate() behavior is unchanged.
#
# Enabled via STARVLA_FUSED_TEXT_STACK=1 (imported by profile_entry).
# Compile mode override: STARVLA_FUSED_COMPILE_MODE (default reduce-overhead).
import os
import sys

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as _m

_orig_forward = _m.Qwen3_5TextModel.forward


def _constant_fold_fla_device_checks():
    """Make fla's host-side device-capability checks dynamo-constant.

    With fla's @torch.compiler.disable removed from chunk_gated_delta_rule
    (STARVLA_FLA_TRACE=1 expects the container's fla chunk.py edited
    accordingly), the only remaining graph break is fla.utils.check_shared_mem
    -> torch.cuda.device_count() (returns a non-Tensor int). The result is a
    runtime constant per (arch, device), so precompute it and swap in a pure
    dict lookup that dynamo constant-folds. Verified fullgraph fwd+bwd OK on
    torch 2.7.1 (2026-07-05).
    """
    import fla.utils as _fu

    orig = _fu.check_shared_mem
    const = {}
    for arch in ("hopper", "ampere", "ada", "volta", "none"):
        for idx in range(torch.cuda.device_count()):
            try:
                const[(arch, idx)] = orig(arch, idx)
            except Exception:
                const[(arch, idx)] = False

    def _const_check_shared_mem(arch="none", tensor_idx=0):
        return const[(arch, tensor_idx)]

    # Same story for get_multiprocessor_count (FusedRMSNormGated queries it per
    # forward; lru_cache internals + triton get_device_properties are pybind
    # calls dynamo can't trace).
    orig_mp = _fu.get_multiprocessor_count
    mp_const = {idx: orig_mp(idx) for idx in range(torch.cuda.device_count())}

    def _const_get_multiprocessor_count(tensor_idx=0):
        return mp_const[tensor_idx]

    _fu.check_shared_mem = _const_check_shared_mem
    _fu.get_multiprocessor_count = _const_get_multiprocessor_count
    # Several fla modules bind these via `from fla.utils import ...`.
    for name, mod in list(sys.modules.items()):
        if name.startswith("fla"):
            if hasattr(mod, "check_shared_mem"):
                mod.check_shared_mem = _const_check_shared_mem
            if hasattr(mod, "get_multiprocessor_count"):
                mod.get_multiprocessor_count = _const_get_multiprocessor_count


if os.environ.get("STARVLA_FLA_TRACE"):
    _constant_fold_fla_device_checks()
    print("[fused_text_stack_patch] fla device checks constant-folded (STARVLA_FLA_TRACE)", flush=True)


# ---------------------------------------------------------------------------
# Fast causal-conv as a FUNCTIONAL custom op. The DaoAILab package (1.5.2)
# ships opaque custom ops, but with out=-mutation signatures: under
# torch.compile their auto-functionalization re-records/recompiles every step
# (measured 146-148 s/step at bs32 with reduce-overhead group runners). A pure
# functional wrapper around the same CUDA kernels stays a single opaque node:
# normal compile time, fast kernels inside the graph.
try:
    from causal_conv1d.cpp_functions import (
        causal_conv1d_fwd_function as _cc_fwd_fn,
        causal_conv1d_bwd_function as _cc_bwd_fn,
    )
except Exception:
    _cc_fwd_fn = None

if _cc_fwd_fn is not None:
    from typing import Optional as _Opt

    @torch.library.custom_op("starvla::causal_conv1d_silu", mutates_args=())
    def _starvla_causal_conv1d_silu(
        x: torch.Tensor, weight: torch.Tensor, bias: _Opt[torch.Tensor]
    ) -> torch.Tensor:
        # mirror CausalConv1dFn.forward's layout handling
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        b = bias.contiguous() if bias is not None else None
        return _cc_fwd_fn(x, weight, b, None, None, None, True)

    @_starvla_causal_conv1d_silu.register_fake
    def _starvla_causal_conv1d_silu_fake(x, weight, bias):
        return torch.empty_like(x)

    def _starvla_cc_setup(ctx, inputs, output):
        x, weight, bias = inputs
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0, device=x.device))

    def _starvla_cc_backward(ctx, dout):
        x, weight, bias = ctx.saved_tensors
        if not ctx.has_bias:
            bias = None
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        dx, dweight, dbias, _ = _cc_bwd_fn(x, weight, bias, dout, None, None, None, None, False, True)
        return dx, dweight, dbias if ctx.has_bias else None

    torch.library.register_autograd(
        "starvla::causal_conv1d_silu", _starvla_cc_backward, setup_context=_starvla_cc_setup
    )

    def _starvla_fast_conv_shim(x, weight, bias=None, activation=None, seq_idx=None, **kwargs):
        # transformers' GatedDeltaNet training path uses activation="silu",
        # seq_idx=None; anything else takes the stock package path (eager).
        if seq_idx is not None or activation not in ("silu", "swish"):
            from causal_conv1d import causal_conv1d_fn as _pkg_fn
            return _pkg_fn(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx, **kwargs)
        return torch.ops.starvla.causal_conv1d_silu(x, weight, bias)
else:
    _starvla_fast_conv_shim = None


def _group_runner(layers_group, mode):
    layer_forward = _m.Qwen3_5DecoderLayer.forward  # unbound: no hooks, no per-layer compile wrapper

    def _run_group(hidden_states, cos, sin, causal_mask, linear_attn_mask, position_ids, cache_position):
        collected = []
        for layer in layers_group:
            mask = linear_attn_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer_forward(
                layer,
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
            )
            collected.append(hidden_states)
        return collected

    return torch.compile(_run_group, dynamic=False, mode=mode)


def _make_fused_runner(self):
    # NOTE: mode="reduce-overhead" over the WHOLE stack is NOT usable on
    # torch 2.6 NOR 2.7.1 (both verified 2026-07-05, identical crash; fla-core
    # 0.5.1 still ships the compiler.disable, so no fla upgrade removes the
    # breaks; old-style cudagraphs via cudagraph_trees=False fails the
    # static-input data_ptr assert immediately): the fla
    # @torch.compiler.disable graph breaks split one
    # compiled function into many cudagraph partitions sharing one memory
    # pool, and cudagraph_trees' allocator checkpointing collides with the
    # ZeRO-2 grad-hook NCCL allocations during backward
    # ("_cuda_setCheckpointPoolState: curr_block->next == nullptr", observed
    # 2026-07-05). Per-layer compiles dodge this because each layer
    # is its own top-level compile. Two supported shapes:
    #   STARVLA_FUSED_GROUP_SIZE=0 (default): whole stack in one compile,
    #     mode default (fusion, NO cudagraphs);
    #   STARVLA_FUSED_GROUP_SIZE=N: ceil(32/N) independent top-level compiles
    #     (safe with reduce-overhead, like per-layer, but N x less glue).
    layers = tuple(self.layers[: self.config.num_hidden_layers])
    if os.environ.get("STARVLA_CONV_TORCH_FALLBACK") or _starvla_fast_conv_shim is None:
        # Rollback path (formerly the default, keyed off STARVLA_FLA_TRACE):
        # force the HF torch-native conv (F.silu(self.conv1d(x))) inside the
        # graph — numerically the same op, but inductor lowers it to
        # conv_depthwise2d at ~8.5x the kernel time (measured 29 ms/step at
        # bs24 vs ~4 ms for the CUDA kernels, fwd+bwd). Kept as an opt-out
        # because it is the only conv path with zero custom-op dependencies.
        for layer in layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.linear_attn.causal_conv1d_fn = None
        print("[fused_text_stack_patch] conv: torch-native fallback (STARVLA_CONV_TORCH_FALLBACK)", flush=True)
    else:
        # Default: route the DeltaNet short conv through our functional
        # custom op (see registration above) — fast CUDA kernels inside the
        # compiled graph, single opaque node, no per-step recompiles. The
        # stock package fn must NOT be left in place under compile: its
        # out=-mutation ops trigger per-step re-functionalization.
        for layer in layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.linear_attn.causal_conv1d_fn = _starvla_fast_conv_shim
        print("[fused_text_stack_patch] conv: starvla::causal_conv1d_silu custom op", flush=True)
    group_size = int(os.environ.get("STARVLA_FUSED_GROUP_SIZE", "0") or 0)
    mode = os.environ.get(
        "STARVLA_FUSED_COMPILE_MODE", "reduce-overhead" if group_size else "default"
    )
    if mode == "reduce-overhead-notrees":
        # Old-style per-graph cudagraphs: no allocator-checkpoint machinery, so
        # the fla graph breaks don't hit the torch<=2.7 cudagraph_trees crash.
        # Global inductor switch — also affects the DiT compile (single graph,
        # safe). Validate loss when using this.
        torch._inductor.config.triton.cudagraph_trees = False
        mode = "reduce-overhead"
    if mode in ("", "default", "none"):
        mode = None
    if group_size <= 0:
        group_size = len(layers)
    groups = [layers[i : i + group_size] for i in range(0, len(layers), group_size)]
    runners = [_group_runner(g, mode) for g in groups]
    norm = self.norm

    def _run_layers(hidden_states, cos, sin, causal_mask, linear_attn_mask, position_ids, cache_position):
        collected = [hidden_states]
        for run_group in runners:
            states = run_group(
                hidden_states, cos, sin, causal_mask, linear_attn_mask, position_ids, cache_position
            )
            collected.extend(states)
            hidden_states = states[-1]
        return norm(hidden_states), collected

    return _run_layers


def _fused_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cache_position=None,
    **kwargs,
):
    # Cached / generation / checkpointed paths keep the stock implementation.
    if past_key_values is not None or use_cache or (self.gradient_checkpointing and self.training):
        return _orig_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    output_hidden_states = kwargs.get("output_hidden_states", False)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    causal_mask = _m.create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    # HF's _update_linear_attn_mask runs torch.all(attention_mask==1)
    # in an `if` — a per-step GPU->CPU sync just to swap the mask for None as
    # an all-visible optimization. With left padding to collate_pad_to that
    # case never triggers, and passing the mask when it IS all-ones is still
    # correct (just not the fast path), so skip the sync entirely.
    linear_attn_mask = attention_mask

    cos, sin = self.rotary_emb(inputs_embeds, position_ids)

    runner = getattr(self, "_starvla_fused_runner", None)
    if runner is None:
        runner = _make_fused_runner(self)
        self._starvla_fused_runner = runner

    normed, collected = runner(
        inputs_embeds, cos, sin, causal_mask, linear_attn_mask, position_ids, cache_position
    )

    out = _m.Qwen3_5ModelOutputWithPast(last_hidden_state=normed, past_key_values=None)
    if output_hidden_states:
        # tie_last_hidden_states semantics: last element is the POST-norm state.
        out["hidden_states"] = tuple(collected[:-1]) + (normed,)
    return out


_m.Qwen3_5TextModel.forward = _fused_forward
print("[fused_text_stack_patch] applied", flush=True)


# ---------------------------------------------------------------------------
# Fused VISION tower (STARVLA_FUSED_VISION=1): compile the 24 ViT
# blocks (FA2 varlen) into one CUDA-graphed region. Historically blocked by
# two dynamo breaks, both host-side constants (verified fullgraph fwd+bwd OK
# on torch 2.7.1 + flash_attn 2.8.0.post2, 2026-07-06):
#   1. max_seqlen = (cu_seqlens[1:]-cu_seqlens[:-1]).max() is a GPU 0-dim
#      tensor but flash-attn's custom op wants a SymInt. With fixed camera
#      resolutions the value is a per-grid constant -> computed on host once
#      per distinct image_grid_thw and stamped on the attention modules as a
#      python int.
#   2. transformers' lazy_import_flash_attention does importlib at call time
#      -> prewarmed and constant-folded to a dict lookup.
# ---------------------------------------------------------------------------


def _patch_vision_tower():
    import torch.nn.functional as F
    import transformers.modeling_flash_attention_utils as _fau

    # (2) constant-fold the lazy flash import
    _orig_lazy = _fau.lazy_import_flash_attention
    _lazy_cache = {}

    def _folded_lazy(implementation, attention_wrapper=None, allow_all_kernels=False):
        key = (implementation, attention_wrapper, allow_all_kernels)
        if key not in _lazy_cache:
            _lazy_cache[key] = _orig_lazy(implementation, attention_wrapper, allow_all_kernels)
        return _lazy_cache[key]

    _folded_lazy("flash_attention_2")  # prewarm so compiled regions fold the lookup
    _fau.lazy_import_flash_attention = _folded_lazy

    # (1) attention forward that takes max_seqlen as a stamped python int
    _orig_vattn_forward = _m.Qwen3_5VisionAttention.forward

    def _vattn_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        ms = getattr(self, "_starvla_max_seqlen", None)
        if ms is None:
            return _orig_vattn_forward(
                self, hidden_states, cu_seqlens,
                rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs,
            )
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query, key = _m.apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        iface = _m.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, _m.eager_attention_forward)
        attn_output, _ = iface(
            self, query, key, value,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=ms,
            max_length_k=ms,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    _m.Qwen3_5VisionAttention.forward = _vattn_forward

    _orig_vm_forward = _m.Qwen3_5VisionModel.forward

    # ViT input-structure cache (STARVLA_VIT_INPUT_CACHE=1). Everything the
    # HF vision preamble derives from grid_thw alone is grid-constant across
    # steps in this workload (static resize -> identical grid_thw bytes every
    # batch): the rotary cos/sin tables, cu_seqlens, max_seqlen, and — the
    # expensive one — the pos-embed interpolation STRUCTURE. HF's
    # fast_pos_embed_interpolate rebuilds python lists per image per step
    # (grid.tolist() D2H sync + 4x len-16k list extends + torch.tensor H2D);
    # at bs32 the whole preamble is ~90 ms of exposed CPU (716 memcpys, 587
    # stream syncs). Only the gather from the TRAINABLE self.pos_embed.weight
    # must run every step, so we cache (idx, weight, perm) and reduce the
    # per-step work to gather * weight -> sum -> index_select, which keeps
    # the autograd path to pos_embed.weight intact. The first hit per grid
    # key verifies the cached-path output against HF's original computation.
    _vit_input_cache_on = bool(os.environ.get("STARVLA_VIT_INPUT_CACHE"))

    def _build_vit_input_entry(self, grid_thw, gcpu):
        device = self.pos_embed.weight.device
        wdtype = self.pos_embed.weight.dtype
        merge_size = self.config.spatial_merge_size
        grid_list = gcpu.tolist()

        # --- pos-embed interpolation structure (mirrors HF fast_pos_embed_interpolate)
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - h_floor, w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=wdtype, device=device)

        # --- row permutation replicating HF's split -> repeat(t) -> view/permute/flatten
        tokens_per_img = [h * w for _, h, w in grid_list]
        parts = torch.arange(sum(tokens_per_img)).split(tokens_per_img)
        perm_parts = []
        for part, (t, h, w) in zip(parts, grid_list):
            p = part.repeat(t)
            p = (
                p.view(t, h // merge_size, merge_size, w // merge_size, merge_size)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1)
            )
            perm_parts.append(p)
        perm = torch.cat(perm_parts).to(device)

        # --- grid-constant rotary tables and cu_seqlens (HF expressions verbatim)
        with torch.no_grad():
            rotary = self.rot_pos_emb(grid_thw)
            seq_len = rotary.shape[0]
            rotary = rotary.reshape(seq_len, -1)
            emb = torch.cat((rotary, rotary), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        ms = int((gcpu[:, 1] * gcpu[:, 2]).repeat_interleave(gcpu[:, 0]).max())

        entry = {
            "idx": idx_tensor, "w": weight_tensor, "perm": perm,
            "cos": cos, "sin": sin, "cu": cu_seqlens, "ms": ms,
        }
        # one-time equivalence check of the cached pos-embed path vs HF
        # (4-term add in HF's order -> bitwise-identical, not just allclose)
        with torch.no_grad():
            ref = self.fast_pos_embed_interpolate(grid_thw).float()
            pe = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
            got = (pe[0] + pe[1] + pe[2] + pe[3]).index_select(0, perm).float()
            err = (ref - got).abs().max().item()
            if err > 2e-3:
                raise RuntimeError(f"[vit_input_cache] pos-embed mismatch: max|diff|={err}")
            print(f"[fused_text_stack_patch] vit input cache entry built (seq {seq_len}, max|diff|={err:.2e})", flush=True)
        return entry

    def _vm_forward(self, hidden_states, grid_thw, **kwargs):
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            return _orig_vm_forward(self, hidden_states, grid_thw, **kwargs)
        hidden_states = self.patch_embed(hidden_states)

        gcpu = grid_thw.cpu()
        gkey = gcpu.numpy().tobytes()
        if _vit_input_cache_on:
            cache = getattr(self, "_starvla_vit_in_cache", None)
            if cache is None:
                cache = self._starvla_vit_in_cache = {}
            entry = cache.get(gkey)
            if entry is None:
                entry = cache[gkey] = _build_vit_input_entry(self, grid_thw, gcpu)
            # per-step: only the gather from the trainable pos_embed.weight
            # (4-term add matches HF's summation order bit-for-bit)
            pe = self.pos_embed(entry["idx"]) * entry["w"][:, :, None]
            pos_embeds = (pe[0] + pe[1] + pe[2] + pe[3]).index_select(0, entry["perm"])
            hidden_states = hidden_states + pos_embeds
            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            cos, sin, cu_seqlens, ms = entry["cos"], entry["sin"], entry["cu"], entry["ms"]
        else:
            # original per-step preamble (pos embeds + rotary + cu_seqlens)
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            # per-grid constant max_seqlen (host int, cached; the tiny D2H here
            # replaces the .item() the eager flash path already paid every step)
            if not hasattr(self, "_starvla_ms_cache"):
                self._starvla_ms_cache = {}
            ms = self._starvla_ms_cache.get(gkey)
            if ms is None:
                ms = int((gcpu[:, 1] * gcpu[:, 2]).repeat_interleave(gcpu[:, 0]).max())
                self._starvla_ms_cache[gkey] = ms
        for blk in self.blocks:
            blk.attn._starvla_max_seqlen = ms

        runner = getattr(self, "_starvla_vis_runner", None)
        if runner is None:
            blocks = tuple(self.blocks)
            block_forward = _m.Qwen3_5VisionBlock.forward  # unbound: no hooks

            def _run_blocks(hidden_states, cu_seqlens, cos, sin):
                for blk in blocks:
                    hidden_states = block_forward(
                        blk, hidden_states, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin)
                    )
                return hidden_states

            runner = torch.compile(_run_blocks, dynamic=False, mode="reduce-overhead")
            self._starvla_vis_runner = runner

        hidden_states = runner(hidden_states, cu_seqlens, cos, sin)
        merged_hidden_states = self.merger(hidden_states)
        return _m.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    _m.Qwen3_5VisionModel.forward = _vm_forward

    # split_sizes on the vision->text handoff are another grid-only constant:
    # HF recomputes them every step as a GPU prod kernel + a .tolist() D2H
    # sync (Qwen3_5Model.get_image_features), right where the merged image
    # embeds hand over to the text stack. Cache the host list per grid key;
    # first hit verifies against HF's own expression.
    if _vit_input_cache_on:
        _orig_gif = _m.Qwen3_5Model.get_image_features

        def _gif_cached(self, pixel_values, image_grid_thw=None, **kwargs):
            if image_grid_thw is None:
                return _orig_gif(self, pixel_values, image_grid_thw=image_grid_thw, **kwargs)
            kwargs.pop("return_dict", None)  # @can_return_tuple pops this on the original
            cache = getattr(self, "_starvla_split_cache", None)
            if cache is None:
                cache = self._starvla_split_cache = {}
            gcpu = image_grid_thw.cpu()
            gkey = gcpu.numpy().tobytes()
            split_sizes = cache.get(gkey)
            if split_sizes is None:
                m2 = self.visual.spatial_merge_size**2
                split_sizes = (gcpu.prod(-1) // m2).tolist()
                ref = (image_grid_thw.prod(-1) // m2).tolist()  # HF's expression, one-time
                if split_sizes != ref:
                    raise RuntimeError("[vit_input_cache] split_sizes mismatch vs HF expression")
                cache[gkey] = split_sizes
                print(f"[fused_text_stack_patch] split_sizes cache entry built ({len(split_sizes)} images)", flush=True)
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
            vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
            return vision_output

        _m.Qwen3_5Model.get_image_features = _gif_cached


if os.environ.get("STARVLA_FUSED_VISION"):
    _patch_vision_tower()
    print("[fused_text_stack_patch] vision tower fused (STARVLA_FUSED_VISION)", flush=True)


# ---------------------------------------------------------------------------
# Sync-free multimodal embedding merge (STARVLA_FAST_MM_MERGE=1).
#
# HF's Qwen3_5Model.get_placeholder_mask validates that the number of image
# placeholder tokens matches the vision features via
# `inputs_embeds[special_image_mask].numel() == image_features.numel()` — a
# boolean-mask gather (cub DeviceSelect) followed by a host bool() readback,
# i.e. one more CPU<->GPU round trip per modality per step, sitting exactly
# in the exposed gap between the vision graph and the first text-stack graph
# (milestone4 step-40: ~150 tiny D2H syncs in 16 ms there, shared with the
# in-model MRoPE compute that the MRoPE position-id cache removes).
#
# The fast path keeps the mask computation (pure GPU, async) and DROPS the
# validation. Trade-off: a genuine token/feature count mismatch would surface
# as a shifted masked_scatter instead of a clean error — enable only with a
# processor/collate combination that has been validated once (e.g. with the
# stock path or STARVLA_CHECK_POSIDS runs).
# ---------------------------------------------------------------------------


def _patch_mm_merge():
    def _fast_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            # embeds-only path is generation-oriented; keep stock behavior
            return _orig_placeholder_mask(self, input_ids, inputs_embeds,
                                          image_features=image_features, video_features=video_features)
        special_image_mask = (input_ids == self.config.image_token_id)
        special_video_mask = (input_ids == self.config.video_token_id)
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return special_image_mask, special_video_mask

    _orig_placeholder_mask = _m.Qwen3_5Model.get_placeholder_mask
    _m.Qwen3_5Model.get_placeholder_mask = _fast_placeholder_mask


if os.environ.get("STARVLA_FAST_MM_MERGE"):
    _patch_mm_merge()
    print("[fused_text_stack_patch] sync-free mm merge (STARVLA_FAST_MM_MERGE)", flush=True)


# ---------------------------------------------------------------------------
# Index-based multimodal merge (STARVLA_INDEX_MM_MERGE=1).
#
# masked_scatter's BACKWARD runs grad.masked_select(mask) -> torch.nonzero,
# which must count the mask's nonzeros and read that count back to the CPU:
# one 4-byte DtoH + cudaStreamSynchronize per step, landing at the deepest
# point of the backward queue. Measured on milestone1: the autograd thread
# stalls ~63 ms behind the stream-7 backlog, freezing the ZeRO-2 bucket
# hooks so allreduce buckets 4-6 pile up at the step tail (the direct cause
# of the exposed-allreduce pattern). The sync-free forward-side merge
# (STARVLA_FAST_MM_MERGE) only removed the forward validation readback - the
# nonzero was deferred, not removed.
#
# Fix: image-token positions are a pure function of input_ids (known before
# the step), so compute flat indices on the CPU from an input-only readback
# (cached by input_ids bytes, like the MRoPE position-id cache) and merge
# with index_copy - its backward is index_select: no nonzero, no sync.
# Bonus: len(index) == image_embeds.shape[0] is a FREE host-side integer
# check, restoring the count validation that FAST_MM_MERGE dropped.
# ---------------------------------------------------------------------------


def _patch_index_mm_merge():
    _orig_model_forward = _m.Qwen3_5Model.forward

    def _image_token_indices(self, input_ids):
        cache = getattr(self, "_starvla_img_idx_cache", None)
        if cache is None:
            cache = self._starvla_img_idx_cache = {}
        cpu_ids = input_ids.cpu()  # input-only readback, H2D already complete
        key = cpu_ids.numpy().tobytes()
        idx = cache.get(key)
        if idx is None:
            flat = (cpu_ids.view(-1) == self.config.image_token_id).nonzero(as_tuple=False).squeeze(1)
            idx = flat.to(input_ids.device)
            if len(cache) < 64:
                cache[key] = idx
        return idx

    def _mm_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                    video_grid_thw=None, mm_token_type_ids=None, cache_position=None, **kwargs):
        # Only the image-only training shape takes the fast path; anything
        # else (embeds-only, video, generation) keeps stock behavior.
        if input_ids is None or inputs_embeds is not None or pixel_values is None or pixel_values_videos is not None:
            return _orig_model_forward(
                self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position, **kwargs)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        idx = self._starvla_image_token_indices(input_ids)
        if idx.numel() != image_embeds.shape[0]:  # host ints: free, sync-less validation
            raise RuntimeError(
                f"image token count ({idx.numel()}) != vision features ({image_embeds.shape[0]}); "
                "processor/collate mismatch")
        b, t, h = inputs_embeds.shape
        inputs_embeds = inputs_embeds.view(-1, h).index_copy(0, idx, image_embeds).view(b, t, h)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values, mm_token_type_ids=mm_token_type_ids)

        outputs = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            cache_position=cache_position, **kwargs)
        return _m.Qwen3_5ModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)

    _m.Qwen3_5Model._starvla_image_token_indices = _image_token_indices
    _m.Qwen3_5Model.forward = _mm_forward


if os.environ.get("STARVLA_INDEX_MM_MERGE"):
    _patch_index_mm_merge()
    print("[fused_text_stack_patch] index-based mm merge (STARVLA_INDEX_MM_MERGE)", flush=True)
