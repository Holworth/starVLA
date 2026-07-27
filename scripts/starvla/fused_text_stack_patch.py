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

from starVLA.model.qwenpi_metadata import (
    VisionEntry,
    device_key,
    env_bool,
    get_active_qwen_metadata,
)

_orig_forward = _m.Qwen3_5TextModel.forward
_check_metadata_cache = env_bool("STARVLA_CHECK_METADATA_CACHE")


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


if env_bool("STARVLA_FLA_TRACE"):
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
    if env_bool("STARVLA_CONV_TORCH_FALLBACK") or _starvla_fast_conv_shim is None:
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

    standard_full_prefill = cache_position is None
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

    active = get_active_qwen_metadata()
    metadata_causal_compatible = (
        active is not None
        and standard_full_prefill
        and self.config._attn_implementation == "sdpa"
        and getattr(self.config, "is_causal", True)
    )
    if not metadata_causal_compatible:
        causal_mask = _m.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
    else:
        layout = active.layout
        expected_shape = (layout.batch_size, layout.seq_len)
        if attention_mask is None or tuple(attention_mask.shape) != expected_shape:
            raise RuntimeError(
                "active Qwen metadata does not match the text attention mask: "
                f"descriptor={expected_shape}, tensor="
                f"{None if attention_mask is None else tuple(attention_mask.shape)}"
            )
        if tuple(inputs_embeds.shape[:2]) != expected_shape:
            raise RuntimeError(
                "active Qwen metadata does not match the text embeddings: "
                f"descriptor={expected_shape}, tensor={tuple(inputs_embeds.shape[:2])}"
            )

        # HF's SDPA mask for this no-cache, full-attention training shape is
        # exactly `(key <= query) & attention_mask[key]`.  Building the mask
        # from the worker-produced immutable bytes avoids the
        # `padding_mask.all()` GPU->CPU synchronization in
        # `_ignore_causal_mask_sdpa`.  Values are cached per sample because
        # padding patterns repeat even when whole batches do not.
        all_visible = all(
            sample.attention_mask_u8 == b"\x01" * layout.seq_len
            for sample in layout.samples
        )
        if all_visible:
            # Matches create_causal_mask's is_causal fast path when the whole
            # batch is unpadded.
            causal_mask = None
        else:
            dev_key = device_key(inputs_embeds)

            def _sample_mask(sample):
                key = ("causal", sample.seq_len, sample.attention_mask_u8, dev_key)

                def _build():
                    valid_keys = torch.tensor(
                        tuple(sample.attention_mask_u8),
                        dtype=torch.bool,
                        device=inputs_embeds.device,
                    )
                    positions = torch.arange(sample.seq_len, device=inputs_embeds.device)
                    lower_triangle = positions[:, None] >= positions[None, :]
                    return (lower_triangle & valid_keys[None, :]).unsqueeze(0).unsqueeze(0)

                return active.cache.causal_masks.get_or_create(key, _build)

            causal_mask = torch.cat([_sample_mask(sample) for sample in layout.samples], dim=0)
        if _check_metadata_cache and not getattr(
            self, "_starvla_metadata_causal_checked", False
        ):
            reference_mask = _m.create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            masks_match = (
                reference_mask is None
                if causal_mask is None
                else reference_mask is not None
                and torch.equal(reference_mask, causal_mask)
            )
            if not masks_match:
                raise RuntimeError(
                    "[metadata_cache] cached causal mask does not match HF"
                )
            self._starvla_metadata_causal_checked = True
            print(
                "[fused_text_stack_patch] metadata check: causal mask OK",
                flush=True,
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
        if ms is None or not _m.is_flash_attention_requested(self.config):
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
    # the autograd path to pos_embed.weight intact. The optional metadata
    # correctness check verifies the cached path against HF once.
    _vit_input_cache_on = env_bool("STARVLA_VIT_INPUT_CACHE")

    def _rotary_from_grids(self, grids):
        """Mirror Qwen3_5VisionModel.rot_pos_emb without a GPU tolist()."""

        merge_size = self.spatial_merge_size
        max_hw = max(max(height, width) for _, height, width in grids)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = sum(t * h * w for t, h, w in grids)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grids:
            merged_height = height // merge_size
            merged_width = width // merge_size
            block_rows = torch.arange(merged_height, device=device)
            block_cols = torch.arange(merged_width, device=device)
            intra_rows = torch.arange(merge_size, device=device)
            intra_cols = torch.arange(merge_size, device=device)
            row_ids = (
                block_rows[:, None, None, None] * merge_size
                + intra_rows[None, None, :, None]
            )
            col_ids = (
                block_cols[None, :, None, None] * merge_size
                + intra_cols[None, None, None, :]
            )
            row_ids = row_ids.expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            col_ids = col_ids.expand(
                merged_height, merged_width, merge_size, merge_size
            ).reshape(-1)
            coordinates = torch.stack((row_ids, col_ids), dim=-1)
            if num_frames > 1:
                coordinates = coordinates.repeat(num_frames, 1)
            num_tokens = coordinates.shape[0]
            pos_ids[offset : offset + num_tokens] = coordinates
            offset += num_tokens
        return freq_table[pos_ids].flatten(1)

    def _build_vit_input_entry(self, grid_thw, vision):
        device = self.pos_embed.weight.device
        weight_dtype = self.pos_embed.weight.dtype
        merge_size = vision.spatial_merge_size
        grids = vision.grids

        # Mirrors fast_pos_embed_interpolate.  The list conversions below are
        # CPU-only temporaries built from immutable Python grid tuples.
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, height, width in grids:
            height_ids = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_ids = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor, width_floor = height_ids.int(), width_ids.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = height_ids - height_floor, width_ids - width_floor
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side
            indices = [
                (base_height[None].T + width_floor[None]).flatten(),
                (base_height[None].T + width_ceil[None]).flatten(),
                (base_height_ceil[None].T + width_floor[None]).flatten(),
                (base_height_ceil[None].T + width_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=weight_dtype, device=device)

        tokens_per_image = [height * width for _, height, width in grids]
        parts = torch.arange(sum(tokens_per_image)).split(tokens_per_image)
        permutation_parts = []
        for part, (num_frames, height, width) in zip(parts, grids):
            permutation_parts.append(
                part.repeat(num_frames)
                .view(
                    num_frames,
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(-1)
            )
        permutation = torch.cat(permutation_parts).to(device)

        with torch.no_grad():
            rotary = _rotary_from_grids(self, grids)
            seq_len = rotary.shape[0]
            rotary = rotary.reshape(seq_len, -1)
            doubled = torch.cat((rotary, rotary), dim=-1)
            cos, sin = doubled.cos(), doubled.sin()

        cumulative = [0]
        for num_frames, height, width in grids:
            for _ in range(num_frames):
                cumulative.append(cumulative[-1] + height * width)
        cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=device)
        entry = VisionEntry(
            idx=idx_tensor,
            weights=weight_tensor,
            permutation=permutation,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=vision.max_seqlen,
        )

        if _check_metadata_cache and not getattr(
            self, "_starvla_metadata_vision_checked", False
        ):
            # This is intentionally the only vision path that reads GPU
            # metadata back.  It is opt-in and runs once for validation.
            actual_grids = tuple(
                tuple(int(value) for value in row)
                for row in grid_thw.cpu().tolist()
            )
            if actual_grids != grids:
                raise RuntimeError(
                    "[metadata_cache] vision grids mismatch: "
                    f"descriptor={grids}, tensor={actual_grids}"
                )
            with torch.no_grad():
                reference_pos = self.fast_pos_embed_interpolate(grid_thw).float()
                pos = self.pos_embed(entry.idx) * entry.weights[:, :, None]
                cached_pos = (
                    pos[0] + pos[1] + pos[2] + pos[3]
                ).index_select(0, entry.permutation).float()
                pos_error = (reference_pos - cached_pos).abs().max().item()
                reference_rotary = self.rot_pos_emb(grid_thw).reshape(seq_len, -1)
                rotary_error = (reference_rotary - rotary).abs().max().item()
                reference_cu = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                reference_cu = F.pad(reference_cu, (1, 0), value=0)
                cu_matches = torch.equal(reference_cu, entry.cu_seqlens)
            if pos_error > 2e-3 or rotary_error != 0 or not cu_matches:
                raise RuntimeError(
                    "[metadata_cache] vision metadata mismatch: "
                    f"pos_error={pos_error}, rotary_error={rotary_error}, "
                    f"cu_seqlens_equal={cu_matches}"
                )
            self._starvla_metadata_vision_checked = True
            print("[fused_text_stack_patch] metadata check: vision OK", flush=True)
        return entry

    def _fallback_vm_forward(self, hidden_states, grid_thw, **kwargs):
        # Attention is patched globally; clear a value stamped by a previous
        # fast-path call before using the stock HF vision preamble.
        for block in self.blocks:
            block.attn._starvla_max_seqlen = None
        return _orig_vm_forward(self, hidden_states, grid_thw, **kwargs)

    def _vm_forward(self, hidden_states, grid_thw, **kwargs):
        active = get_active_qwen_metadata()
        if (
            active is None
            or not _vit_input_cache_on
            or not _m.is_flash_attention_requested(self.config)
            or kwargs.get("output_hidden_states")
            or kwargs.get("output_attentions")
        ):
            return _fallback_vm_forward(self, hidden_states, grid_thw, **kwargs)

        vision = active.layout.vision
        if not vision.grids:
            raise RuntimeError(
                "active Qwen metadata has no image grids for a vision forward"
            )
        if tuple(grid_thw.shape) != (len(vision.grids), 3):
            raise RuntimeError(
                "active Qwen metadata does not match image_grid_thw: "
                f"descriptor={(len(vision.grids), 3)}, tensor={tuple(grid_thw.shape)}"
            )
        if vision.spatial_merge_size != self.config.spatial_merge_size:
            raise RuntimeError(
                "active Qwen metadata spatial merge size does not match the vision model: "
                f"{vision.spatial_merge_size} != {self.config.spatial_merge_size}"
            )
        expected_patches = sum(t * h * w for t, h, w in vision.grids)
        if hidden_states.shape[0] != expected_patches:
            raise RuntimeError(
                "active Qwen metadata does not match pixel patches: "
                f"descriptor={expected_patches}, tensor={hidden_states.shape[0]}"
            )

        hidden_states = self.patch_embed(hidden_states)
        cache_key = (
            "vision",
            vision,
            device_key(self.pos_embed.weight),
            self.pos_embed.weight.dtype,
            self.num_grid_per_side,
        )
        entry = active.cache.vision_batches.get_or_create(
            cache_key, lambda: _build_vit_input_entry(self, grid_thw, vision)
        )

        # Keep the gather live so gradients reach the trainable position table.
        pos = self.pos_embed(entry.idx) * entry.weights[:, :, None]
        pos_embeds = (pos[0] + pos[1] + pos[2] + pos[3]).index_select(
            0, entry.permutation
        )
        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        for block in self.blocks:
            block.attn._starvla_max_seqlen = entry.max_seqlen

        runner = getattr(self, "_starvla_vis_runner", None)
        if runner is None:
            blocks = tuple(self.blocks)
            block_forward = _m.Qwen3_5VisionBlock.forward

            def _run_blocks(states, cu_seqlens, cos, sin):
                for block in blocks:
                    states = block_forward(
                        block,
                        states,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=(cos, sin),
                    )
                return states

            runner = torch.compile(_run_blocks, dynamic=False, mode="reduce-overhead")
            self._starvla_vis_runner = runner

        hidden_states = runner(
            hidden_states, entry.cu_seqlens, entry.cos, entry.sin
        )
        merged_hidden_states = self.merger(hidden_states)
        return _m.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    _m.Qwen3_5VisionModel.forward = _vm_forward

    # The vision->text split sizes are part of the same descriptor.  Keep the
    # fast path narrow so tuple-return and diagnostic HF calls retain the
    # exact original behavior.
    _orig_gif = _m.Qwen3_5Model.get_image_features

    def _gif_cached(self, pixel_values, image_grid_thw=None, **kwargs):
        active = get_active_qwen_metadata()
        if (
            active is None
            or not _vit_input_cache_on
            or image_grid_thw is None
            or kwargs.get("return_dict", True) is False
        ):
            return _orig_gif(
                self, pixel_values, image_grid_thw=image_grid_thw, **kwargs
            )

        split_sizes = active.layout.vision.split_sizes
        if len(split_sizes) != image_grid_thw.shape[0]:
            raise RuntimeError(
                "active Qwen metadata split sizes do not match image_grid_thw: "
                f"{len(split_sizes)} != {image_grid_thw.shape[0]}"
            )
        if _check_metadata_cache and not getattr(
            self, "_starvla_metadata_split_checked", False
        ):
            reference = tuple(
                (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2)
                .cpu()
                .tolist()
            )
            if reference != split_sizes:
                raise RuntimeError(
                    f"[metadata_cache] split sizes mismatch: {split_sizes} != {reference}"
                )
            self._starvla_metadata_split_checked = True
            print(
                "[fused_text_stack_patch] metadata check: split sizes OK",
                flush=True,
            )

        visual_kwargs = dict(kwargs)
        visual_kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **visual_kwargs,
        )
        vision_output.pooler_output = torch.split(
            vision_output.pooler_output, split_sizes
        )
        return vision_output

    _m.Qwen3_5Model.get_image_features = _gif_cached


if env_bool("STARVLA_FUSED_VISION"):
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
        if input_ids is None or get_active_qwen_metadata() is None:
            return _orig_placeholder_mask(
                self,
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
        special_image_mask = (input_ids == self.config.image_token_id)
        special_video_mask = (input_ids == self.config.video_token_id)
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return special_image_mask, special_video_mask

    _orig_placeholder_mask = _m.Qwen3_5Model.get_placeholder_mask
    _m.Qwen3_5Model.get_placeholder_mask = _fast_placeholder_mask


if env_bool("STARVLA_FAST_MM_MERGE"):
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
# Fix: the dataloader worker records the image-token positions as immutable
# flat indices in QwenBatchLayout.  The model copies each repeated index tuple
# into a bounded per-rank device cache and merges with index_copy; its backward
# is index_select, with no nonzero and no synchronization.  Comparing the
# descriptor length with image_embeds.shape[0] restores count validation using
# host-side shape integers only.
# ---------------------------------------------------------------------------


def _patch_index_mm_merge():
    _orig_model_forward = _m.Qwen3_5Model.forward

    def _mm_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                    video_grid_thw=None, mm_token_type_ids=None, cache_position=None, **kwargs):
        # Only the image-only training shape takes the fast path; anything
        # else (embeds-only, video, generation) keeps stock behavior.
        active = get_active_qwen_metadata()
        if (
            active is None
            or kwargs.get("return_dict", True) is False
            or input_ids is None
            or inputs_embeds is not None
            or pixel_values is None
            or pixel_values_videos is not None
            or image_grid_thw is None
            or video_grid_thw is not None
            or past_key_values is not None
            or kwargs.get("use_cache")
        ):
            return _orig_model_forward(
                self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position, **kwargs)

        layout = active.layout
        if tuple(input_ids.shape) != (layout.batch_size, layout.seq_len):
            raise RuntimeError(
                "active Qwen metadata does not match input_ids: "
                f"descriptor={(layout.batch_size, layout.seq_len)}, "
                f"tensor={tuple(input_ids.shape)}"
            )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        flat_indices = layout.flat_image_token_indices
        index_key = (
            "image_indices",
            layout.batch_size,
            layout.seq_len,
            flat_indices,
            device_key(input_ids),
        )
        idx = active.cache.image_indices.get_or_create(
            index_key,
            lambda: torch.tensor(
                flat_indices, dtype=torch.long, device=input_ids.device
            ),
        )
        if len(flat_indices) != image_embeds.shape[0]:
            raise RuntimeError(
                f"image token count ({len(flat_indices)}) != vision features "
                f"({image_embeds.shape[0]}); processor/collate mismatch"
            )
        if _check_metadata_cache and not getattr(
            self, "_starvla_metadata_indices_checked", False
        ):
            reference = tuple(
                (input_ids.cpu().view(-1) == self.config.image_token_id)
                .nonzero(as_tuple=False)
                .squeeze(1)
                .tolist()
            )
            if reference != flat_indices:
                raise RuntimeError(
                    "[metadata_cache] image indices mismatch: "
                    f"descriptor={flat_indices}, tensor={reference}"
                )
            self._starvla_metadata_indices_checked = True
            print(
                "[fused_text_stack_patch] metadata check: image indices OK",
                flush=True,
            )
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

    _m.Qwen3_5Model.forward = _mm_forward


if env_bool("STARVLA_INDEX_MM_MERGE"):
    _patch_index_mm_merge()
    print("[fused_text_stack_patch] index-based mm merge (STARVLA_INDEX_MM_MERGE)", flush=True)
