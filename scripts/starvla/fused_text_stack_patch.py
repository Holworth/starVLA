# [OPT #14 fused text stack / #15 fused vision tower,
#  docs/qwenpi_zero2_h200_optimization_log.md Round 9]
# Fused text-stack compile: put the forward-phase glue between the 32
# per-layer CUDA graphs INTO the compiled region.
#
# REQUIRES the [OPT #13] torch 2.7.1 container stack (torch 2.7.1+cu126,
# flash_attn 2.8.0.post2, causal-conv1d 1.5.2) AND, for STARVLA_FLA_TRACE=1,
# the container-side fla edit: comment out @torch.compiler.disable at
# fla/ops/gated_delta_rule/chunk.py:220 (a .bak is kept next to it). Neither
# is part of the customer-pinned environment — see the optimization log for
# the deviation list.
#
# Background (milestone2 gap analysis, docs/qwenpi_zero2_h200_final_report.md
# section 3): with per-layer compiled forwards ([OPT #6]) the GPU still idles
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
#     per-layer compiled bound method ([OPT #6]) is involved;
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


def _group_runner(layers_group, mode):
    layer_forward = _m.Qwen3_5DecoderLayer.forward  # unbound: no hooks, no [OPT #6] wrapper

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
    # 2026-07-05). Per-layer compiles ([OPT #6]) dodge this because each layer
    # is its own top-level compile. Two supported shapes:
    #   STARVLA_FUSED_GROUP_SIZE=0 (default): whole stack in one compile,
    #     mode default (fusion, NO cudagraphs);
    #   STARVLA_FUSED_GROUP_SIZE=N: ceil(32/N) independent top-level compiles
    #     (safe with reduce-overhead, like per-layer, but N x less glue).
    layers = tuple(self.layers[: self.config.num_hidden_layers])
    if os.environ.get("STARVLA_FLA_TRACE"):
        # causal_conv1d 1.5.2 registers custom ops but calls them with a
        # non-contiguous out= tensor, which dynamo rejects. Force the HF
        # torch-native fallback (F.silu(self.conv1d(x))) — numerically the
        # same op — so inductor owns the conv inside the graph.
        for layer in layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.linear_attn.causal_conv1d_fn = None
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
    # [OPT #17] HF's _update_linear_attn_mask runs torch.all(attention_mask==1)
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
# [OPT #15] Fused VISION tower (STARVLA_FUSED_VISION=1): compile the 24 ViT
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

    def _vm_forward(self, hidden_states, grid_thw, **kwargs):
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            return _orig_vm_forward(self, hidden_states, grid_thw, **kwargs)
        # preamble identical to HF (patch embed + pos embeds are weight-
        # dependent, must recompute every step)
        hidden_states = self.patch_embed(hidden_states)
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
        gcpu = grid_thw.cpu()
        gkey = gcpu.numpy().tobytes()
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


if os.environ.get("STARVLA_FUSED_VISION"):
    _patch_vision_tower()
    print("[fused_text_stack_patch] vision tower fused (STARVLA_FUSED_VISION)", flush=True)


# ---------------------------------------------------------------------------
# [OPT #17] Sync-free multimodal embedding merge (STARVLA_FAST_MM_MERGE=1).
#
# HF's Qwen3_5Model.get_placeholder_mask validates that the number of image
# placeholder tokens matches the vision features via
# `inputs_embeds[special_image_mask].numel() == image_features.numel()` — a
# boolean-mask gather (cub DeviceSelect) followed by a host bool() readback,
# i.e. one more CPU<->GPU round trip per modality per step, sitting exactly
# in the exposed gap between the vision graph and the first text-stack graph
# (milestone4 step-40: ~150 tiny D2H syncs in 16 ms there, shared with the
# in-model MRoPE compute that [OPT #12] removes).
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
