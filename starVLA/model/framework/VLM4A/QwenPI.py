# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPIDefaultConfig:
    """QwenPI (QwenFM) framework default parameters.

    Layer-wise cross-DiT flow-matching action prediction conditioned on
    multi-layer VLM hidden states.  All fields can be overridden by the
    corresponding key in the YAML ``framework:`` section.
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenPI"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (auto-set at runtime from model config)
            "vl_hidden_dim": 2048,
            # Number of VL transformer layers (auto-set at runtime)
            "num_vl_layers": 36,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "LayerwiseFM",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            # Repeat factor for flow-matching loss
            "repeated_diffusion_steps": 2,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # DiT architecture settings — shape fields (num_layers,
            # input_embedding_dim, cross_attention_dim, num_attention_heads)
            # are auto-populated by populate_layerwise_dit_cfg at runtime.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFM")
@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model (PI variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Layer-wise cross-DiT diffusion head fed by multi-layer VLM hidden states

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    #
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenPIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors;
        # we keep the last num_hidden_layers of them, so DiT depth must match.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # QwenPI: DiT runs at the LLM hidden size (no compression).  Tell the
        # action head exactly that — the head itself does not look at qwenvl.*.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # [OPT mrope-cache] Cache the 3D MRoPE position ids across steps
        # (framework.qwenvl.mrope_posid_cache). HF recomputes them every
        # forward via a per-sample python loop with 3x .item() per image —
        # ~150 GPU syncs / ~13 ms of serialized host time per step. The ids
        # are a pure function of (mm_token_type_ids, attention_mask,
        # image_grid_thw) — sequence structure only, no weights, no token
        # values — so identical inputs (fixed collate_pad_to + fixed camera
        # resolutions => a handful of distinct keys) can reuse the cached
        # GPU tensor; HF skips its own compute whenever position_ids is
        # not None. Verify anytime with STARVLA_CHECK_POSIDS=1.
        self._mrope_cache_enabled = (
            str(self.config.framework.qwenvl.get("mrope_posid_cache", False)).lower() in ("true", "1")
        )
        self._mrope_posid_cache = {}

    def _inject_cached_position_ids(self, qwen_inputs):
        """[OPT mrope-cache] Fill qwen_inputs["position_ids"] from the cache,
        computing once per distinct (mm_token_type_ids, attention_mask,
        image_grid_thw) via the model's OWN compute_3d_position_ids (so the
        semantics always match the loaded model — no cross-variant risk).
        The key readbacks touch pure input tensors (already H2D-complete),
        so they cost ~µs of API time, not a GPU pipeline drain."""
        if "position_ids" in qwen_inputs:
            return
        input_ids = qwen_inputs.get("input_ids")
        mm_tt = qwen_inputs.get("mm_token_type_ids")
        grid = qwen_inputs.get("image_grid_thw")
        mask = qwen_inputs.get("attention_mask")
        if input_ids is None or mm_tt is None or grid is None:
            return
        inner = getattr(getattr(self.qwen_vl_interface, "model", None), "model", None)
        if inner is None or not hasattr(inner, "compute_3d_position_ids"):
            return
        mm_cpu = mm_tt.cpu().numpy()
        mask_cpu = mask.cpu().numpy() if mask is not None else None
        grid_cpu = grid.cpu().numpy()
        key = (
            mm_cpu.tobytes(),
            mask_cpu.tobytes() if mask_cpu is not None else b"",
            grid_cpu.tobytes(),
        )
        pos = self._mrope_posid_cache.get(key)
        if pos is None:
            # Batch-key miss. At small bs the batch key repeats and this is
            # rare; at bs>=24 the key is a new combination of per-sample
            # instruction lengths nearly every step, so the batch cache
            # silently degrades to per-step misses (HF's stock per-sample
            # loop: ~60 ms/step at bs32, 579 D2H readbacks + ~1.5k tiny
            # kernels). position ids are computed independently per sample
            # inside HF's loop, so cache at SAMPLE granularity (per-sample
            # keys DO repeat: one instruction x a fixed image layout) and
            # assemble the batch with a single cat.
            pos = self._assemble_posids_per_sample(
                inner, input_ids, mm_tt, mask, grid, mm_cpu, mask_cpu, grid_cpu
            )
            if pos is None:
                # Non-uniform images/sample or sample compute declined:
                # stock full-batch fallback (original PR#10 behavior).
                pos = inner.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=None,
                    image_grid_thw=grid,
                    attention_mask=mask,
                    mm_token_type_ids=mm_tt,
                )
            if pos is None:
                return
            if len(self._mrope_posid_cache) < 64:
                self._mrope_posid_cache[key] = pos
        qwen_inputs["position_ids"] = pos

    def _assemble_posids_per_sample(
        self, inner, input_ids, mm_tt, mask, grid, mm_cpu, mask_cpu, grid_cpu
    ):
        """[OPT mrope-cache] Sample-granularity position-id cache. Requires a
        uniform images-per-sample layout to map grid rows to samples (true
        for this workload; returns None otherwise so the caller falls back).
        Per-sample results come from the model's OWN compute_3d_position_ids
        on 1-sample slices, so semantics match HF exactly; the batch is
        rebuilt with one cat along dim=1 ([3, B, S])."""
        B = input_ids.shape[0]
        n_img = grid.shape[0]
        if B == 0 or n_img % B != 0:
            return None
        ipp = n_img // B
        cache = getattr(self, "_mrope_sample_cache", None)
        if cache is None:
            cache = self._mrope_sample_cache = {}
        parts = []
        for i in range(B):
            skey = (
                mm_cpu[i].tobytes(),
                mask_cpu[i].tobytes() if mask_cpu is not None else b"",
                grid_cpu[i * ipp : (i + 1) * ipp].tobytes(),
            )
            p = cache.get(skey)
            if p is None:
                p = inner.compute_3d_position_ids(
                    input_ids=input_ids[i : i + 1],
                    inputs_embeds=None,
                    image_grid_thw=grid[i * ipp : (i + 1) * ipp],
                    attention_mask=mask[i : i + 1] if mask is not None else None,
                    mm_token_type_ids=mm_tt[i : i + 1],
                )
                if p is None:
                    return None
                if len(cache) < 4096:
                    cache[skey] = p
            parts.append(p)
        return torch.cat(parts, dim=1)

    def _check_precomputed_position_ids(self, qwen_inputs):
        """[OPT mrope-cache] Debug guard (STARVLA_CHECK_POSIDS=1): recompute
        the 3D MRoPE position ids with the stock HF path and assert the
        cached ones are identical. Costs the ~13 ms/step this optimization
        removes, so only enable for validation runs."""
        import os

        if not os.environ.get("STARVLA_CHECK_POSIDS") or "position_ids" not in qwen_inputs:
            return
        ref = self.qwen_vl_interface.model.model.compute_3d_position_ids(
            input_ids=qwen_inputs.get("input_ids"),
            inputs_embeds=None,
            image_grid_thw=qwen_inputs.get("image_grid_thw"),
            attention_mask=qwen_inputs.get("attention_mask"),
            mm_token_type_ids=qwen_inputs.get("mm_token_type_ids"),
        )
        # Explicit raise (not assert): must not be strippable by python -O,
        # and the OK line below must never print without the comparison.
        if ref is None or not torch.equal(ref, qwen_inputs["position_ids"]):
            raise RuntimeError(
                "cached MRoPE position_ids != HF compute_3d_position_ids "
                f"(shapes {qwen_inputs['position_ids'].shape} vs {None if ref is None else ref.shape})"
            )
        if not getattr(self, "_posids_check_logged", False):
            print("[QwenPI] STARVLA_CHECK_POSIDS: cached position_ids == HF compute OK", flush=True)
            self._posids_check_logged = True

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        """Run QwenVL and return (layer-wise hidden states, attention_mask) for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list, attention_mask

    def forward(
        self,
        examples=None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: either
                - List[dict], each dict requires:
                    - image: List[PIL.Image] (multi-view)
                    - lang: str instruction
                    - action: np.ndarray or list shaped [T, action_dim]
                - or a pre-collated dict from QwenVLPreprocessCollate with keys
                  qwen_inputs (CPU tensors), actions, optional state. The HF
                  processor already ran in DataLoader workers in that case.
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        # [OPT #3] Two input formats, selected by how the DataLoader was
        # configured (datasets.vla_data.preprocess_in_collate):
        # (A) pre-collated dict from QwenVLPreprocessCollate (this branch):
        #     {"qwen_inputs": HF-processor output as pure CPU tensors
        #      (input_ids / attention_mask / mm_token_type_ids [B, T],
        #      pixel_values, image_grid_thw), "actions": tensor,
        #      optional "state": tensor}. The HF processor already ran inside
        #     the DataLoader workers, so the main thread only does H2D copies
        #     + the VLM forward.
        # (B) legacy list-of-dicts (else branch): raw PIL images + str
        #     instructions; the HF processor runs HERE, on the training
        #     critical path, inside _encode_vl_hidden_states.
        # Both branches meet the same contract for Step 4 below: vl_embs_list
        # (last N layer hidden states), backbone_attention_mask, and
        # actions/state as device tensors in base_hidden's dtype.
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            # Pre-collated path: only H2D copies + VLM forward on the main thread.
            device = self.qwen_vl_interface.model.device
            qwen_inputs = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in examples["qwen_inputs"].items()
            }
            backbone_attention_mask = qwen_inputs.get("attention_mask", None)
            if self._mrope_cache_enabled:
                self._inject_cached_position_ids(qwen_inputs)
                self._check_precomputed_position_ids(qwen_inputs)
            # Same bf16 autocast the legacy path applies inside
            # _encode_vl_hidden_states: the VLM backbone loads and trains in
            # bf16 (DeepSpeed bf16 mode); only the execution site of the HF
            # preprocessing differs between the two branches, not precision.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                expected_layers = len(self.action_model.model.transformer_blocks)
                vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
            base_hidden = vl_embs_list[-1]
            actions = examples["actions"].to(device, dtype=base_hidden.dtype, non_blocking=True)
            state = examples.get("state", None)
            if state is not None:
                state = state.to(device, dtype=base_hidden.dtype, non_blocking=True)
        else:
            batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions_list = [example["action"] for example in examples]  # label [B， len, 7]
            state_list = [example["state"] for example in examples] if "state" in examples[0] else None

            # Step 1: encode through QwenVL
            vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
            base_hidden = vl_embs_list[-1]
            # The actions/state tensor conversion moved here VERBATIM from
            # Step 4 below: branch (A) already receives device tensors, and
            # re-wrapping a CUDA tensor in torch.tensor(np.array(...)) would
            # force a GPU->CPU sync plus an extra copy. Each branch now hands
            # Step 4 ready device tensors.
            actions = torch.tensor(
                np.array(actions_list), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            state = (
                torch.tensor(np.array(state_list), device=base_hidden.device, dtype=base_hidden.dtype)
                if state_list is not None
                else None
            )

        # Optional right-padding of the encoder sequence to a fixed
        # length so the DiT sees static shapes (required for torch.compile
        # dynamic=False / CUDA Graphs to avoid per-length recompiles). Padded
        # positions are masked out, so cross-attention results are unchanged.
        pad_to = int(self.config.framework.action_model.get("pad_encoder_seq_to", 0) or 0)
        cur_len = vl_embs_list[0].shape[1]
        if pad_to and cur_len < pad_to:
            pad_len = pad_to - cur_len
            if backbone_attention_mask is None:
                backbone_attention_mask = torch.ones(
                    vl_embs_list[0].shape[0], cur_len, dtype=torch.bool, device=vl_embs_list[0].device
                )
            backbone_attention_mask = torch.nn.functional.pad(backbone_attention_mask, (0, pad_len), value=0)
            vl_embs_list = [torch.nn.functional.pad(h, (0, 0, 0, pad_len)) for h in vl_embs_list]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Label alignment: take the last chunk_len segment
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat features for each layer
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1) if state is not None else None

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(  # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to directly regress future actions (no diffusion sampling).

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_realtime(
        self,
        examples: Optional[List[dict]] = None,
        prev_action_chunk_normalized: Optional[np.ndarray] = None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        """RTC-aware inference: condition sampling on a known prefix.

        Fixes a known prefix of the action chunk to the previous prediction
        and resamples the rest, so chunks splice smoothly under execution
        latency.  Mode / schedule kwargs are forwarded to the action head's
        ``predict_action_realtime``.

        Args:
            examples: list of dict, same shape as ``predict_action``.
            prev_action_chunk_normalized: (B, T, action_dim) *normalized*
                continuous actions from the previous prediction.
            inference_delay: number of leading timesteps to keep as prefix.
            **kwargs: forwarded to ``self.action_model.predict_action_realtime``
                (e.g. ``mode``, ``suffix_length``, ``prefix_attention_schedule``,
                ``max_guidance_weight``).

        Returns:
            dict with key ``"normalized_actions"`` -> np.ndarray of shape
            ``(B, T, action_dim)``.
        """
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples)

        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, _ = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        prev_chunk_t = torch.from_numpy(np.array(prev_action_chunk_normalized)).to(
            base_hidden.device, dtype=torch.float32
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_realtime(
                vl_embs_list,
                state_t,
                prev_action_chunk=prev_chunk_t,
                inference_delay=inference_delay,
                **kwargs,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwen_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
