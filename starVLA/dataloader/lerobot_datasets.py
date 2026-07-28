# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import logging
from pathlib import Path
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)
from starVLA.model.qwenpi_metadata import (
    build_qwen_batch_layout,
    env_bool,
    metadata_cache_enabled,
)

logger = logging.getLogger(__name__)

def collate_fn(batch):
    return batch


class QwenVLPreprocessCollate:
    """[OPT #3, docs/qwenpi_zero2_h200_final_report.md] Run the HF Qwen-VL
    processor inside DataLoader workers (opt-in).

    The default identity collate ships raw PIL images + strings to the main
    process, which then spends ~40 ms/step of GPU-idle time running the
    processor inside model.forward. Doing it here overlaps that CPU work with
    the previous step's GPU execution, and pin_memory=True becomes effective
    because the batch is now a dict of CPU tensors.

    Enabled via ``datasets.vla_data.preprocess_in_collate: true``; consumed by
    ``Qwen_PI.forward`` which accepts this dict batch alongside the legacy
    list-of-examples format.
    """

    def __init__(
        self, processor, cot_prompt=None, pixel_dtype=None, keep_examples=False, pad_to=0
    ):
        self.processor = processor
        self._metadata_cache_enabled = metadata_cache_enabled()
        self._metadata_config = None
        self._metadata_compat_warning_emitted = False
        self.cot_prompt = cot_prompt
        self.pixel_dtype = pixel_dtype
        # [OPT #6/#10] Pad token sequences to a fixed length (left padding) so
        # downstream torch.compile(dynamic=False) / CUDA Graphs see static
        # shapes. 0 = batch-dynamic.
        self.pad_to = int(pad_to or 0)
        # Shipping the raw PIL examples through the worker queue costs extra
        # pickle/unpickle time per batch; only keep them when a consumer (e.g.
        # eval_action_model) actually needs the raw batch.
        self.keep_examples = keep_examples

    def _resolve_metadata_config(self):
        """Resolve Qwen3.5 processor constants lazily for legacy compatibility."""

        if self._metadata_config is not None:
            return self._metadata_config

        token_id = getattr(self.processor, "image_token_id", None)
        if token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            converter = getattr(tokenizer, "convert_tokens_to_ids", None)
            if converter is None:
                raise ValueError(
                    "the processor exposes neither image_token_id nor a tokenizer "
                    "with convert_tokens_to_ids('<|image_pad|>')"
                )
            token_id = converter("<|image_pad|>")

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("processor.image_processor is missing")

        merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is None:
            merge_size = getattr(image_processor, "spatial_merge_size", None)
        if merge_size is None:
            raise ValueError(
                "processor.image_processor exposes neither merge_size nor "
                "spatial_merge_size"
            )
        self._metadata_config = (token_id, merge_size)
        return self._metadata_config

    def _build_metadata(self, qwen_inputs, image_counts):
        """Build immutable CPU-only layout metadata for Qwen3.5 batches."""
        import torch

        def fallback(message, *args):
            if not self._metadata_compat_warning_emitted:
                logger.warning("[dataloader] " + message, *args)
                self._metadata_compat_warning_emitted = True
            return None

        if "mm_token_type_ids" not in qwen_inputs:
            return fallback(
                "metadata cache needs Qwen3.5 mm_token_type_ids; using the stock path"
            )
        image_grid = qwen_inputs.get("image_grid_thw")
        if torch.is_tensor(image_grid) and image_grid.dtype != torch.long:
            return fallback(
                "image_grid_thw has dtype %s; cached grid reconstruction requires "
                "torch.int64, so this batch uses the stock path",
                image_grid.dtype,
            )
        if any(
            qwen_inputs.get(name) is not None
            for name in ("pixel_values_videos", "video_grid_thw")
        ):
            return fallback("metadata caching is image/text only; using the stock video path")

        image_token_id, spatial_merge_size = self._resolve_metadata_config()
        return build_qwen_batch_layout(
            qwen_inputs,
            image_counts=image_counts,
            image_token_id=image_token_id,
            spatial_merge_size=spatial_merge_size,
        )

    def __call__(self, batch):
        import numpy as np
        import torch
        import os as _os
        import time as _time

        _t0 = _time.perf_counter()
        image_counts = [len(ex["image"]) for ex in batch]
        messages = []
        for ex in batch:
            content = [{"type": "image", "image": img} for img in ex["image"]]
            instruction = ex["lang"]
            prompt = (
                self.cot_prompt.replace("{instruction}", instruction)
                if self.cot_prompt
                else instruction
            )
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        pad_kwargs = (
            {"padding": "max_length", "max_length": self.pad_to}
            if self.pad_to
            else {"padding": True}
        )
        # qwen_inputs = the HF processor's full output, a dict of pure torch
        # tensors: input_ids / attention_mask / mm_token_type_ids [B, T]
        # (left-padded to pad_to when set), pixel_values (patch sequences of
        # ALL images in the batch, optionally pre-cast to bf16 below) and,
        # while qwen_metadata is built, image_grid_thw [n_images, 3]. The
        # immutable grid tensor is then removed and reconstructed from the
        # descriptor's per-rank GPU cache. No raw PIL.Image / str survives.
        qwen_inputs = dict(
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **pad_kwargs,
            )
        )
        if self.pixel_dtype is not None and "pixel_values" in qwen_inputs:
            qwen_inputs["pixel_values"] = qwen_inputs["pixel_values"].to(self.pixel_dtype)

        out = {
            "qwen_inputs": qwen_inputs,
            "actions": torch.from_numpy(np.stack([np.asarray(ex["action"]) for ex in batch])),
        }
        if self._metadata_cache_enabled:
            qwen_metadata = self._build_metadata(qwen_inputs, image_counts)
            if qwen_metadata is not None:
                # Keep the descriptor separate: qwen_inputs is splatted
                # directly into the HF model and must remain tensor-only.
                out["qwen_metadata"] = qwen_metadata
                # The grid is immutable layout metadata. Removing its CPU
                # tensor here prevents Accelerate from copying it to CUDA on
                # every batch; Qwen_PI materializes one per-rank device copy
                # from qwen_metadata and reuses it through the bounded cache.
                qwen_inputs.pop("image_grid_thw")
        if self.keep_examples:
            out["examples"] = batch
        if "state" in batch[0]:
            out["state"] = torch.from_numpy(np.stack([np.asarray(ex["state"]) for ex in batch]))
        if env_bool("STARVLA_COLLATE_TIMING"):
            print(
                f"[collate] pid={_os.getpid()} t={_time.perf_counter():.3f} "
                f"dur={(_time.perf_counter()-_t0)*1000:.1f}ms",
                flush=True,
            )
        return out


def build_preprocess_collate(cfg, default=None):
    """[OPT #3] Dispatch for the opt-in worker-side preprocessing collate.

    Pure dispatch, no model-family specifics: returns ``default`` (the
    identity collate) when ``datasets.vla_data.preprocess_in_collate`` is
    disabled or the configured backbone has no preprocessing collate
    implemented; otherwise delegates to the per-family builder. Adding a
    new backbone = one more dispatch arm here + its build_*_collate.
    """
    vla_dataset_cfg = cfg.datasets.vla_data
    if str(vla_dataset_cfg.get("preprocess_in_collate", False)).lower() not in ("true", "1"):
        return default
    # Qwen-VL family: frameworks carrying a ``qwenvl`` config section.
    if "qwenvl" in cfg.framework:
        return build_qwenvl_preprocess_collate(cfg)
    logger.warning(
        "[dataloader] preprocess_in_collate=true but no preprocessing collate "
        "is implemented for this backbone; falling back to the default collate"
    )
    return default


def build_qwenvl_preprocess_collate(cfg):
    """Construct the Qwen-VL worker-side preprocessing collate from config
    (see QwenVLPreprocessCollate for what it does per batch)."""
    import torch
    from transformers import AutoProcessor

    vla_dataset_cfg = cfg.datasets.vla_data
    processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
    processor.tokenizer.padding_side = "left"
    cot_prompt = vla_dataset_cfg.get("CoT_prompt", None) if "CoT_prompt" in vla_dataset_cfg else None
    pixel_dtype = (
        torch.bfloat16
        if str(vla_dataset_cfg.get("collate_pixels_bf16", True)).lower() in ("true", "1")
        else None
    )
    keep_examples = str(vla_dataset_cfg.get("collate_keep_examples", False)).lower() in ("true", "1")
    logger.info("[dataloader] preprocess_in_collate enabled: HF processor runs in DataLoader workers")
    return QwenVLPreprocessCollate(
        processor,
        cot_prompt=cot_prompt,
        pixel_dtype=pixel_dtype,
        keep_examples=keep_examples,
        pad_to=int(vla_dataset_cfg.get("collate_pad_to", 0) or 0),
    )


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        return data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/bar/starvla_cotrain_libero.yaml", help="Path to YAML config")
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
        pass
