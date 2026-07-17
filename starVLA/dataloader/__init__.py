import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )

        # [OPT #3/#7, docs/qwenpi_zero2_h200_final_report.md] Opt-in fast
        # collate: HF preprocessing in workers (+ HostPinnedBatch wrapping).
        chosen_collate = collate_fn
        if str(vla_dataset_cfg.get("preprocess_in_collate", False)).lower() in ("true", "1"):
            from transformers import AutoProcessor
            from starVLA.dataloader.lerobot_datasets import QwenVLPreprocessCollate
            import torch as _torch

            processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
            processor.tokenizer.padding_side = "left"
            cot_prompt = (
                vla_dataset_cfg.get("CoT_prompt", None) if "CoT_prompt" in vla_dataset_cfg else None
            )
            pixel_dtype = (
                _torch.bfloat16
                if str(vla_dataset_cfg.get("collate_pixels_bf16", True)).lower() in ("true", "1")
                else None
            )
            keep_examples = str(vla_dataset_cfg.get("collate_keep_examples", False)).lower() in ("true", "1")
            # [OPT #12 MRoPE precompute] Compute 3D MRoPE position ids in the
            # workers and ship them with the batch (see QwenVLPreprocessCollate).
            mrope_config = None
            if str(vla_dataset_cfg.get("collate_mrope_posids", False)).lower() in ("true", "1"):
                from transformers import AutoConfig

                mrope_config = AutoConfig.from_pretrained(cfg.framework.qwenvl.base_vlm)
                # The collate uses a Qwen3_5Model shim; other Qwen-VL variants
                # have DIFFERENT mrope semantics (e.g. Qwen2.5-VL scales image
                # temporal ids by tokens_per_second=4) and would train silently
                # wrong. Refuse rather than inject mismatched position ids.
                if getattr(mrope_config, "model_type", None) != "qwen3_5":
                    logger.warning(
                        "[dataloader] collate_mrope_posids disabled: base_vlm model_type "
                        f"'{getattr(mrope_config, 'model_type', None)}' != 'qwen3_5' "
                        "(the precompute shim implements Qwen3.5 mrope semantics only)"
                    )
                    mrope_config = None
                else:
                    logger.info("[dataloader] collate_mrope_posids enabled: MRoPE position ids precomputed in workers")
            chosen_collate = QwenVLPreprocessCollate(
                processor,
                cot_prompt=cot_prompt,
                pixel_dtype=pixel_dtype,
                keep_examples=keep_examples,
                pad_to=int(vla_dataset_cfg.get("collate_pad_to", 0) or 0),
                host_batch=str(vla_dataset_cfg.get("collate_host_batch", False)).lower() in ("true", "1"),
                mrope_config=mrope_config,
            )
            logger.info("[dataloader] preprocess_in_collate enabled: HF processor runs in DataLoader workers")

        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": chosen_collate,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
