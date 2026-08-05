# nsys 入口:先 import nvtx_patch(给 VLATrainer/QwenPI 打 NVTX + cudaProfiler 门),
# 再复刻 train_starvla.__main__ 的参数解析并调 main(cfg)。
# accelerate launch 用本文件替代 train_starvla.py(同样的 CLI 参数透传)。
import argparse
import os

from omegaconf import OmegaConf
import starVLA.training.train_starvla as T
import nvtx_patch  # noqa: F401  (导入即生效,必须在 T 之后)
from starVLA.model.qwenpi_metadata import env_bool

if env_bool("STARVLA_FUSED_TEXT_STACK"):
    import fused_text_stack_patch  # noqa: F401  (fused text-stack groups + fused vision tower + sync-free mm merge)
else:
    print("[profile_entry] fused text stack disabled", flush=True)

if env_bool("STARVLA_DS_FAST_GRAD_NORM", default=True):
    import ds_grad_norm_patch  # noqa: F401  (multi-tensor ZeRO grad norm)
else:
    print("[profile_entry] ds fast grad norm disabled", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", type=str, required=True)
args, clip = parser.parse_known_args()

cfg = OmegaConf.load(args.config_yaml)
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(T.normalize_dotlist_args(clip)))
cfg = T.apply_config_compat(cfg)
cfg.config_yaml = args.config_yaml

# The trainer historically seeds inside prepare_training(), after the model
# and randomly initialized action head have already been built.  Opt-in early
# seeding makes independent profile runs genuinely comparable without changing
# the normal training entry point.
if env_bool("STARVLA_PROFILE_PREBUILD_SEED"):
    from accelerate.utils import set_seed

    rank = int(os.environ.get("RANK", "0"))
    seed = int(cfg.get("seed", 3047)) + rank
    set_seed(seed)
    print(f"[profile_entry] prebuild_seed={seed}", flush=True)

T.main(cfg)
