# nsys 入口:先 import nvtx_patch(给 VLATrainer/QwenPI 打 NVTX + cudaProfiler 门),
# 再复刻 train_starvla.__main__ 的参数解析并调 main(cfg)。
# accelerate launch 用本文件替代 train_starvla.py(同样的 CLI 参数透传)。
import argparse
import os

from omegaconf import OmegaConf
import starVLA.training.train_starvla as T
import nvtx_patch  # noqa: F401  (导入即生效,必须在 T 之后)

if os.environ.get("STARVLA_FUSED_TEXT_STACK"):
    import fused_text_stack_patch  # noqa: F401  (fused text-stack groups + fused vision tower + sync-free mm merge)

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", type=str, required=True)
args, clip = parser.parse_known_args()

cfg = OmegaConf.load(args.config_yaml)
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(T.normalize_dotlist_args(clip)))
cfg = T.apply_config_compat(cfg)
cfg.config_yaml = args.config_yaml
T.main(cfg)
