# 实测 starVLA QwenPI 的 backbone 序列长度:patch _encode_vl_hidden_states,
# 打印真实 attention_mask 形状/每样本非 pad 长度,然后立即退出(首个 forward 后)。
import argparse, os
from omegaconf import OmegaConf
import starVLA.training.train_starvla as T
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

_orig = Qwen_PI._encode_vl_hidden_states
def _patched(self, batch_images, instructions):
    embs, attn = _orig(self, batch_images, instructions)
    if attn is not None:
        lens = [int(x) for x in attn.sum(dim=1).tolist()]
        print(f"[SEQLEN] attn.shape={tuple(attn.shape)}  per-sample non-pad lens={lens}", flush=True)
    print(f"[SEQLEN] vl hidden last-layer shape={tuple(embs[-1].shape)} (B, L, D)", flush=True)
    os._exit(0)
Qwen_PI._encode_vl_hidden_states = _patched

p = argparse.ArgumentParser(); p.add_argument("--config_yaml", required=True)
a, clip = p.parse_known_args()
cfg = OmegaConf.load(a.config_yaml)
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(T.normalize_dotlist_args(clip)))
cfg = T.apply_config_compat(cfg); cfg.config_yaml = a.config_yaml
T.main(cfg)
