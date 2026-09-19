"""视觉编码器实现。

torch 是**惰性导入**的:只有真正构造 SigLIP2 时才 import。这样在没装
torch 的机器上(本机 dt env 就没有)仍能导入本模块、跑 mock 编码器、
测通整条流水线。
"""
from __future__ import annotations

import numpy as np

from ..config import EncoderConfig
from ..registry import register


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@register("encoder", "hist")
class HistEncoder:
    """无需权重的占位编码器:RGB 联合直方图 + 梯度能量。

    ⚠️ 这不是好的语义表征。它的唯一用途是让流水线在没有权重时可跑通。
    任何用它得出的精度数字都不能说明方法有效 —— 参见早先的教训:
    smptebars vs testsrc2 上 AUC=1.000 只证明管路通了。
    """

    def __init__(self, cfg: EncoderConfig | None = None, *, bins: int = 8):
        self.bins = bins
        self.dim = bins ** 3 + 4
        self.cfg = cfg

    def encode(self, frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_3d(frames)
        if frames.ndim == 3:
            frames = frames[None]
        out = np.empty((len(frames), self.dim), dtype=np.float32)
        b = self.bins
        for i, f in enumerate(frames):
            q = np.clip((f * b).astype(int), 0, b - 1)
            idx = (q[..., 0] * b + q[..., 1]) * b + q[..., 2]
            h = np.bincount(idx.ravel(), minlength=b ** 3).astype(np.float32)
            h /= max(h.sum(), 1.0)
            gy, gx = np.gradient(f.mean(axis=2))
            out[i] = np.concatenate(
                [h, [np.abs(gx).mean(), np.abs(gy).mean(), f.mean(), f.std()]])
        return _l2(out)


@register("encoder", "random")
class RandomEncoder:
    """确定性随机投影。用于测试维度契约,不做任何语义。"""

    def __init__(self, cfg: EncoderConfig | None = None, *, dim: int = 128,
                 seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._proj: np.ndarray | None = None

    def encode(self, frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_3d(frames)
        if frames.ndim == 3:
            frames = frames[None]
        flat = frames.reshape(len(frames), -1).astype(np.float32)
        if self._proj is None or self._proj.shape[0] != flat.shape[1]:
            self._proj = self._rng.normal(
                0, 1 / np.sqrt(flat.shape[1]), (flat.shape[1], self.dim)
            ).astype(np.float32)
        return _l2(flat @ self._proj)


@register("encoder", "siglip2")
class SigLIP2Encoder:
    """冻结的 SigLIP 2。逐帧,不做时序平均。

    逐帧正是 needle 检测要的粒度 —— 已知 SigLIP2 抽帧取平均做视频*检索*
    很弱,但那测的是视频级语义检索,与本用途无关(docs/01_MODELS.md §1.1)。
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._model = None
        self._proc = None
        self.dim = 1152                    # so400m

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        dtype = getattr(torch, self.cfg.dtype)
        self._proc = AutoProcessor.from_pretrained(self.cfg.model_id)
        self._model = AutoModel.from_pretrained(
            self.cfg.model_id, torch_dtype=dtype).to(self.cfg.device).eval()
        for p in self._model.parameters():        # 冻结
            p.requires_grad_(False)
        self.dim = self._model.config.vision_config.hidden_size

    def encode(self, frames: np.ndarray) -> np.ndarray:
        import torch
        self._lazy()
        frames = np.atleast_3d(frames)
        if frames.ndim == 3:
            frames = frames[None]
        outs = []
        bs = self.cfg.batch_size
        for i in range(0, len(frames), bs):
            batch = (frames[i:i + bs] * 255).astype(np.uint8)
            inputs = self._proc(images=list(batch), return_tensors="pt")
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self._model.get_image_features(**inputs)
            # transformers 5.x 可能返回输出对象而非张量
            if not isinstance(feat, torch.Tensor):
                feat = getattr(feat, "pooler_output", None)
                if feat is None:
                    raise RuntimeError(
                        "get_image_features 未返回张量,也没有 pooler_output")
            outs.append(feat.float().cpu().numpy())
        return _l2(np.concatenate(outs, axis=0))


def build_encoder(cfg: EncoderConfig):
    from ..registry import build
    return build("encoder", cfg.name, cfg)
