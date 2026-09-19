"""Sentinel 融合头。

把多个廉价通道(压缩域/ASR/OCR/弹幕/视觉嵌入)的分数融合成**单一连续风险分**,
供 CUSUM 使用。这是级联里唯一 always-on 的可训部件,约 2M 参数以内。

无 torch 时自动退化为 numpy 实现的逻辑回归 —— 本机 Turing sm_75 装 torch
需要 cu121 轮子(docs/04_TRAINING.md §1.1),而融合头本身不需要 GPU。

⚠️ 选型指标是 **AUPRC 不是 AUC**:正例率 <1% 时 AUC 会虚高。
⚠️ 验证集必须**按源视频切**,不能按帧切 —— 相邻帧几乎相同,按帧切会让
   验证集变成训练集的副本,指标虚高十几个点。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusionHead:
    """多通道 -> 单一风险分。

    默认用 numpy 逻辑回归(带类别权重),足以支撑 CUSUM 所需的连续分数。
    需要非线性时切到 torch 实现,接口不变。
    """
    dim: int
    lr: float = 0.05
    epochs: int = 300
    l2: float = 1e-3
    _w: np.ndarray | None = field(default=None, repr=False)
    _b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            sample_weight: np.ndarray | None = None) -> "FusionHead":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if d != self.dim:
            raise ValueError(f"特征维度 {d} != 声明的 {self.dim}")

        # 极度不平衡:正例率常 <1%,不加权则模型全预测负例
        if sample_weight is None:
            pos = max(y.sum(), 1.0)
            neg = max(n - pos, 1.0)
            sample_weight = np.where(y > 0, neg / pos, 1.0)
        sample_weight = sample_weight / sample_weight.mean()

        self._w = np.zeros(d)
        self._b = 0.0
        for _ in range(self.epochs):
            z = X @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * sample_weight
            self._w -= self.lr * ((X.T @ g) / n + self.l2 * self._w)
            self._b -= self.lr * g.mean()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """返回 logit —— CUSUM 要的是无界连续量,不是 [0,1] 概率。"""
        if self._w is None:
            raise RuntimeError("未训练")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self._w + self._b

    @property
    def weights(self) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("未训练")
        return self._w.copy()


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """平均精度。正例率 <1% 时这才是有意义的指标,AUC 会虚高。"""
    y = np.asarray(y_true).ravel()
    order = np.argsort(-np.asarray(scores).ravel())
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    return float((precision * y).sum() / n_pos)


def split_by_source(groups: np.ndarray, *, val_frac: float = 0.25,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """按源视频分组切分。

    按帧切会让相邻帧同时落进训练与验证集,指标虚高。groups 传每个样本
    所属的源视频 id(ClipRecord.source["provenance"] 或 clip id)。
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_ids = set(uniq[:n_val].tolist())
    mask = np.array([g in val_ids for g in groups])
    return ~mask, mask
