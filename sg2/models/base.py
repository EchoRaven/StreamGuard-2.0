"""组件协议。

每个协议定义一层的**契约**,实现可随意替换。所有协议都有一个 mock 实现
(见各自模块),这样整条流水线在没有 GPU、没有权重的情况下也能端到端跑通
—— 对当前硬件状况(4×2080Ti、无真实数据)这不是妥协而是必需。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..contracts import SentinelOutput, StreamStep  # noqa: F401  重导出


# ---------------------------------------------------------------- 编码器


@runtime_checkable
class VisionEncoder(Protocol):
    """冻结的视觉编码器。逐帧,不做时序聚合。"""

    dim: int

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) float[0,1] -> (N, dim) 已 L2 归一化。"""
        ...


# ---------------------------------------------------------------- Sentinel


@runtime_checkable
class Sentinel(Protocol):
    """多通道廉价打分器。覆盖 100% 流量。"""

    def score_frame(self, t_s: float, *, frame: np.ndarray | None = None,
                    codec: np.ndarray | None = None,
                    text: str | None = None) -> SentinelOutput:
        ...

    def reset(self) -> None:
        ...


# ---------------------------------------------------------------- 中间层


@runtime_checkable
class StreamingVLM(Protocol):
    """中间层:KV 分段 + 三元动作。

    实现必须是**因果**的:`step()` 只能看到已经 `ingest` 过的帧。
    """

    def set_policy(self, policy_text: str, key: str) -> bool:
        """装载政策头。返回 True 表示缓存命中(未重编码)。"""
        ...

    def ingest(self, frames: np.ndarray, t_s: float) -> None:
        """追加帧到视觉滑窗。"""
        ...

    def step(self) -> StreamStep:
        """产出一个动作。"""
        ...

    def close_event(self) -> int:
        """闭合事件 = 截断事件段。返回释放的 token 数。"""
        ...

    def reset(self) -> None:
        ...
