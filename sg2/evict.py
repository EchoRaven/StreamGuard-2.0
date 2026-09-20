"""帧驱逐:保留哪些帧,丢掉哪些。

**一个常见误解:以为这需要"哪一帧重要"的人工标注。不需要。**

驱逐是**压缩**问题不是分类问题。要问的是"丢掉这帧会不会改变判决",
而这个问题的答案**由模型自己给出** —— 用它自身的输出当目标,
不用 ground truth。所以监督信号是**自监督**的。

真正难的是别的:**在线决策时不知道未来**。t 时刻要决定留不留某帧,
而它有没有用取决于 t+k 发生什么。这是**延迟反馈下的预测问题**。

对策分两层,因为两种驱逐的**可逆性不同**:

| | 可逆? | 策略 |
| --- | --- | --- |
| KV 驱逐(视觉窗) | **可逆** —— 帧还在 ring buffer,可重新 ingest | 可激进、可学 |
| ring buffer 驱逐 | **不可逆** —— 滚出去就是硬 miss | 必须保守,只按时长/字节 |

离线时可以**事后回标**:重放整条流,用反事实消融确定哪些帧当时真的需要,
以此训练在线策略。全程不需要人工看一帧。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FrameRef:
    """驱逐决策看到的一帧。**不含像素** —— 决策只用得上嵌入与元数据。"""
    t_s: float
    emb: np.ndarray
    is_keyframe: bool = False
    motion: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------- 策略


def evict_fifo(frames: Sequence[FrameRef], keep: int) -> list[int]:
    """最旧的先丢。基线。

    ⚠️ 对**流式检测**特别糟:needle 往往就在刚过去的那段,
    但 FIFO 丢的恰恰是最早进来的 —— 而 needle 可能正在那儿。
    """
    return list(range(len(frames)))[-keep:] if keep < len(frames) \
        else list(range(len(frames)))


def evict_stride(frames: Sequence[FrameRef], keep: int) -> list[int]:
    """等间隔保留。保住时间跨度,牺牲局部密度。"""
    n = len(frames)
    if keep >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, keep).round().astype(int)
    return sorted(set(idx.tolist()))


def evict_coverage(frames: Sequence[FrameRef], keep: int) -> list[int]:
    """k-center 贪心:保留在嵌入空间**覆盖最广**的一组。

    自监督 —— 只用嵌入之间的距离,不用任何标签。
    直觉:两帧几乎一样时留一帧就够,冗余帧应该先丢。
    """
    n = len(frames)
    if keep >= n:
        return list(range(n))
    E = np.stack([f.emb for f in frames])
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    sel = [n - 1]                       # 必留最新帧
    d = 1.0 - E @ E[sel[0]]
    while len(sel) < keep:
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, 1.0 - E @ E[i])
        d[i] = -1.0
    return sorted(sel)


def evict_salience(frames: Sequence[FrameRef], keep: int, *,
                   w_score: float = 1.0, w_motion: float = 0.5,
                   w_keyframe: float = 0.3, w_recency: float = 0.2
                   ) -> list[int]:
    """按显著性打分保留。

    压缩域信号(关键帧、运动能量)**免费** —— 编码器已经算过了。
    """
    n = len(frames)
    if keep >= n:
        return list(range(n))
    t = np.array([f.t_s for f in frames])
    rec = (t - t.min()) / max(float(np.ptp(t)), 1e-9)
    s = (w_score * np.array([f.score for f in frames])
         + w_motion * np.array([f.motion for f in frames])
         + w_keyframe * np.array([float(f.is_keyframe) for f in frames])
         + w_recency * rec)
    return sorted(np.argsort(-s)[:keep].tolist())


def evict_hybrid(frames: Sequence[FrameRef], keep: int, *,
                 coverage_share: float = 0.5) -> list[int]:
    """一半给覆盖,一半给显著性。**最新帧永远保留。**"""
    n = len(frames)
    if keep >= n:
        return list(range(n))
    k_cov = max(1, int(keep * coverage_share))
    sel = set(evict_coverage(frames, k_cov))
    sel.add(n - 1)
    for i in evict_salience(frames, keep):
        if len(sel) >= keep:
            break
        sel.add(i)
    return sorted(sel)


POLICIES: dict[str, Callable[..., list[int]]] = {
    "fifo": evict_fifo, "stride": evict_stride, "coverage": evict_coverage,
    "salience": evict_salience, "hybrid": evict_hybrid,
}


# ---------------------------------------------------------------- 自监督重要性


@dataclass
class ImportanceResult:
    per_frame: np.ndarray            # 丢掉该帧后判决的改变程度
    baseline: str
    n_calls: int

    @property
    def ranking(self) -> list[int]:
        return np.argsort(-self.per_frame).tolist()

    @property
    def concentrated(self) -> float:
        """重要性有多集中。1 = 全在一帧上,0 = 完全均匀。

        ⚠️ 接近 0 意味着**丢哪一帧都一样** —— 此时任何驱逐策略都无所谓,
        再优化也没有收益。先看这个数再决定要不要做。
        """
        p = np.asarray(self.per_frame, dtype=float)
        if p.sum() <= 0:
            return 0.0
        p = p / p.sum()
        h = -(p * np.log(p + 1e-12)).sum()
        return float(1.0 - h / math.log(max(len(p), 2)))


def leave_one_out_importance(frames: Sequence[FrameRef],
                             judge: Callable[[Sequence[FrameRef]], str],
                             ) -> ImportanceResult:
    """留一消融:丢掉每一帧,看判决变不变。

    **这就是"标注"** —— 但它来自模型自己,不需要人看任何一帧。
    代价是 N+1 次前向。

    ⚠️ 这是**离线**方法。在线时你不知道未来,所以它的用途是
    **事后回标**:离线算出真值重要性,拿来训一个在线策略。
    """
    base = judge(frames)
    imp = np.zeros(len(frames))
    for i in range(len(frames)):
        sub = [f for j, f in enumerate(frames) if j != i]
        imp[i] = 0.0 if judge(sub) == base else 1.0
    return ImportanceResult(imp, base, len(frames) + 1)


def hindsight_targets(frames: Sequence[FrameRef],
                      judge: Callable[[Sequence[FrameRef]], str],
                      keep: int) -> np.ndarray:
    """事后回标:哪些帧**当时**应该被保留。

    离线重放整条流、掌握全部未来后算出的答案,用作在线策略的训练目标。
    全程零人工标注。
    """
    imp = leave_one_out_importance(frames, judge).per_frame
    y = np.zeros(len(frames))
    y[np.argsort(-imp)[:keep]] = 1.0
    y[len(frames) - 1] = 1.0            # 最新帧总该留
    return y


# ---------------------------------------------------------------- 评测


def agreement_at_budget(frames: Sequence[FrameRef],
                        judge: Callable[[Sequence[FrameRef]], str],
                        keep: int, policy: str = "hybrid") -> dict:
    """在给定预算下,某策略的判决与全上下文基线是否一致。

    这是驱逐策略**唯一该看的指标**:省了多少不重要,
    重要的是省完之后判决还对不对。
    """
    base = judge(frames)
    idx = POLICIES[policy](frames, keep)
    kept = [frames[i] for i in idx]
    return {"policy": policy, "keep": keep, "n": len(frames),
            "baseline": base, "after": judge(kept),
            "agree": judge(kept) == base}
