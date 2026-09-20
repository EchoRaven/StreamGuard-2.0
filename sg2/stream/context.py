"""流式 KV 分段管理器。

刻意不依赖任何模型:驱逐策略、段边界、事件闭合这些逻辑出错很隐蔽
(模型照跑,只是注意力悄悄坏掉),必须能在没有 GPU 的情况下穷举测试。

分段结构见 docs/07_STREAMING_LLM.md §2:

    [S0 sink | S1 政策头 | S2 事件上下文 | S3 视觉滑窗]
     永不驱逐  按政策版本   事件闭合时截断   FIFO

⚠️ S0 attention sink 必须保留。StreamingLLM 的发现:注意力大量汇聚到序列
最前端的几个 token,驱逐它们会让注意力分布崩坏 —— 即使这些 token 语义上
毫无意义。这是最容易漏掉的一条。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Segment(str, Enum):
    SINK = "sink"
    POLICY = "policy"
    EVENT = "event"
    VISION = "vision"


@dataclass(frozen=True)
class Block:
    """KV 里的一段连续 token。tokens 是长度,不是内容。"""
    seg: Segment
    tokens: int
    t_s: float = 0.0
    tag: str = ""


class SinkEvicted(RuntimeError):
    """试图驱逐 attention sink。这会让注意力分布崩坏。"""


@dataclass
class StreamContext:
    """按段管理 KV 占用。

    本类只跟踪**token 预算与顺序**,不持有真实 KV 张量 —— 后端在
    `sg2/stream/runner.py` 里按本类给出的驱逐指令操作真实缓存。
    """
    sink_tokens: int = 4
    max_policy_tokens: int = 2048
    max_event_tokens: int = 512
    max_vision_tokens: int = 4096
    vision_window_s: float = 16.0

    _policy: list[Block] = field(default_factory=list, repr=False)
    _event: deque[Block] = field(default_factory=deque, repr=False)
    _vision: deque[Block] = field(default_factory=deque, repr=False)
    _importance: deque[float] = field(default_factory=deque, repr=False)
    _policy_key: str | None = None
    _event_open: bool = False
    _evictions: int = 0
    _event_truncations: int = 0

    # ---------- S1 政策头 ----------

    def set_policy(self, tokens: int, key: str) -> bool:
        """装载政策头。key 应是政策语料的 sha256。

        Returns:
            True 表示缓存命中(未重编码),False 表示需要重新编码。
        """
        if self._policy_key == key:
            return True
        if tokens > self.max_policy_tokens:
            raise ValueError(
                f"政策头 {tokens} token 超出上限 {self.max_policy_tokens}")
        self._policy = [Block(Segment.POLICY, tokens, tag=key)]
        self._policy_key = key
        return False

    # ---------- S3 视觉滑窗 ----------

    def append_frame(self, tokens: int, t_s: float,
                     importance: float | None = None) -> list[Block]:
        """追加一帧的视觉 token,返回被驱逐的块。

        双重约束:token 上限与时间窗,任一超出即驱逐。

        ⚠️ `importance` 给定时按**重要性**驱逐而非纯 FIFO。
        FIFO 对流式检测特别糟:实测在预算 3-8 帧下,FIFO 与等间隔采样
        **全部丢掉 needle 导致判决翻转**,而覆盖/显著性策略全部保住
        (见 sg2/evict.py)。needle 常常就在刚过去那段,而 FIFO 丢的
        恰恰是最早进来的。

        重要性可来自压缩域(关键帧/运动能量,免费)或 sentinel 分数,
        **不需要人工标注**。
        """
        if tokens > self.max_vision_tokens:
            raise ValueError(f"单帧 {tokens} token 超出视觉窗上限")
        self._vision.append(Block(Segment.VISION, tokens, t_s=t_s,
                                  tag=f"imp={importance:.3f}"
                                  if importance is not None else ""))
        if importance is not None:
            self._importance.append(importance)
        evicted: list[Block] = []

        # 时间窗:只能按时间驱逐,与重要性无关(超窗的帧本来就该走)
        while self._vision and t_s - self._vision[0].t_s > self.vision_window_s:
            evicted.append(self._pop_oldest())

        # token 预算:有重要性信息时丢**最不重要**的,否则退回 FIFO
        while self._vision and self.vision_tokens > self.max_vision_tokens:
            if self._importance and len(self._importance) == len(self._vision):
                i = min(range(len(self._importance) - 1),
                        key=lambda k: self._importance[k]) \
                    if len(self._vision) > 1 else 0
                evicted.append(self._pop_at(i))
            else:
                evicted.append(self._pop_oldest())
        return evicted

    def _pop_oldest(self) -> Block:
        b = self._vision.popleft()
        if self._importance:
            self._importance.popleft()
        self._evictions += 1
        return b

    def _pop_at(self, i: int) -> Block:
        """丢掉第 i 个。**最新帧永不丢** —— 它正是当前要判的那帧。"""
        items = list(self._vision)
        imps = list(self._importance)
        b = items.pop(i)
        if imps:
            imps.pop(i)
        self._vision.clear(); self._vision.extend(items)
        self._importance.clear(); self._importance.extend(imps)
        self._evictions += 1
        return b

    # ---------- S2 事件上下文 ----------

    def open_event(self) -> None:
        self._event_open = True

    def append_event(self, tokens: int, tag: str = "") -> list[Block]:
        """向事件上下文追加证据。超长时驱逐最旧的证据,不动 sink/政策。"""
        if not self._event_open:
            self.open_event()
        self._event.append(Block(Segment.EVENT, tokens, tag=tag))
        evicted = []
        while self._event and self.event_tokens > self.max_event_tokens:
            evicted.append(self._event.popleft())
            self._evictions += 1
        return evicted

    def close_event(self) -> int:
        """事件闭合 = 截断 S2。这是 1.0 的 context reset 在缓存层的对应物。

        Returns:
            释放的 token 数。
        """
        freed = self.event_tokens
        self._event.clear()
        self._event_open = False
        self._event_truncations += 1
        return freed

    # ---------- 不变量 ----------

    def evict_to(self, budget: int) -> list[Block]:
        """把总占用压到 budget 以内。

        驱逐顺序:视觉窗 -> 事件上下文。**sink 与政策头永不驱逐** ——
        压不下去就抛异常,而不是悄悄动它们。
        """
        floor = self.sink_tokens + self.policy_tokens
        if budget < floor:
            raise SinkEvicted(
                f"预算 {budget} 低于 sink+政策头 {floor};"
                f"驱逐它们会让注意力分布崩坏,应改小政策头或换更大上下文")
        evicted = []
        while self.total_tokens > budget and self._vision:
            evicted.append(self._vision.popleft())
            self._evictions += 1
        while self.total_tokens > budget and self._event:
            evicted.append(self._event.popleft())
            self._evictions += 1
        return evicted

    # ---------- 观测 ----------

    @property
    def policy_tokens(self) -> int:
        return sum(b.tokens for b in self._policy)

    @property
    def event_tokens(self) -> int:
        return sum(b.tokens for b in self._event)

    @property
    def vision_tokens(self) -> int:
        return sum(b.tokens for b in self._vision)

    @property
    def total_tokens(self) -> int:
        return (self.sink_tokens + self.policy_tokens
                + self.event_tokens + self.vision_tokens)

    @property
    def event_open(self) -> bool:
        return self._event_open

    @property
    def n_frames(self) -> int:
        return len(self._vision)

    @property
    def stats(self) -> dict:
        return {"total": self.total_tokens, "sink": self.sink_tokens,
                "policy": self.policy_tokens, "event": self.event_tokens,
                "vision": self.vision_tokens, "frames": self.n_frames,
                "evictions": self._evictions,
                "event_truncations": self._event_truncations}
