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

    def append_frame(self, tokens: int, t_s: float) -> list[Block]:
        """追加一帧的视觉 token,返回被驱逐的块。

        双重约束:token 上限与时间窗,任一超出即驱逐。
        """
        if tokens > self.max_vision_tokens:
            raise ValueError(f"单帧 {tokens} token 超出视觉窗上限")
        self._vision.append(Block(Segment.VISION, tokens, t_s=t_s))
        evicted: list[Block] = []
        while self._vision and (
                self.vision_tokens > self.max_vision_tokens
                or t_s - self._vision[0].t_s > self.vision_window_s):
            evicted.append(self._vision.popleft())
            self._evictions += 1
        return evicted

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
