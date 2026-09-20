"""纯数据契约。

刻意**不依赖任何东西** —— 这是为了打破循环导入:
`protocol` 需要 StreamStep,而它原先放在 `models/base.py` 里,
导入它会执行 `models/__init__` -> `streaming` -> 回头 import `protocol`。

数据契约不该拉起整个模型包。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SentinelOutput:
    """always-on 层的输出。

    ⚠️ `score` 必须是**连续**量。小 VLM 吐 safe/unsafe 是 1-bit 信号,
    量化太狠,做 CUSUM 输入很糟(docs/01_MODELS.md §1.1)。
    """
    t_s: float
    score: float
    channels: dict[str, float] = field(default_factory=dict)
    decoded: bool = True


@dataclass
class StreamStep:
    """流式 VLM 在一个 tick 的输出。"""
    action: str                           # hold | flag | clear | uncovered
    raw: str
    category: str | None = None
    policy_citation: str | None = None
    evidence_frames: tuple[int, ...] = ()
    confidence: float | None = None
    tokens: int = 0
    description: str | None = None        # uncovered 时描述看到了什么
    suggested_category: str | None = None  # uncovered 时建议的新类别

    @property
    def is_valid(self) -> bool:
        if self.action not in ("hold", "flag", "clear", "uncovered"):
            return False
        # uncovered 不要求引用 —— 正因为没有条款可引才会走这条
        return not (self.action == "flag" and not self.policy_citation)
