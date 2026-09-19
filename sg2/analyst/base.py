"""Analyst backend 契约。

换 backbone 只改配置,不改调用方 —— "换模型免费变强"是可测主张(见
docs/06_EXPERIMENTS.md §3.5),前提是接口层真的统一。

拒绝(refusal)是一等返回值,不是异常里的字符串:每个 backbone 的拒绝率要
作为发现被报告,不能被静默吞掉或重试掩盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class RefusalError(RuntimeError):
    """backbone 的内置安全过滤拒绝了本次请求。

    不重试、不改写提示去绕过。记录下来,作为该 backbone 的拒绝率数据点。
    """
    def __init__(self, backend: str, reason: str = ""):
        super().__init__(f"{backend} 拒绝: {reason}")
        self.backend, self.reason = backend, reason


@dataclass
class AnalystRequest:
    """一次升级窗口的分析请求。

    perception_only=True 时只问中性感知问题(描述可见内容、核验属性),
    判断留给下一步对照政策条款 —— 见 docs/05_RUNTIME.md §5.4。
    """
    clip_id: str
    frames: list                      # PIL.Image 或路径
    t_start_s: float
    t_end_s: float
    policy_clauses: list[str] = field(default_factory=list)
    exemplars: list = field(default_factory=list)   # 必须来自 pool="exemplar"
    perception_only: bool = True
    max_tokens: int = 40_000


@dataclass
class AnalystResponse:
    verdict: str                      # "safe" | "unsafe" | "uncertain"
    category: str | None
    confidence: float | None
    evidence_frames: list[int]
    policy_citations: list[str]
    raw_text: str
    tokens_used: int
    latency_ms: int
    refused: bool = False

    def __post_init__(self):
        if self.verdict not in ("safe", "unsafe", "uncertain"):
            raise ValueError(f"非法 verdict: {self.verdict}")
        if self.verdict == "unsafe" and not self.policy_citations:
            raise ValueError(
                f"{self.verdict} 判决必须带政策条款引用 —— 见提案 §5 强制引用")


@runtime_checkable
class AnalystBackend(Protocol):
    """所有 analyst 后端实现此协议。

    实现类:OpenAIAnalyst / AnthropicAnalyst / VLLMAnalyst。
    模型 id 走配置,不写死在代码里。
    """
    name: str

    def analyze(self, req: AnalystRequest) -> AnalystResponse:
        """分析一个窗口。被内置过滤拒绝时抛 RefusalError。"""
        ...

    def cost_per_1k_tokens(self) -> tuple[float, float]:
        """(输入, 输出) 单价,美元。用于 06_EXPERIMENTS 的成本核算。"""
        ...
