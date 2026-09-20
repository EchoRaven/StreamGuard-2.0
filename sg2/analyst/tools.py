"""Analyst 的工具集。

⚠️ **只在已升级窗口内可用。** 全局分配是 CUSUM(统计,零成本,有界);
让 agent 决定全局去哪看等于把贵模型放回内层循环,且会破坏 §3 的延迟界
(docs/07 §0)。工具是"钱已经决定要花了,在这个窗口里怎么花"。

每个工具都带**预算计量**。agentic 是多轮的,不设上限时单次升级可以
无限膨胀 —— 而这段延迟要计入 E[τ−ν](docs/05 §5.3)。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..buffer import RingBuffer
from ..policy import PolicyCorpus


class BudgetExceeded(RuntimeError):
    """工具预算耗尽。返回当前最佳判断,不是无限重试。"""


@dataclass
class ToolBudget:
    max_calls: int = 10
    max_tokens: int = 40_000
    max_decoded_frames: int = 256

    _calls: int = 0
    _tokens: int = 0
    _frames: int = 0

    def charge(self, *, calls: int = 1, tokens: int = 0,
               frames: int = 0) -> None:
        self._calls += calls
        self._tokens += tokens
        self._frames += frames
        if self._calls > self.max_calls:
            raise BudgetExceeded(f"工具调用 {self._calls} > {self.max_calls}")
        if self._tokens > self.max_tokens:
            raise BudgetExceeded(f"token {self._tokens} > {self.max_tokens}")
        if self._frames > self.max_decoded_frames:
            raise BudgetExceeded(f"解码帧 {self._frames} > "
                                 f"{self.max_decoded_frames}")

    @property
    def spent(self) -> dict:
        return {"calls": self._calls, "tokens": self._tokens,
                "frames": self._frames}

    def reset(self) -> None:
        self._calls = self._tokens = self._frames = 0


@dataclass
class ToolResult:
    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    cost: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.ok:
            return f"{self.name}: 失败 — {self.error}"
        v = self.value
        if isinstance(v, np.ndarray):
            v = f"<array {v.shape}>"
        elif isinstance(v, list):
            v = f"<{len(v)} 项>"
        return f"{self.name}: {v}"


@dataclass
class ToolBox:
    """绑定到一次升级窗口的工具集。"""
    buffer: RingBuffer
    corpus: PolicyCorpus
    budget: ToolBudget = field(default_factory=ToolBudget)
    exemplar_lookup: Callable[[np.ndarray, int], list] | None = None
    ocr_fn: Callable[[np.ndarray], str] | None = None
    asr_fn: Callable[[float, float], str] | None = None
    _log: list[ToolResult] = field(default_factory=list, repr=False)

    # ---------- 视觉 ----------

    def zoom(self, t_s: float, bbox: tuple[float, float, float, float]
             ) -> ToolResult:
        """取某时刻的一块区域(相对坐标 0-1)。分辨率分级的"看仔细点"。"""
        return self._run("zoom", lambda: self._zoom(t_s, bbox), frames=1)

    def _zoom(self, t_s, bbox):
        if not self.buffer.covers(t_s):
            raise ValueError(
                f"t={t_s:.2f}s 已滚出 ring buffer(现存 "
                f"{self.buffer.span_s:.1f}s) —— 取不回来,是硬 miss 不是延迟")
        f = min(self.buffer, key=lambda x: abs(x.t_s - t_s))
        h, w = f.data.shape[:2]
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"bbox 必须是 0-1 的相对坐标且 x0<x1,y0<y1: {bbox}")
        return f.data[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    def rewind(self, t0_s: float, t1_s: float, fps: float = 4.0) -> ToolResult:
        """回溯区间并按目标帧率重采样。告警后的突发采样。"""
        def go():
            fr = self.buffer.resample(t0_s, t1_s, fps)
            if not fr:
                raise ValueError(
                    f"[{t0_s:.1f},{t1_s:.1f}] 不在 buffer 内(现存 "
                    f"{self.buffer.span_s:.1f}s)")
            return [f.data for f in fr]
        return self._run("rewind", go, frames=int((t1_s - t0_s) * fps) + 1)

    def diff(self, t0_s: float, t1_s: float) -> ToolResult:
        """两个时刻的变化摘要。比 rewind 便宜得多,先用它定位。"""
        def go():
            a = self.buffer.rewind(t0_s, t0_s + 0.05)
            b = self.buffer.rewind(t1_s, t1_s + 0.05)
            if not a or not b:
                raise ValueError("两端至少一端不在 buffer 内")
            d = np.abs(a[0].data.astype(float) - b[0].data.astype(float))
            return {"mean_abs_diff": round(float(d.mean()), 4),
                    "max_abs_diff": round(float(d.max()), 4),
                    "changed_frac": round(float((d.mean(axis=2) > 0.1).mean()), 4)}
        return self._run("diff", go, frames=2)

    # ---------- 文本 ----------

    def ocr(self, t_s: float) -> ToolResult:
        def go():
            if self.ocr_fn is None:
                raise NotImplementedError("未接 OCR 后端(rapidocr)")
            if not self.buffer.covers(t_s):
                raise ValueError(f"t={t_s:.2f}s 已滚出 buffer")
            f = min(self.buffer, key=lambda x: abs(x.t_s - t_s))
            return self.ocr_fn(f.data)
        return self._run("ocr", go, frames=1)

    def transcribe(self, t0_s: float, t1_s: float) -> ToolResult:
        def go():
            if self.asr_fn is None:
                raise NotImplementedError("未接 ASR 后端(faster-whisper)")
            return self.asr_fn(t0_s, t1_s)
        return self._run("transcribe", go)

    # ---------- 检索 ----------

    def policy_lookup(self, query: str = "", *, top_k: int = 5) -> ToolResult:
        """检索政策条款。空 query 返回全部生效条款。"""
        def go():
            cs = self.corpus.enforced()
            if query:
                q = query.lower()
                cs = [c for c in cs
                      if q in c.title.lower() or q in c.text.lower()
                      or q in c.id.lower()]
            return [{"id": c.id, "title": c.title, "text": c.text}
                    for c in cs[:top_k]]
        return self._run("policy_lookup", go, tokens=200)

    def case_lookup(self, emb: np.ndarray, top_k: int = 3) -> ToolResult:
        """检索相似案例。

        ⚠️ 只能来自 `pool="exemplar"`。取到校准池的样本会让 conformal
        保证失效,而且失效是静默的(docs/03 §5)。
        """
        def go():
            if self.exemplar_lookup is None:
                raise NotImplementedError("未接案例库")
            return self.exemplar_lookup(emb, top_k)
        return self._run("case_lookup", go, tokens=400)

    # ---------- 内部 ----------

    def _run(self, name: str, fn, *, tokens: int = 0,
             frames: int = 0) -> ToolResult:
        try:
            self.budget.charge(tokens=tokens, frames=frames)
        except BudgetExceeded as e:
            r = ToolResult(name, False, error=str(e), cost=self.budget.spent)
            self._log.append(r)
            return r
        try:
            v = fn()
            r = ToolResult(name, True, value=v, cost=self.budget.spent)
        except Exception as e:                        # noqa: BLE001
            r = ToolResult(name, False, error=f"{type(e).__name__}: {e}",
                           cost=self.budget.spent)
        self._log.append(r)
        return r

    @property
    def trace(self) -> list[ToolResult]:
        return list(self._log)

    @property
    def spent(self) -> dict:
        return self.budget.spent


TOOL_SPECS = [
    {"name": "zoom", "args": "t_s, bbox=(x0,y0,x1,y1) 相对坐标",
     "desc": "取某时刻某区域的全分辨率画面"},
    {"name": "rewind", "args": "t0_s, t1_s, fps",
     "desc": "回溯区间并按目标帧率重采样"},
    {"name": "diff", "args": "t0_s, t1_s",
     "desc": "两时刻的变化摘要,比 rewind 便宜,先用它定位"},
    {"name": "ocr", "args": "t_s", "desc": "读画面中的文字"},
    {"name": "transcribe", "args": "t0_s, t1_s", "desc": "转写语音"},
    {"name": "policy_lookup", "args": "query", "desc": "检索政策条款"},
    {"name": "case_lookup", "args": "embedding, top_k", "desc": "检索相似案例"},
]


def render_tool_specs() -> str:
    return "\n".join(f"- {t['name']}({t['args']}): {t['desc']}"
                     for t in TOOL_SPECS)
