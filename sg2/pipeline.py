"""端到端装配:sentinel -> CUSUM -> 中间层 -> (analyst)。

本模块只做**编排**,不含任何模型细节 —— 所有组件从 registry 按配置名构造。
编排逻辑正是最该被脱离 GPU 测试的部分:它出错时系统照跑,只是悄悄
少看了帧、或者事件永远不闭合。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .config import SG2Config
from .cusum import CusumController
from .models import build_midtier, build_sentinel
from .models.base import StreamStep


@dataclass
class Tick:
    """一次采样的完整记录。轨迹由 Tick 序列构成,直接喂给奖励函数。"""
    t_s: float
    sentinel_score: float
    cusum_stat: float
    alarmed: bool
    escalated: bool
    forced_by_coverage: bool = False
    step: StreamStep | None = None


@dataclass
class StreamResult:
    ticks: list[Tick] = field(default_factory=list)
    alarms: int = 0
    escalations: int = 0
    coverage_forced: int = 0
    events_closed: int = 0

    @property
    def first_flag_t_s(self) -> float | None:
        for t in self.ticks:
            if t.step is not None and t.step.action == "flag":
                return t.t_s
        return None

    def detection_delay(self, nu_s: float | None) -> float | None:
        """E[(τ-ν)⁺] 的单次实现。未检出返回 None。"""
        if nu_s is None:
            return None
        tau = self.first_flag_t_s
        return None if tau is None else max(0.0, tau - nu_s)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / len(self.ticks) if self.ticks else 0.0


class Pipeline:
    """把各层串起来。

    ⚠️ 保底随机覆盖 `rho` 是**旁路**:它独立于 sentinel 分数,直接把流量
    送进中间层。这是优雅退化定理(docs/06 §6.1)成立的机制 —— sentinel
    对某类完全无判别力时,召回不会掉到 0。
    """

    def __init__(self, cfg: SG2Config | None = None, *,
                 sentinel=None, midtier=None, cusum=None):
        self.cfg = cfg or SG2Config()
        self.sentinel = sentinel or build_sentinel(self.cfg.sentinel)
        self.midtier = midtier or build_midtier(self.cfg.midtier)
        self.cusum = cusum or CusumController(cfg=self.cfg.cusum,
                                              seed=self.cfg.runtime.seed)
        self._rng = random.Random(self.cfg.runtime.seed)

    def run(self, frames: list[tuple[float, np.ndarray]], *,
            policy_text: str = "政策: 见清单。", policy_key: str = "v1",
            codec: np.ndarray | None = None) -> StreamResult:
        """跑一条流。frames 是 (时间戳, 帧) 的**因果**序列。"""
        res = StreamResult()
        self.midtier.set_policy(policy_text, policy_key)
        event_open = False

        for i, (t_s, frame) in enumerate(frames):
            out = self.sentinel.score_frame(
                t_s, frame=frame,
                codec=codec[i:i + 1] if codec is not None else None)
            alarmed = self.cusum.update(out.score, t_s)

            # 旁路:与分数无关的保底覆盖
            forced = self._rng.random() < self.cfg.coverage.rho
            escalate = alarmed or forced

            step = None
            if escalate:
                self.midtier.ingest(frame, t_s)
                step = self.midtier.step()
                res.escalations += 1
                if step.action == "flag":
                    event_open = True
                elif step.action == "clear" and event_open:
                    self.midtier.close_event()
                    event_open = False
                    res.events_closed += 1
            if alarmed:
                res.alarms += 1
                self.cusum.reset()
            if forced:
                res.coverage_forced += 1

            res.ticks.append(Tick(
                t_s=t_s, sentinel_score=out.score,
                cusum_stat=self.cusum.statistic, alarmed=alarmed,
                escalated=escalate, forced_by_coverage=forced, step=step))
        return res

    def reset(self) -> None:
        self.sentinel.reset()
        self.midtier.reset()
        self.cusum.reset()
        self._rng = random.Random(self.cfg.runtime.seed)
