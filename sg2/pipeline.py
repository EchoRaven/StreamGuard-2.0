"""端到端装配:ring buffer -> sentinel -> CUSUM -> 中间层 -> ACI。

本模块只做**编排**,不含模型细节 —— 组件从 registry 按配置名构造。
编排逻辑正是最该脱离 GPU 测试的部分:它出错时系统照跑,只是悄悄少看了
帧、事件永远不闭合、或者在线校准从没被喂过反馈。

⚠️ 本文件是那次"接线审计"的产物。早先 `aci` / `buffer` / `outofpolicy`
三个模块都实现了、测试全绿,**但主流程一次都没调用它们** ——
机制写对了却没接线,是这个项目最贵的一类返工。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .aci import AdaptiveConformal
from .buffer import RingBuffer
from .config import SG2Config
from .cusum import CusumController
from .models import build_midtier, build_sentinel
from .models.base import StreamStep
from .outofpolicy import PolicyGapTracker, UncoveredCase
from .policy import PolicyCorpus
from .realtime import RealtimeBudget
from .router import CompetenceRouter, Route


@dataclass
class Tick:
    """一次采样的完整记录。Tick 序列即轨迹,可直接喂奖励函数。"""
    t_s: float
    sentinel_score: float
    cusum_stat: float
    alarmed: bool
    escalated: bool
    route: str = ""
    ood: float = 0.0
    voi: float = 0.0
    forced_by_coverage: bool = False
    audited: bool = False
    step: StreamStep | None = None
    rewound_frames: int = 0


@dataclass
class StreamResult:
    ticks: list[Tick] = field(default_factory=list)
    alarms: int = 0
    escalations: int = 0
    coverage_forced: int = 0
    audited: int = 0
    events_closed: int = 0
    uncovered_cases: int = 0
    rewinds: int = 0
    routes: dict = field(default_factory=dict)
    realtime: dict = field(default_factory=dict)
    buffer_stats: dict = field(default_factory=dict)
    aci_stats: dict = field(default_factory=dict)

    @property
    def first_flag_t_s(self) -> float | None:
        for t in self.ticks:
            if t.step is not None and t.step.action == "flag":
                return t.t_s
        return None

    def detection_delay(self, nu_s: float | None) -> float | None:
        """E[(τ−ν)⁺] 的单次实现。未检出返回 None。"""
        if nu_s is None:
            return None
        tau = self.first_flag_t_s
        return None if tau is None else max(0.0, tau - nu_s)

    @property
    def escalation_rate(self) -> float:
        return self.escalations / len(self.ticks) if self.ticks else 0.0


class Pipeline:
    """把各层串起来。

    ⚠️ 保底随机覆盖 `rho` 是**旁路**:独立于 sentinel 分数直接送中间层。
    这是优雅退化定理(docs/06 §6.1)成立的机制 —— sentinel 对某类完全
    无判别力时,召回不会掉到 0。

    ⚠️ 审计采样 `audit_rate` 与 `rho` 是**两件事**:rho 决定送不送中间层,
    audit 决定这条样本的真值会不会回流给 ACI。漏报按定义不进审核队列,
    只有随机审计能给 ACI 无偏信号(docs/04 §4.4)。
    """

    def __init__(self, cfg: SG2Config | None = None, *,
                 sentinel=None, midtier=None, cusum=None,
                 corpus: PolicyCorpus | None = None,
                 buffer: RingBuffer | None = None,
                 aci: AdaptiveConformal | None = None,
                 router: CompetenceRouter | None = None):
        self.cfg = cfg or SG2Config()
        self.sentinel = sentinel or build_sentinel(self.cfg.sentinel)
        self.midtier = midtier or build_midtier(self.cfg.midtier)
        self.cusum = cusum or CusumController(cfg=self.cfg.cusum,
                                              seed=self.cfg.runtime.seed)
        self.corpus = corpus or self.cfg.midtier.policy.build_corpus()
        self.buffer = buffer or RingBuffer(
            window_s=self.cfg.runtime.ring_buffer_s)
        self.aci = aci or AdaptiveConformal(
            alpha=1.0 - self.cfg.calibration.target_recall,
            gamma=self.cfg.calibration.aci_gamma)
        # ⚠️ 路由按**能力**不按风险:"低级处理不了就上送"。
        # CUSUM 回答"什么时候变了",router 回答"这一条我自己能不能定"。
        # 二者互补 —— 纯风险路由永远抓不到"分数低是因为看不懂"那一格。
        self.router = router or CompetenceRouter()
        if router is None:
            # ⚠️ tau 必须来自**校准**,不能用默认 0。实测:随机编码器的
            # 分数全在 0 附近,tau=0 会让每一帧都"近阈值"从而全部上送,
            # 升级率 100% —— 级联完全失效,而且不报任何错。
            self._tau_calibrated = False
        else:
            self._tau_calibrated = True
        self.gaps = PolicyGapTracker(self.corpus)
        self._rng = random.Random(self.cfg.runtime.seed)

    def calibrate_router(self, safe_scores) -> float:
        """用一批**已知安全**的分数标定 router 的 tau。

        取安全分数的高分位作为阈值:高于它才算可疑。
        没做这一步就跑,升级率会失控。
        """
        import numpy as _np
        a = _np.asarray(list(safe_scores), dtype=float)
        if a.size < 10:
            raise ValueError(f"至少需要 10 个安全样本来标定 tau,收到 {a.size}")
        self.router.tau = float(_np.quantile(a, 0.95))
        self.router.margin = max(float(a.std()), 1e-3)
        self._tau_calibrated = True
        return self.router.tau
        self.gaps = PolicyGapTracker(self.corpus)
        self._rng = random.Random(self.cfg.runtime.seed)

    # ---------- 政策 ----------

    def _policy_text(self, event_idx: int) -> str:
        """按配置的粒度决定要不要换条款顺序。

        换顺序缓解位置偏置,但会让 KV 前缀缓存失效 —— 所以默认按**事件**
        换而不是按 tick 换(docs/09 §6)。
        """
        scope = self.cfg.midtier.policy.shuffle_scope
        seed = {"never": None, "event": event_idx,
                "tick": self._rng.randrange(1 << 30)}[scope]
        return self.corpus.render(shuffle_seed=seed)

    # ---------- 主循环 ----------

    def run(self, frames: list[tuple[float, np.ndarray]], *,
            nu_s: float | None = None,
            codec: np.ndarray | None = None) -> StreamResult:
        """跑一条流。frames 是 (时间戳, 帧) 的**因果**序列。

        Args:
            nu_s: 真值变点。仅用于生成 ACI 的审计反馈 —— **不影响判决**。
        """
        if not self._tau_calibrated:
            import warnings
            warnings.warn(
                "router.tau 未标定(仍为默认值)。分数分布不以 0 为中心时,"
                "每一帧都会被判为'近阈值'从而全部上送,升级率趋近 100%。"
                "请先调用 calibrate_router(safe_scores)。", RuntimeWarning,
                stacklevel=2)
        res = StreamResult()
        event_idx = 0
        self.midtier.set_policy(self._policy_text(event_idx))
        event_open = False

        for i, (t_s, frame) in enumerate(frames):
            self.buffer.append(t_s, frame)

            out = self.sentinel.score_frame(
                t_s, frame=frame,
                codec=codec[i:i + 1] if codec is not None else None)
            alarmed = self.cusum.update(out.score, t_s)

            forced = self._rng.random() < self.cfg.coverage.rho
            rd = self.router.route(out.score, ood=out.ood,
                                   forced=forced, alarmed=alarmed)
            escalate = rd.escalated

            step, n_rewound = None, 0
            if escalate:
                # 告警时回溯 ring buffer 突发采样;保底覆盖只看当前帧。
                if alarmed:
                    win = self.buffer.resample(
                        max(0.0, t_s - self.cfg.cusum.rewind_s), t_s,
                        self.cfg.cusum.burst_fps)
                    for f in win:
                        self.midtier.ingest(f.data, f.t_s)
                    n_rewound = len(win)
                    res.rewinds += 1
                else:
                    self.midtier.ingest(frame, t_s)

                step = self.midtier.step()
                res.escalations += 1

                if step.action == "flag":
                    event_open = True
                elif step.action == "uncovered":
                    # 缺口只记录,**不改政策** —— 改政策要人工批准
                    if self.gaps.record(UncoveredCase(
                            clip_id=f"t{t_s:.2f}", t_s=t_s,
                            description=step.description or "",
                            suggested_category=step.suggested_category)):
                        res.uncovered_cases += 1
                elif step.action == "clear" and event_open:
                    self.midtier.close_event()
                    event_open = False
                    event_idx += 1
                    res.events_closed += 1
                    self.midtier.set_policy(self._policy_text(event_idx))

            if alarmed:
                res.alarms += 1
                self.cusum.reset()
            if forced:
                res.coverage_forced += 1

            # ---- ACI:只有被审计到的 tick 才有真值反馈 ----
            audited = self._rng.random() < self.cfg.coverage.audit_rate
            err = None
            if audited and nu_s is not None:
                # 漏报 = 内容已出现但本 tick 没 flag
                err = (t_s >= nu_s
                       and not (step is not None and step.action == "flag"))
                res.audited += 1
            self.aci.update(err)

            res.ticks.append(Tick(
                t_s=t_s, sentinel_score=out.score,
                cusum_stat=self.cusum.statistic, alarmed=alarmed,
                escalated=escalate, route=rd.route.value, ood=out.ood,
                voi=rd.voi, forced_by_coverage=forced,
                audited=audited, step=step, rewound_frames=n_rewound))

        res.buffer_stats = self.buffer.stats
        res.aci_stats = self.aci.stats
        res.routes = self.router.stats
        res.realtime = self.realtime_check(res.escalation_rate)
        return res

    def realtime_check(self, escalation_rate: float) -> dict:
        """实时可行性。**与成本是两条独立约束** ——
        钱能买更多卡,但单条流的每一级必须在截止期内完成,否则积压,
        而排队延迟要计入 E[τ−ν]。"""
        b = RealtimeBudget(sentinel_fps=self.cfg.sentinel.sample_fps,
                           escalation_rate=max(escalation_rate, 1e-6))
        return {"feasible": b.feasible, "bottleneck": b.bottleneck().name,
                "max_streams_per_gpu": b.max_streams_per_gpu(),
                "added_delay_ms": round(b.total_added_delay_s() * 1000, 1),
                "tiers": {t.name: {"headroom": round(t.headroom, 1),
                                   "util": round(t.utilization, 4)}
                          for t in b.tiers()}}

    def reset(self) -> None:
        self.sentinel.reset()
        self.midtier.reset()
        self.cusum.reset()
        self.buffer.clear()
        self._rng = random.Random(self.cfg.runtime.seed)
