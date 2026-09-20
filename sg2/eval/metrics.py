"""评测指标:把 docs/06 的实验协议变成可执行的量。

三个主张各自对应一个函数:
  自适应 vs 均匀(§3.1)  -> pareto_curve
  优雅退化(§3.3)        -> graceful_degradation_check
  逐分层 ROC(§3.2)      -> stratified_roc

⚠️ **延迟与误报必须联合报告。** 分别报均值可以让"一个变差换另一个变好"
看起来像全面改善(docs/06 §5.4)。所以这里不提供单独的 mean_delay 接口,
只提供 (延迟, 误报) 的曲线。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StreamOutcome:
    """一条流的评测结果。"""
    stream_id: str
    nu_s: float | None           # 真值变点;None = 该流本身安全
    tau_s: float | None          # 首次告警时刻;None = 未检出
    n_false_alarms: int = 0
    cost: float = 0.0            # $ 或 token,由调用方定义
    stratum: str | None = None   # A / B / C
    category: str | None = None

    @property
    def detected(self) -> bool:
        return self.nu_s is not None and self.tau_s is not None \
            and self.tau_s >= self.nu_s

    @property
    def delay_s(self) -> float | None:
        return (self.tau_s - self.nu_s) if self.detected else None  # type: ignore

    @property
    def missed(self) -> bool:
        return self.nu_s is not None and not self.detected


@dataclass
class OperatingPoint:
    """一个操作点:(召回, 误报率, 延迟, 成本) 四元组。

    四个量绑在一起报,因为单独看任何一个都能被另外三个换来。
    """
    label: str
    recall: float
    fa_per_hour: float
    mean_delay_s: float | None
    p90_delay_s: float | None
    cost: float
    n: int = 0

    def __str__(self) -> str:
        d = f"{self.mean_delay_s:.1f}s" if self.mean_delay_s is not None else "—"
        p = f"{self.p90_delay_s:.1f}s" if self.p90_delay_s is not None else "—"
        return (f"{self.label:<18} 召回={self.recall:.3f} "
                f"误报/h={self.fa_per_hour:.2f} 延迟均值={d} p90={p} "
                f"成本={self.cost:.3f}")


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def operating_point(outcomes: list[StreamOutcome], *, label: str,
                    hours: float | None = None) -> OperatingPoint:
    """把一批流的结果压成一个操作点。"""
    unsafe = [o for o in outcomes if o.nu_s is not None]
    n_det = sum(1 for o in unsafe if o.detected)
    delays = [o.delay_s for o in unsafe if o.delay_s is not None]
    total_fa = sum(o.n_false_alarms for o in outcomes)
    h = hours if hours is not None else max(len(outcomes), 1) / 60.0
    return OperatingPoint(
        label=label,
        recall=n_det / len(unsafe) if unsafe else math.nan,
        fa_per_hour=total_fa / h if h else math.nan,
        mean_delay_s=(sum(delays) / len(delays)) if delays else None,
        p90_delay_s=_quantile(delays, 0.9) if delays else None,
        cost=sum(o.cost for o in outcomes) / max(len(outcomes), 1),
        n=len(outcomes))


def pareto_curve(points: list[OperatingPoint], *,
                 x: str = "cost", y: str = "recall") -> list[OperatingPoint]:
    """取帕累托前沿(x 越小越好、y 越大越好)。

    头图(docs/06 §3.1)是**两条曲线**:自适应分配 vs 均匀分配。
    ⚠️ 基线必须**也是曲线** —— "我们是曲线、基线是一个点"的对比会被判
    不公平。
    """
    pts = sorted(points, key=lambda p: (getattr(p, x), -getattr(p, y)))
    front, best = [], -math.inf
    for p in pts:
        v = getattr(p, y)
        if v > best:
            front.append(p)
            best = v
    return front


def dominates(a: OperatingPoint, b: OperatingPoint) -> bool:
    """a 在四个量上都不差于 b,且至少一个严格更好。"""
    da = a.mean_delay_s if a.mean_delay_s is not None else math.inf
    db = b.mean_delay_s if b.mean_delay_s is not None else math.inf
    ge = (a.recall >= b.recall and a.fa_per_hour <= b.fa_per_hour
          and da <= db and a.cost <= b.cost)
    gt = (a.recall > b.recall or a.fa_per_hour < b.fa_per_hour
          or da < db or a.cost < b.cost)
    return ge and gt


@dataclass
class DegradationResult:
    category: str
    observed_recall: float
    floor_recall: float
    rho: float
    n: int

    @property
    def holds(self) -> bool:
        """优雅退化定理是否成立(容一点抽样噪声)。"""
        return self.observed_recall >= self.floor_recall - self.tolerance

    @property
    def tolerance(self) -> float:
        """二项抽样的 1.96 sigma。n 小的时候这个容差很大,结论就弱。"""
        p = max(self.floor_recall, 1e-9)
        return 1.96 * math.sqrt(p * (1 - p) / max(self.n, 1))

    def __str__(self) -> str:
        mark = "✓" if self.holds else "✗ 穿透下界"
        return (f"{mark} {self.category:<18} 实测召回={self.observed_recall:.3f} "
                f"下界={self.floor_recall:.3f}±{self.tolerance:.3f} "
                f"(ρ={self.rho:.3f}, n={self.n})")


def graceful_degradation_check(outcomes: list[StreamOutcome], *, rho: float,
                               uniform_recall_at_rho: float | None = None,
                               category: str = "all") -> DegradationResult:
    """验证 docs/06 §6.1 的优雅退化:

        对任意类别,召回 >= 预算 ρ·B 下均匀分配的召回,与 sentinel 质量无关。

    `uniform_recall_at_rho` 是均匀分配在同预算下的实测召回;不给时用 ρ
    作为下界的保守近似(每条流独立以概率 ρ 被抽中)。

    ⚠️ **穿透下界 = 保底覆盖没有真正旁路**,不是"定理错了"。这是实现 bug
    的检测器,不是理论验证。
    """
    unsafe = [o for o in outcomes if o.nu_s is not None]
    n = len(unsafe)
    obs = sum(1 for o in unsafe if o.detected) / n if n else math.nan
    floor = uniform_recall_at_rho if uniform_recall_at_rho is not None else rho
    return DegradationResult(category=category, observed_recall=obs,
                             floor_recall=floor, rho=rho, n=n)


@dataclass
class StratumROC:
    stratum: str
    n_pos: int
    n_neg: int
    recall: float
    fpr: float
    escalation_rate: float

    def __str__(self) -> str:
        return (f"{self.stratum:<4} 正{self.n_pos:>4} 负{self.n_neg:>4}  "
                f"召回={self.recall:.3f} FPR={self.fpr:.3f} "
                f"升级率={self.escalation_rate:.3f}")


def stratified_roc(outcomes: list[StreamOutcome],
                   escalated: dict[str, bool] | None = None
                   ) -> list[StratumROC]:
    """按 A/B/C 分层统计。

    ⚠️ **不要看聚合数字。** 聚合会被占绝大多数的 A 类抬得很好看,把 B 类
    的问题完全盖住(docs/06 §3.2)。逐项性质不能用全局指标去测。
    """
    by: dict[str, list[StreamOutcome]] = {}
    for o in outcomes:
        by.setdefault(o.stratum or "?", []).append(o)
    out = []
    for st in sorted(by):
        grp = by[st]
        pos = [o for o in grp if o.nu_s is not None]
        neg = [o for o in grp if o.nu_s is None]
        esc = sum(1 for o in grp
                  if (escalated or {}).get(o.stream_id, o.tau_s is not None))
        out.append(StratumROC(
            stratum=st, n_pos=len(pos), n_neg=len(neg),
            recall=(sum(1 for o in pos if o.detected) / len(pos))
            if pos else math.nan,
            fpr=(sum(1 for o in neg if o.n_false_alarms > 0) / len(neg))
            if neg else math.nan,
            escalation_rate=esc / len(grp) if grp else math.nan))
    return out


def cost_crossover(c_uniform: float, c_sentinel: float, c_midtier: float,
                   c_frontier: float, *, q_frontier: float = 0.3,
                   c_audit: float = 0.0, audit_rate: float = 0.0) -> float:
    """解出自适应方案能赢的**升级率上界** r*(docs/06 实验 1)。

        C_2.0(r) = c_sentinel + r*(c_midtier + q*c_frontier) + eps*c_audit
        令 C_2.0(r*) = c_uniform

    ⚠️ `eps*c_audit` 是保证的价格,必须显式入账,不能藏起来。
    返回 <=0 表示**任何**升级率都赢不了 —— 前提就不成立。
    """
    per_esc = c_midtier + q_frontier * c_frontier
    if per_esc <= 0:
        raise ValueError("单次升级成本必须为正")
    budget = c_uniform - c_sentinel - audit_rate * c_audit
    return budget / per_esc
