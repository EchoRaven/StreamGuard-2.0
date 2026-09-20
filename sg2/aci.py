"""Adaptive Conformal Inference:漂移下的长程覆盖保证。

    alpha_{t+1} = alpha_t + gamma * (err_t - alpha)

其中 err_t = 1{本次漏报}。漏报时 alpha_t 上调 -> 下游阈值放宽(更容易
报警);长期不漏报则下调。alpha_t 可读作"系统当前认为自己需要多大的
错误预算"。

覆盖界对**任意序列**成立,不需要任何分布假设 —— 包括任意漂移与对抗性
漂移。这正是 docs/03_ADAPTATION.md §2.5 要的那一层:离线 conformal 给
每个 epoch 的操作点,ACI 给 epoch 之间的漂移。

**两个必须说清楚的代价:**

1. 保证是**长程平均**覆盖,不是逐时刻。
2. `err_t = 1{miss}` 是**反馈流看不见的量** —— 审核员只看被标出来的,
   漏报按定义不会进审核队列。必须靠随机审计采样提供无偏信号,
   价格是 `1/(eps*alpha)` 条未升级内容换 1 个信号(docs/04 §4.4)。

⚠️ ACI 只保 coverage,**不保 efficiency**。context 一改分数分布就平移,
阈值会震荡 -> 升级率震荡 -> 成本震荡,而成本是 Pareto 图的另一根轴。
所以要按**纪元**节流(docs/03 §4)。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ACIState:
    alpha_t: float
    t: int = 0
    n_err: int = 0
    n_fed: int = 0                    # 真正拿到反馈的次数(不含无反馈的 tick)


@dataclass
class AdaptiveConformal:
    """单个 gamma 的 ACI。

    Args:
        alpha:  目标错误率(= 1 - 目标召回)
        gamma:  步长。大 -> 追漂移快但震荡;小 -> 稳但滞后
        clip:   alpha_t 的截断范围。不截断的话一段连续漏报能把它推到负数,
                之后阈值永远打不开。
    """
    alpha: float = 0.05
    # ⚠️ 默认 0.002 而非 0.02。实测:错误率在目标附近 3%<->8% 漂移、
    # 跑 3200 步时,gamma=0.02 让 alpha_t 摆满 [0.001,0.5] 全量程,
    # gamma=0.002 才稳在 [0.027,0.091] 且零饱和。
    # 经验法则:gamma 约取 1/(一个漂移周期内的反馈条数)。
    gamma: float = 0.002
    clip: tuple[float, float] = (0.001, 0.5)

    _st: ACIState = field(default=None, repr=False)  # type: ignore
    _n_clipped: int = field(default=0, repr=False)

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha 必须在 (0,1): {self.alpha}")
        if self.gamma <= 0:
            raise ValueError(f"gamma 必须为正: {self.gamma}")
        lo, hi = self.clip
        if not 0 < lo < hi < 1:
            raise ValueError(f"clip 非法: {self.clip}")
        self._st = ACIState(alpha_t=self.alpha)

    def update(self, err: bool | None) -> float:
        """喂一次反馈,返回新的 alpha_t。

        Args:
            err: True=漏报, False=覆盖到了, **None=本 tick 没有反馈**。

        ⚠️ `None` 不更新 alpha_t。把"没反馈"当成"没漏报"会让 alpha_t 单调
        下降 —— 阈值越收越紧,而这看起来完全正常。审计采样之外的绝大多数
        tick 都是 None。
        """
        self._st.t += 1
        if err is None:
            return self._st.alpha_t
        self._st.n_fed += 1
        self._st.n_err += int(err)
        lo, hi = self.clip
        # 符号:err=1(漏报)时要**收紧** -> alpha_t 下调;err=0 时放松。
        # 写成 +gamma*(alpha-err) 是覆盖率语义下的形式,这里 err 是漏报
        # 指示,必须取反 —— 实测写错方向时,错误率低于目标反而把 alpha_t
        # 推到上限,高于目标反而推到下限,且过程中不报任何错。
        raw = self._st.alpha_t + self.gamma * (float(err) - self.alpha)
        self._st.alpha_t = min(hi, max(lo, raw))
        if raw != self._st.alpha_t:
            self._n_clipped += 1
        return self._st.alpha_t

    def warm_start(self, alpha_t: float) -> None:
        """新纪元开始时用离线校准的结果热启动。"""
        lo, hi = self.clip
        self._st.alpha_t = min(hi, max(lo, alpha_t))

    # ---------- 观测 ----------

    @property
    def alpha_t(self) -> float:
        return self._st.alpha_t

    @property
    def empirical_error(self) -> float:
        """已观测到的经验错误率。长程应收敛到 alpha。"""
        return self._st.n_err / self._st.n_fed if self._st.n_fed else 0.0

    @property
    def coverage_gap(self) -> float:
        """|经验错误率 - 目标|。ACI 的理论保证就是它随 T 趋于 0。"""
        return abs(self.empirical_error - self.alpha)

    def coverage_bound(self) -> float:
        """Gibbs & Candes 的有限样本界:|经验err - alpha| <= (a_1 + 1/gamma)/T。

        对**任意序列**成立。T 小的时候这个界很松,别拿它当强保证。
        """
        T = max(self._st.n_fed, 1)
        return (self.alpha + 1.0 / self.gamma) / T

    @property
    def saturation(self) -> float:
        """被截断的比例。

        ⚠️ 截断是**诊断信号不是安全网**。持续撞边界说明 gamma 与实际漂移
        速率不匹配:单向偏离一路累积到边界后,alpha_t 就再也不动了 ——
        自适应失效,而系统照跑不报错。实测 gamma=0.02 跑 3000 步、
        真实错误率恒定偏离目标时,饱和度会接近 1。

        >0.2 就该换 gamma,或改用 DtACI(它并行多个 gamma,不必预先猜对)。
        """
        return self._n_clipped / self._st.n_fed if self._st.n_fed else 0.0

    @property
    def is_saturated(self) -> bool:
        return self.saturation > 0.2

    @property
    def stats(self) -> dict:
        return {"t": self._st.t, "fed": self._st.n_fed,
                "saturation": round(self.saturation, 3),
                "alpha_t": round(self._st.alpha_t, 5),
                "empirical_err": round(self.empirical_error, 4),
                "target": self.alpha,
                "gap": round(self.coverage_gap, 4),
                "bound": round(self.coverage_bound(), 4)}


@dataclass
class DtACI:
    """多专家 ACI。

    单个 gamma 要在"追漂移快"和"稳"之间二选一,而**合适的 gamma 取决于
    漂移速率,而漂移速率事先不知道**。DtACI 并行跑多个 gamma,按近期表现
    指数加权 —— 不必预先猜对。
    """
    alpha: float = 0.05
    gammas: tuple[float, ...] = (0.005, 0.02, 0.08, 0.32)
    eta: float = 2.0                  # 专家权重的学习率
    window: int = 200                 # 只用近期表现评专家

    _experts: list[AdaptiveConformal] = field(default_factory=list, repr=False)
    _w: list[float] = field(default_factory=list, repr=False)
    _recent: list[deque] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.gammas:
            raise ValueError("至少要有一个 gamma")
        self._experts = [AdaptiveConformal(alpha=self.alpha, gamma=g)
                         for g in self.gammas]
        self._w = [1.0 / len(self.gammas)] * len(self.gammas)
        self._recent = [deque(maxlen=self.window) for _ in self.gammas]

    def update(self, err: bool | None) -> float:
        if err is None:
            for e in self._experts:
                e.update(None)
            return self.alpha_t

        for i, e in enumerate(self._experts):
            e.update(err)
            # pinball 损失:专家的 alpha_t 与实际错误的偏离
            self._recent[i].append(abs(e.alpha_t - (1.0 if err else 0.0)))

        losses = [sum(r) / len(r) if r else 0.0 for r in self._recent]
        m = min(losses)
        raw = [math.exp(-self.eta * (l - m)) for l in losses]
        z = sum(raw) or 1.0
        self._w = [r / z for r in raw]
        return self.alpha_t

    def warm_start(self, alpha_t: float) -> None:
        for e in self._experts:
            e.warm_start(alpha_t)

    @property
    def alpha_t(self) -> float:
        return sum(w * e.alpha_t for w, e in zip(self._w, self._experts))

    @property
    def empirical_error(self) -> float:
        fed = self._experts[0]._st.n_fed
        return self._experts[0]._st.n_err / fed if fed else 0.0

    @property
    def weights(self) -> dict[float, float]:
        return {g: round(w, 4) for g, w in zip(self.gammas, self._w)}

    @property
    def saturation(self) -> float:
        """加权饱和度。DtACI 的意义正是让权重从饱和的专家身上移开。"""
        return sum(w * e.saturation for w, e in zip(self._w, self._experts))

    @property
    def stats(self) -> dict:
        return {"alpha_t": round(self.alpha_t, 5),
                "empirical_err": round(self.empirical_error, 4),
                "target": self.alpha,
                "saturation": round(self.saturation, 3),
                "weights": self.weights}


def audit_signal_cost(audit_rate: float, miss_rate: float) -> float:
    """平均多少条未升级内容才换来 1 个漏报信号。

    这是 ACI 在线层的**价格**,应显式进成本核算而不是藏起来
    (docs/04 §4.4)。eps=1%, alpha=5% -> 2000 条换 1 个。
    """
    if not 0 < audit_rate <= 1 or not 0 < miss_rate <= 1:
        raise ValueError("audit_rate 与 miss_rate 必须在 (0,1]")
    return 1.0 / (audit_rate * miss_rate)


# ---------------------------------------------------------------- 可行性

def abstention_floor(mu: float, alpha: float, M: float = 1.0) -> float:
    """任何 distribution-free 方法都逃不掉的**弃权下界**。

        P(abstain) >= (mu - alpha) / (M - alpha)

    出处:Kotte 2026, arXiv:2606.29054, Proposition 3
    (Sharpened two-sided abstention floor)。
    mu = 基础风险 E[R], M = ess sup R。

    **在级联里"弃权"就是"升级"** —— 所以这是**升级率的闭式下界**:
    廉价层的基础漏报率 mu 高于目标 alpha 时,要想拿到保证,
    至少这么大比例的流量必须上送。**调参救不了不可行。**

    M<1 才比保守的 (mu-alpha)/(1-alpha) 更紧。原文实测 NER/QA/CLS 上
    审计出的 M 恰好 =1.0(每个格子里都有完全漏掉的样本),
    所以通常就用 M=1。我们的漏报损失是 0/1,同样 M=1。
    """
    if not 0 < alpha < M <= 1:
        raise ValueError(f"需要 0 < alpha({alpha}) < M({M}) <= 1")
    if not 0 <= mu <= 1:
        raise ValueError(f"mu 必须在 [0,1]: {mu}")
    return max(0.0, (mu - alpha) / (M - alpha))


def is_certifiable(mu: float, alpha: float, max_escalation: float,
                   M: float = 1.0) -> dict:
    """在给定升级预算下,目标 alpha 能不能被认证。

    ⚠️ 这是**部署前第一步**,先于选哪个界、用哪个分数。
    原文的三步配方:先查可行性,再选界与分数,最后在目标域重查。
    """
    floor = abstention_floor(mu, alpha, M)
    return {"mu": mu, "alpha": alpha, "floor": round(floor, 4),
            "budget": max_escalation,
            "feasible": floor <= max_escalation,
            "reason": ("基础风险低于目标,无需强制升级" if mu <= alpha
                       else (f"需升级 >= {floor:.1%},预算 {max_escalation:.1%}"
                             + ("(够)" if floor <= max_escalation
                                else " —— **不可行**"))),
            }


def certified_cost_conflict(mu: float, alpha: float, crossover_r: float,
                            M: float = 1.0) -> dict:
    """**认证与成本的冲突**:两个约束把升级率从两头夹。

        下界 = (mu - alpha)/(M - alpha)   要保证就至少升这么多
        上界 = r*(成本交叉点)             要比基线便宜就至多升这么多

    下界 > 上界时,**不可能同时做到"有保证"和"更便宜"** —— 必须放弃一个。
    这个张力可以闭式算出来,不需要跑任何实验。
    """
    floor = abstention_floor(mu, alpha, M)
    return {"floor": round(floor, 4), "crossover_r": round(crossover_r, 4),
            "window": round(crossover_r - floor, 4),
            "both_achievable": floor <= crossover_r,
            "verdict": ("可同时认证且更便宜" if floor <= crossover_r
                        else "**认证与省钱不可兼得** —— 要保证就得比基线贵")}


# ACI 的适用边界(Kotte 2026 §3.5 实测)
ACI_REGIMES = {
    "temporal_drift": {
        "desc": "流水线随时间漂移(我们的流内漂移)",
        "static_crc_violation": 0.60, "aci_violation": 0.04,
        "aci_helps": True},
    "gradual_degradation": {
        "desc": "质量逐渐劣化",
        "static_crc_violation": 0.56, "aci_violation": 0.12,
        "aci_helps": True},
    "cross_dataset": {
        "desc": "换数据集/换政策(我们的零日政策场景)",
        "static_crc_violation": 14 / 16, "aci_violation": 14 / 16,
        "aci_helps": False,
        "note": "mu=0.40-0.84 >> alpha=0.10,是**不可行**区间;"
                "合规的那几次弃权 75-100%。调参救不了不可行。"},
}


def aci_applicable(regime: str) -> bool:
    """ACI 在这个漂移类型下管不管用。

    ⚠️ **Proposition 8(emit-only feedback obstruction)**:
    只在"被放行的样本"上观测风险时,任意保证 anytime emitted-risk
    的方法都能被构造出违反。**ACI 正是 emit-only**,所以它的失败是
    反馈模型的性质,不是步长 gamma 调不好。

    正面保证需要 **full feedback** —— 验证器,或**被弃权样本上的标签**。
    在我们的系统里,那就是**随机审计采样**(docs/04 §4.4)。
    审计不是可选优化,它是保证成立的**前提**。
    """
    if regime not in ACI_REGIMES:
        raise ValueError(f"未知漂移类型 {regime};可选 {sorted(ACI_REGIMES)}")
    return ACI_REGIMES[regime]["aci_helps"]
