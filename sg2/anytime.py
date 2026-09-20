"""Anytime-valid 风险监控:跨域漂移下替代 ACI。

**为什么需要它。** Kotte 2026 (arXiv:2606.29054) Proposition 8:

> 只在**被放行的样本**上观测风险(emit-only feedback)时,任意保证
> anytime emitted-risk 控制的方法都能被构造出违反。**ACI 正是
> emit-only**,所以它的失败是反馈模型的性质,不是步长 γ 调不好。

正面保证需要 **full feedback** —— 验证器,或**被弃权样本上的标签**。
在我们的系统里那就是**随机审计采样**:审计不是可选优化,是前提。

原文实测:同样 16 组跨数据集迁移上,静态 CRC 与所有 γ 的 ACI 违反 14/16,
而 full-feedback 的 anytime-valid monitor **0/16**,且不平凡
(放行 65–97%)。

机制是**每阈值一条 test supermartingale** + 可预测的放行规则,
由 Ville 不等式给出任意时刻有效的保证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ThresholdState:
    """某个阈值上的检验上鞅。"""
    tau: float
    wealth: float = 1.0
    peak: float = 1.0
    n: int = 0
    n_risk: int = 0
    certified: bool = False        # H0 已被否定 -> 该阈值可放行
    recent: list = None            # 近期风险窗口

    def __post_init__(self):
        if self.recent is None:
            self.recent = []

    @property
    def empirical_risk(self) -> float:
        return self.n_risk / self.n if self.n else 0.0


@dataclass
class AnytimeRiskMonitor:
    """SAVeR 式的 full-feedback 风险监控。

    与 ACI 的三个差别:

    | | ACI | 本监控 |
    | --- | --- | --- |
    | 反馈 | emit-only | **full**(含被弃权样本) |
    | 保证 | 长程平均覆盖 | **任意时刻**有效 |
    | 漂移下 | 跨域失效(Prop.8) | 仍有效 |
    | 代价 | 无 | 需审计标签 + 弃权更多 |

    ⚠️ **不能确证时就弃权** —— 这正是它在跨域下仍成立的原因,
    代价是弃权率会顶到 Prop.3 的下界。
    """
    alpha: float = 0.05
    delta: float = 0.1
    taus: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    kappa_cap: float = 0.5
    # 财富从峰值跌破这个比例即**撤销认证**。没有它认证只涨不跌。
    revoke_ratio: float = 0.5
    window: int = 200              # 近期风险窗口,用于下注和撤销判断

    _states: dict[float, ThresholdState] = field(default_factory=dict,
                                                 repr=False)
    _n_seen: int = 0
    _n_emitted: int = 0
    _n_abstained: int = 0

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha 必须在 (0,1): {self.alpha}")
        if not 0 < self.delta < 1:
            raise ValueError(f"delta 必须在 (0,1): {self.delta}")
        if not self.taus:
            raise ValueError("至少要有一个候选阈值")
        self._states = {t: ThresholdState(tau=t) for t in sorted(self.taus)}

    # ---------- 更新 ----------

    def update(self, score: float, risk: float | None) -> None:
        """喂一个样本。

        Args:
            score: 该样本的分数(高 = 更可能不安全)
            risk:  真值风险 1{漏报}。**None = 没有反馈**,不更新。

        ⚠️ 只更新"该阈值会放行"的样本 —— 阈值 τ 的风险定义在
        {s < τ} 这个放行集上(分数低于阈值才放行)。
        """
        self._n_seen += 1
        if risk is None:
            return
        for t, st in self._states.items():
            if score >= t:            # 该阈值会升级,不计入其放行风险
                continue
            st.n += 1
            st.n_risk += int(risk > 0.5)
            st.recent.append(float(risk))
            if len(st.recent) > self.window:
                st.recent.pop(0)
            # H0: E[R] >= alpha 下 (alpha - r) 的期望非正 -> 财富是上鞅
            kappa = self._bet(st)
            st.wealth *= (1.0 + kappa * (self.alpha - float(risk)))
            if st.wealth <= 0.0:
                st.wealth = 0.0
            if st.wealth >= 1.0 / self.delta:
                st.certified = True   # H0 被否定 -> 该阈值已认证
                st.peak = max(st.peak, st.wealth)
            # ⚠️ **认证必须可撤销。** 上鞅只涨不跌的话,阈值一旦认证就
            # 永久认证,漂移后不会收回 —— 实测跨域漂移(2%->40%)时弃权率
            # 只有 3.8%,远低于 Prop.3 要求的 36.8%,保证是假的。
            # 财富从峰值大幅回落 = 近期证据推翻了先前的认证。
            if st.certified and st.wealth < st.peak * self.revoke_ratio:
                st.certified = False
                st.wealth = 1.0       # 重开一局,不带着旧财富
                st.peak = 1.0

    def _bet(self, st: ThresholdState) -> float:
        """Kelly 式下注,裁剪到 [0, cap]。首观测前不下注。"""
        if st.n <= 1:
            return 0.0
        denom = max(self.alpha * (1 - self.alpha), 1e-6)
        # ⚠️ 用**近期窗口**而非全历史估风险。用全历史的话,漂移前积累的
        # 低风险会一直把下注拉高,监控对新风险反应迟钝。
        m = (sum(st.recent) / len(st.recent)) if st.recent \
            else st.empirical_risk
        return float(min(max((self.alpha - m) / denom, 0.0), self.kappa_cap))

    # ---------- 放行规则 ----------

    def certified_tau(self) -> float | None:
        """当前被认证的最宽松阈值。没有则 None(此时必须全弃权)。

        「认证」= 该阈值上的上鞅财富达到 1/δ,即 H0(风险≥α)被否定。
        """
        cert = [t for t, st in self._states.items() if st.certified]
        return max(cert) if cert else None

    def should_emit(self, score: float) -> bool:
        """**不能确证就弃权。** 这是跨域下仍有效的代价。"""
        tau = self.certified_tau()
        ok = tau is not None and score < tau
        self._n_emitted += int(ok)
        self._n_abstained += int(not ok)
        return ok

    # ---------- 观测 ----------

    @property
    def abstention_rate(self) -> float:
        tot = self._n_emitted + self._n_abstained
        return self._n_abstained / tot if tot else 1.0

    @property
    def stats(self) -> dict:
        tau = self.certified_tau()
        return {"seen": self._n_seen, "certified_tau": tau,
                "abstention_rate": round(self.abstention_rate, 4),
                "n_certified_taus": sum(1 for s in self._states.values()
                                        if s.certified),
                "wealth": {t: round(s.wealth, 3)
                           for t, s in self._states.items()},
                "emit_risk": {t: round(s.empirical_risk, 4)
                              for t, s in self._states.items() if s.n}}

    def respects_floor(self, mu: float) -> bool:
        """弃权率是否达到 Prop.3 的下界。

        不可行区间(µ>α)里,**弃权率低于下界就说明保证是假的**。
        """
        from .aci import abstention_floor
        return self.abstention_rate >= abstention_floor(mu, self.alpha) - 1e-9
