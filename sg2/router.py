"""升级路由:把「检测」与「路由」分开。

**先前的设计把两件事混成了一件。** CUSUM 在风险分上报警就升级,
等于「看起来危险就上送」。但级联的正确判据是**「低级处理不了就上送」**
—— 能力/不确定性判据,不是风险判据。

    |            | 低不确定        | 高不确定            |
    |------------|----------------|--------------------|
    | 高风险分    | 廉价层直接判决   | 上送               |
    | **低风险分**| 放行           | **必须上送** ←漏洞 |

左下那格正是 sentinel 的盲区(docs/06 §6.2):它对某类内容没有判别力,
**所以分数低 —— 而分数低恰恰是因为它看不懂**。纯风险路由永远不升级。

两者是**互补**而非替代:
    CUSUM(风险) 回答"什么时候变了" —— §3 的延迟界建立在它上面
    不确定性     回答"这一条我自己能不能定" —— 决定要不要花钱
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Route(str, Enum):
    DECIDE_UNSAFE = "decide_unsafe"   # 廉价层自己判违规,不上送
    DECIDE_SAFE = "decide_safe"       # 廉价层自己放行
    ESCALATE = "escalate"             # 自己定不了,上送
    FORCED = "forced"                 # 保底随机覆盖,与分数无关


@dataclass
class RoutingDecision:
    route: Route
    score: float
    uncertainty: float
    voi: float                        # 信息价值:上送能改变决定的期望收益
    reason: str = ""

    @property
    def escalated(self) -> bool:
        return self.route in (Route.ESCALATE, Route.FORCED)


@dataclass
class CompetenceRouter:
    """按**能力**而非风险路由。

    核心量是 VOI(value of information):上送之后决定被改变的概率。
    只有决定可能被改变时,花钱才有意义 —— 这是级联的信息论基础。

    Args:
        tau: 廉价层的判决阈值(由 conformal 校准给出)
        margin: |score - tau| 小于它即视为"贴近边界"
        ood_weight: OOD 分数在不确定性里的权重。**这一项是关键** ——
            分数低但 OOD 高,正是"我没见过所以看不懂",必须上送。
    """
    tau: float = 0.0
    margin: float = 0.15
    ood_weight: float = 0.6
    voi_threshold: float = 0.25
    confident_unsafe_margin: float = 0.4   # 高出阈值这么多就自己判

    # ⚠️ **引用门带来的架构约束。** sentinel 只产出连续分数,产不出
    # policy_citation,而 flag 必须带引用(docs/07 §3.2)。所以 sentinel
    # **不能自己判违规** —— 任何"不安全"的判定都必须经中间层。
    #
    # 后果:级联只能在**否定侧**省钱(放行不必上送),肯定侧省不了。
    # 这直接压低了级联的成本上限 —— 升级率的下界就是真实的不安全率。
    can_decide_unsafe: bool = False
    # CUSUM 报警是"刚刚变了"的证据,应提高上送倾向但不等于上送
    alarm_uncertainty_boost: float = 0.3
    # OOD 高到这个程度就**否决放行** —— 低分不足以证明安全
    ood_safe_veto: float = 0.15

    _n: dict = field(default_factory=lambda: {r: 0 for r in Route},
                     repr=False)

    # ---------- 不确定性 ----------

    def boundary_uncertainty(self, score: float) -> float:
        """贴近判决边界的程度。1 = 正好在边界上。"""
        d = abs(score - self.tau)
        return math.exp(-(d / max(self.margin, 1e-9)) ** 2)

    def combine_uncertainty(self, score: float, ood: float = 0.0,
                            disagreement: float = 0.0) -> float:
        """合并三种"我不确定"的来源。

        ⚠️ OOD 与边界不确定是**不同**的东西。分数远离边界但 OOD 很高,
        意味着"我很确定 —— 但我确定的依据是我没见过的分布",
        那种确定不值钱。
        """
        b = self.boundary_uncertainty(score)
        other = max(ood, disagreement)
        return min(1.0, (1 - self.ood_weight) * b + self.ood_weight * other)

    # ---------- VOI ----------

    def value_of_information(self, score: float, uncertainty: float) -> float:
        """上送能改变决定的期望收益。

        决定已经很稳时(远离边界且不确定性低),上送改变不了什么,VOI 低。
        """
        return uncertainty * self.boundary_uncertainty(score) ** 0.5 \
            + uncertainty * 0.5

    # ---------- 路由 ----------

    def route(self, score: float, *, ood: float = 0.0,
              disagreement: float = 0.0, forced: bool = False,
              alarmed: bool = False) -> RoutingDecision:
        """路由一次采样。

        `alarmed` 是 CUSUM 的变点信号 —— 它回答"什么时候变了",
        与"我自己能不能定"是**两件事**,所以只作为不确定性的一项输入,
        不直接等于上送。
        """
        u = self.combine_uncertainty(score, ood, disagreement)
        if alarmed:
            u = min(1.0, u + self.alarm_uncertainty_boost)
        v = self.value_of_information(score, u)

        if forced:
            d = RoutingDecision(Route.FORCED, score, u, v, "保底随机覆盖")
        elif v >= self.voi_threshold:
            d = RoutingDecision(Route.ESCALATE, score, u, v,
                                f"VOI={v:.2f} ≥ {self.voi_threshold}")
        elif (self.can_decide_unsafe
              and score >= self.tau + self.confident_unsafe_margin):
            d = RoutingDecision(Route.DECIDE_UNSAFE, score, u, v,
                                "远高于阈值且确定,自己判")
        elif score >= self.tau + self.confident_unsafe_margin:
            # 引用门:判违规必须能引条款,sentinel 引不了 -> 只能上送
            d = RoutingDecision(Route.ESCALATE, score, u, v,
                                "疑似违规,但 sentinel 无法产出引用")
        elif score <= self.tau - self.confident_unsafe_margin:
            # ⚠️ 放行必须同时满足"分数低"**和**"不 OOD"。
            # 只看分数是设计缺陷:分数低可能是"确实安全",也可能是
            # "我没见过所以打不出高分" —— 后者放行就是漏报。
            # 实测:只按分数判时,分布外帧因相似度低被大量放行。
            if max(ood, disagreement) >= self.ood_safe_veto:
                d = RoutingDecision(Route.ESCALATE, score, u, v,
                                    f"分数低但 OOD={max(ood, disagreement):.2f},"
                                    "低分可能只是看不懂")
            else:
                d = RoutingDecision(Route.DECIDE_SAFE, score, u, v,
                                    "远低于阈值且不 OOD,放行")
        elif score >= self.tau:
            d = (RoutingDecision(Route.DECIDE_UNSAFE, score, u, v,
                                 "近阈值,自己判违规")
                 if self.can_decide_unsafe
                 else RoutingDecision(Route.ESCALATE, score, u, v,
                                      "近阈值偏违规,需中间层给引用"))
        elif max(ood, disagreement) >= self.ood_safe_veto:
            d = RoutingDecision(Route.ESCALATE, score, u, v,
                                f"低于阈值但 OOD={max(ood, disagreement):.2f}")
        else:
            d = RoutingDecision(Route.DECIDE_SAFE, score, u, v,
                                "低于阈值且不 OOD,放行")
        self._n[d.route] += 1
        return d

    @property
    def stats(self) -> dict:
        tot = sum(self._n.values()) or 1
        return {r.value: self._n[r] for r in Route} | {
            "escalation_rate": round(
                (self._n[Route.ESCALATE] + self._n[Route.FORCED]) / tot, 4)}


def compare_routing(scores, oods, *, tau=0.0, risk_threshold=0.5,
                    router: CompetenceRouter | None = None) -> dict:
    """对照:纯风险路由 vs 能力路由。

    重点看**低风险高 OOD**那一格 —— 纯风险路由永远不升级,
    而那正是 sentinel 看不懂的内容。
    """
    router = router or CompetenceRouter(tau=tau)
    risk_esc = comp_esc = blind = 0
    for s, o in zip(scores, oods):
        r_esc = s >= risk_threshold
        c_esc = router.route(s, ood=o).escalated
        risk_esc += r_esc
        comp_esc += c_esc
        if (not r_esc) and c_esc and o > 0.5:
            blind += 1        # 纯风险路由会漏掉、而能力路由抓到的盲区样本
    n = len(scores) or 1
    return {"n": n,
            "risk_escalation_rate": round(risk_esc / n, 4),
            "competence_escalation_rate": round(comp_esc / n, 4),
            "blindspot_caught": blind,
            "blindspot_rate": round(blind / n, 4)}
