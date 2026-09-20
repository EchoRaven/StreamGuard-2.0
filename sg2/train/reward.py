"""RL 奖励函数（GRPO，按轨迹给分）。

设计见 docs/08_SFT_RL.md §3。奖励能被钻空子是 RL 跑砸的头号原因,
所以这里的每一条防御都有对应的测试(tests/test_reward.py)。

⚠️ 最危险的失败模式是**永远不 flag**:若误报惩罚过大,最优策略退化为
沉默,而这在训练曲线上表现为奖励方差小、loss 平滑,极易被误判为收敛良好。
本模块用 `assert_not_degenerate()` 把这条做成可检查的不变量。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

VALID_ACTIONS = ("hold", "flag", "clear", "uncovered")


class Action(str, Enum):
    HOLD = "hold"
    FLAG = "flag"
    CLEAR = "clear"
    UNCOVERED = "uncovered"


@dataclass(frozen=True)
class Step:
    """轨迹里的一步:模型在某个窗口的输出。"""
    t_s: float
    raw: str                      # 模型原始输出
    tokens: int = 0

    def parse(self) -> dict | None:
        """解析成结构化动作。非法返回 None —— 由格式门处理。"""
        try:
            d = json.loads(self.raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(d, dict) or d.get("action") not in VALID_ACTIONS:
            return None
        return d


@dataclass(frozen=True)
class Episode:
    """一条流的轨迹 + 真值。"""
    steps: tuple[Step, ...]
    nu_s: float | None            # 不安全内容开始时刻;None 表示该流本身安全
    category: str | None = None
    window_s: float = 2.0         # 延迟归一化的单位
    covered: bool = True          # 该类别是否有生效条款。False -> 正确动作是
                                  # uncovered 而非 flag(无条款可引)


@dataclass
class RewardConfig:
    gamma_hit: float = 1.0
    alpha_delay: float = 0.5
    beta_fa: float = 0.3
    mu_token: float = 0.05
    token_ref: float = 2000.0
    max_delay_windows: float = 20.0
    # uncovered 的三档。未覆盖时正确指出缺口应当得分,但低于一次正确的
    # flag —— 否则模型会发现"一律说未覆盖"比认真引用更划算。
    gamma_uncovered_right: float = 0.6    # 确实未覆盖,正确指出
    beta_uncovered_lazy: float = 0.4      # 明明有条款却说未覆盖(偷懒)
    beta_uncovered_safe: float = 0.3      # 安全内容却说未覆盖(误报)

    def __post_init__(self):
        if self.gamma_uncovered_right >= self.gamma_hit:
            raise ValueError(
                f"gamma_uncovered_right({self.gamma_uncovered_right}) >= "
                f"gamma_hit({self.gamma_hit}):一律说未覆盖会比认真引用更划算。"
                "见 docs/09_POLICY.md")
        # 结构性约束,不是调参偏好:beta >= gamma 时最优策略退化为永不 flag
        if self.beta_fa >= self.gamma_hit:
            raise ValueError(
                f"beta_fa({self.beta_fa}) >= gamma_hit({self.gamma_hit}):"
                "误报惩罚压过命中奖励,最优策略将退化为永不 flag。"
                "见 docs/08_SFT_RL.md §3.3")


@dataclass
class RewardBreakdown:
    total: float
    gate_failed: str | None = None
    hit: bool = False
    delay_s: float | None = None
    n_false_alarms: int = 0
    tokens: int = 0
    n_uncovered_right: int = 0
    n_uncovered_wrong: int = 0
    parts: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.gate_failed:
            return f"R=0.000  [门未过: {self.gate_failed}]"
        d = f"{self.delay_s:.1f}s" if self.delay_s is not None else "—"
        u = ""
        if self.n_uncovered_right or self.n_uncovered_wrong:
            u = f" 未覆盖(对{self.n_uncovered_right}/错{self.n_uncovered_wrong})"
        return (f"R={self.total:+.3f}  命中={self.hit} 延迟={d} "
                f"误报={self.n_false_alarms}{u} tokens={self.tokens}")


def compute_reward(ep: Episode, cfg: RewardConfig | None = None
                   ) -> RewardBreakdown:
    """按轨迹计算奖励。

    硬门先于一切:格式错、flag 无引用 -> R=0。
    """
    cfg = cfg or RewardConfig()
    tokens = sum(s.tokens for s in ep.steps)

    flags: list[tuple[Step, dict]] = []
    uncovered: list[Step] = []
    for s in ep.steps:
        d = s.parse()
        if d is None:
            return RewardBreakdown(total=0.0, gate_failed="格式门", tokens=tokens)
        if d["action"] == Action.FLAG:
            # 引用门:flag 必须带 policy_citation
            if not d.get("policy_citation"):
                return RewardBreakdown(total=0.0, gate_failed="引用门",
                                       tokens=tokens)
            flags.append((s, d))
        elif d["action"] == Action.UNCOVERED:
            # uncovered 不过引用门 —— 正因为无条款可引才走这条
            uncovered.append(s)

    hit, delay_s, n_fa = False, None, 0
    for s, d in flags:
        # 类别门:引用指向的类别必须与真值一致,否则这次 flag 算误报
        right_cat = (ep.category is None or d.get("category") == ep.category)
        in_time = ep.nu_s is not None and s.t_s >= ep.nu_s
        if in_time and right_cat and not hit:
            hit, delay_s = True, s.t_s - ep.nu_s
        else:
            n_fa += 1

    # uncovered 三档:确实未覆盖 / 有条款却偷懒 / 安全内容误报
    r_unc = 0.0
    n_unc_right = n_unc_wrong = 0
    for _s in uncovered:
        if ep.nu_s is None:
            r_unc -= cfg.beta_uncovered_safe          # 安全流说未覆盖
            n_unc_wrong += 1
        elif ep.covered:
            r_unc -= cfg.beta_uncovered_lazy          # 有条款却说未覆盖
            n_unc_wrong += 1
        else:
            if n_unc_right == 0:
                r_unc += cfg.gamma_uncovered_right    # 正确指出缺口
            n_unc_right += 1
    # 未覆盖的流上,正确指出缺口即视为"接住了",不再按漏报罚
    caught_uncovered = (not ep.covered) and n_unc_right > 0

    r_hit = cfg.gamma_hit if hit else 0.0
    if hit:
        norm = min(delay_s / max(ep.window_s, 1e-6), cfg.max_delay_windows)
        r_delay = -cfg.alpha_delay * norm / cfg.max_delay_windows
    elif ep.nu_s is not None and not caught_uncovered:
        # 漏报:按最大延迟计,必须比任何误报组合都差
        r_delay = -cfg.alpha_delay
    else:
        r_delay = 0.0
    r_fa = -cfg.beta_fa * n_fa
    r_tok = -cfg.mu_token * (tokens / cfg.token_ref)

    return RewardBreakdown(
        total=r_hit + r_delay + r_fa + r_tok + r_unc,
        hit=hit, delay_s=delay_s, n_false_alarms=n_fa, tokens=tokens,
        n_uncovered_right=n_unc_right, n_uncovered_wrong=n_unc_wrong,
        parts={"hit": r_hit, "delay": r_delay, "fa": r_fa, "token": r_tok,
               "uncovered": r_unc})


def flag_breakeven_precision(cfg: RewardConfig | None = None) -> float:
    """flag 比沉默划算所需的最低精度。

    在一条不安全的流上:
        flag 的期望 = q·gamma - (1-q)·beta      (q = 精度)
        沉默的收益   = -alpha                    (漏报)
    令两者相等解出 q*:

        q* = (beta - alpha) / (gamma + beta)

    q* <= 0 表示**任何**精度下 flag 都优于沉默(通常是想要的)。
    q* 接近 1 表示模型必须极准才敢开口 —— 实际会学成沉默。
    """
    cfg = cfg or RewardConfig()
    return (cfg.beta_fa - cfg.alpha_delay) / (cfg.gamma_hit + cfg.beta_fa)


def fa_exchange_rate(cfg: RewardConfig | None = None) -> float:
    """一次误报要几次正确检测来抵。beta/gamma。"""
    cfg = cfg or RewardConfig()
    return cfg.beta_fa / cfg.gamma_hit


@dataclass
class CounterfactualPair:
    """同一窗口在两条政策下的一对 rollout。

    政策 A 覆盖该内容,政策 B 不覆盖。正确行为是 A 下 flag、B 下不 flag。
    """
    raw_covering: str
    raw_not_covering: str
    target_citation: str | None = None    # A 中目标条款的(增强后)id
    tokens: int = 0


@dataclass
class CFConfig:
    """反事实一致性奖励。

    **这是 SFT 结构上做不到的事。** 政策敏感度是关于**一对输入**的性质:
    同一视频换政策,判决应翻转。SFT 一次只看一个 (输入,目标),
    表达不了"这两个输出之间应有什么关系"。

    ⚠️ 必须按**方向**判而不是按"是否不同"。只奖励"翻转"的话,最优策略
    是**一律翻转** —— 看到政策变了就换个答案,与内容无关。
    """
    gamma_consistent: float = 0.5     # 方向正确的翻转
    beta_insensitive: float = 0.4     # 两边同答(在背不在读)
    beta_reversed: float = 0.6        # 翻转但方向反了 —— 比不敏感更糟
    require_citation: bool = True     # A 下的 flag 必须引对条款

    def __post_init__(self):
        if self.beta_reversed <= self.beta_insensitive:
            raise ValueError(
                f"beta_reversed({self.beta_reversed}) 必须 > "
                f"beta_insensitive({self.beta_insensitive}):方向反了比"
                "不敏感更糟 —— 前者是学到了错误关系,后者只是没学到。")


@dataclass
class CFBreakdown:
    total: float
    outcome: str                      # consistent|insensitive|reversed|invalid
    verdict_covering: str = ""
    verdict_not_covering: str = ""
    citation_ok: bool = False

    def __str__(self) -> str:
        return (f"R_cf={self.total:+.3f} [{self.outcome}] "
                f"{self.verdict_covering}→{self.verdict_not_covering}"
                f"{' 引用✓' if self.citation_ok else ''}")


def counterfactual_reward(pair: CounterfactualPair,
                          cfg: CFConfig | None = None) -> CFBreakdown:
    """按方向判的反事实一致性奖励。

    四种结局:
        consistent  A=flag(引对) 且 B≠flag   -> 正奖励
        reversed    A≠flag 且 B=flag         -> 最重惩罚(学到了错误关系)
        insensitive 两边同答                  -> 惩罚(在背不在读)
        invalid     任一侧输出非法            -> 0
    """
    cfg = cfg or CFConfig()
    a = Step(0.0, pair.raw_covering, pair.tokens).parse()
    b = Step(0.0, pair.raw_not_covering, pair.tokens).parse()
    if a is None or b is None:
        return CFBreakdown(0.0, "invalid")

    va, vb = a["action"], b["action"]
    a_flag = va == Action.FLAG
    b_flag = vb == Action.FLAG
    cite_ok = True
    if a_flag and cfg.require_citation and pair.target_citation:
        cite_ok = a.get("policy_citation") == pair.target_citation

    if a_flag and not b_flag:
        r = cfg.gamma_consistent if cite_ok else 0.0
        return CFBreakdown(r, "consistent" if cite_ok else "wrong_citation",
                           va, vb, cite_ok)
    if b_flag and not a_flag:
        return CFBreakdown(-cfg.beta_reversed, "reversed", va, vb, cite_ok)
    return CFBreakdown(-cfg.beta_insensitive, "insensitive", va, vb, cite_ok)


def assert_cf_not_gameable(cfg: CFConfig | None = None) -> None:
    """检查"一律翻转"不会成为最优策略。

    构造两个窗口:一个该翻(内容被覆盖),一个不该翻(内容本就安全,
    两边都应 clear)。一律翻转的策略在第二个上会被判 reversed。
    """
    cfg = cfg or CFConfig()
    F = json.dumps({"action": "flag", "category": "C1",
                    "policy_citation": "X1"})
    C = json.dumps({"action": "clear"})

    honest = (counterfactual_reward(
        CounterfactualPair(F, C, target_citation="X1"), cfg).total
        + counterfactual_reward(CounterfactualPair(C, C), cfg).total)
    always_flip = (counterfactual_reward(
        CounterfactualPair(F, C, target_citation="X1"), cfg).total
        + counterfactual_reward(CounterfactualPair(C, F), cfg).total)
    if always_flip >= honest:
        raise ValueError(
            f"退化:一律翻转得分 {always_flip:.3f} >= 诚实策略 {honest:.3f}。"
            "beta_reversed 需大于 beta_insensitive。")


def assert_not_degenerate(cfg: RewardConfig | None = None, *,
                          max_breakeven_precision: float = 0.3) -> None:
    """检查配置不会让"永远沉默"成为最优策略。

    两条检查:

    1. **零误报的正确 flag 必须优于沉默。** 这条几乎总能通过 —— 单独用它
       是一个空检查(任何 gamma>0 都过),所以它只是底线。
    2. **盈亏平衡精度必须够低。** 这才是有效的那条:模型需要 q* 的精度才敢
       flag,而训练早期精度必然很低,此时它会学会闭嘴然后再也学不回来。

    ⚠️ `max_breakeven_precision=0.3` 是**选定的**阈值,不是推导出来的。
    依据只是"RL 早期精度通常低于 30%"这一经验判断。有了真实训练曲线后
    应该用实测的早期精度来重定这个值。

    ⚠️ 沉默退化在训练曲线上表现为奖励方差小、loss 平滑,极易被误判为收敛。
    """
    cfg = cfg or RewardConfig()

    good = Episode(
        steps=(Step(10.0, json.dumps({"action": "flag", "category": "C1",
                                      "policy_citation": "P1"}), tokens=50),),
        nu_s=8.0, category="C1")
    silent = Episode(
        steps=(Step(10.0, json.dumps({"action": "hold"}), tokens=50),),
        nu_s=8.0, category="C1")
    r_good = compute_reward(good, cfg).total
    r_silent = compute_reward(silent, cfg).total
    if r_good <= r_silent:
        raise ValueError(
            f"退化配置:正确 flag {r_good:.3f} <= 永远沉默 {r_silent:.3f}")

    q = flag_breakeven_precision(cfg)
    if q > max_breakeven_precision:
        raise ValueError(
            f"退化配置:盈亏平衡精度 q*={q:.3f} > {max_breakeven_precision};"
            f"模型需要 {q:.0%} 的精度才敢 flag,训练早期精度必然低于此,"
            "会学成沉默且再也学不回来。见 docs/08_SFT_RL.md §3.3")
