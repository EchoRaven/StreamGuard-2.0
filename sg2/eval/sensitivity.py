"""政策敏感度:模型到底在**读**政策,还是在**背**政策?

这是整条政策归纳链的前提。归纳出一条新政策交给中间层时,中间层是
**不重训**的 —— 它只是读新政策。若它其实在背训练时见过的类别名,
生成的政策就不会改变它的判决,归纳白做。

**判据是反事实:** 同一段视频,给两条不同的政策(一条覆盖、一条不覆盖),
判决应当翻转。不翻转 = 模型在忽略政策。

⚠️ 这个性质**会被 SFT 破坏**。在固定语料上微调,模型可以记住
"C1 就是色情"从此不再读条款文本。所以它必须是**训练中被监控的指标**,
不是训练结束才测一次的评测项。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..policy import PolicyClause, PolicyCorpus


@dataclass
class FlipCase:
    """一次反事实对照。"""
    item_id: str
    verdict_covering: str          # 给"覆盖该内容的政策"时的动作
    verdict_not_covering: str      # 给"不覆盖的政策"时的动作
    expected_flip: bool = True

    @property
    def flipped(self) -> bool:
        return self.verdict_covering != self.verdict_not_covering

    @property
    def correct(self) -> bool:
        """覆盖时应 flag,不覆盖时不应 flag。"""
        return (self.verdict_covering == "flag"
                and self.verdict_not_covering in ("clear", "hold", "uncovered"))


@dataclass
class SensitivityReport:
    cases: list[FlipCase] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def flip_rate(self) -> float:
        """判决随政策改变的比例。低 = 在背不在读。"""
        return sum(c.flipped for c in self.cases) / self.n if self.n else 0.0

    @property
    def correct_rate(self) -> float:
        """不仅翻转,而且翻转方向正确。"""
        return sum(c.correct for c in self.cases) / self.n if self.n else 0.0

    @property
    def always_flag(self) -> bool:
        """两种政策下都 flag —— 典型的"背下来了"。"""
        return self.n > 0 and all(
            c.verdict_covering == "flag" and c.verdict_not_covering == "flag"
            for c in self.cases)

    @property
    def always_same(self) -> bool:
        return self.n > 0 and self.flip_rate == 0.0

    def verdict(self) -> str:
        if self.n == 0:
            return "无样本"
        if self.always_flag:
            return "✗ 两种政策都 flag —— 在背不在读"
        if self.always_same:
            return "✗ 判决完全不随政策变 —— 在背不在读"
        if self.correct_rate >= 0.8:
            return "✓ 读政策"
        if self.flip_rate >= 0.5:
            return "! 会随政策变,但方向常错"
        return "✗ 敏感度过低"

    def __str__(self) -> str:
        return (f"{self.verdict()}  翻转率={self.flip_rate:.2f} "
                f"方向正确率={self.correct_rate:.2f} (n={self.n})")


DISTRACTOR = PolicyClause(
    id="Z9_unrelated", category="Z9_unrelated", title="无关条款",
    text="禁止画面中出现可识别的车牌号码。")

SAFE_EXIT = PolicyClause(
    id="Z0_ok", category="Z0_ok", title="其余一律放行",
    text="未被上述条款覆盖的画面均不违规,应回 clear。")


def make_counterfactual_pair(covering: PolicyClause
                             ) -> tuple[PolicyCorpus, PolicyCorpus]:
    """造一对政策:一条覆盖目标内容,一条不覆盖。

    两边**条款数相同**且都带安全出口 —— 否则差异可能来自条款数或
    "无处可去"(实测:只有禁止性条款时模型会被迫用 uncovered 表达
    不违规),而不是来自政策内容。
    """
    return (PolicyCorpus(name="covering", clauses=[covering, SAFE_EXIT]),
            PolicyCorpus(name="not_covering", clauses=[DISTRACTOR, SAFE_EXIT]))


def measure_sensitivity(items: list[tuple[str, object]],
                        covering: PolicyClause,
                        judge: Callable[[PolicyCorpus, object], str],
                        ) -> SensitivityReport:
    """对一批内容测政策敏感度。

    Args:
        items: [(id, 内容)],内容会原样传给 judge
        covering: 覆盖这批内容的条款
        judge: (政策语料, 内容) -> 动作字符串
    """
    cov, notcov = make_counterfactual_pair(covering)
    rep = SensitivityReport()
    for iid, content in items:
        rep.cases.append(FlipCase(
            item_id=iid,
            verdict_covering=judge(cov, content),
            verdict_not_covering=judge(notcov, content)))
    return rep
