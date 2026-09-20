"""Out-of-policy:内容有害但无条款覆盖。

**当前动作空间的一个洞。** 三元动作 hold/flag/clear + 引用门,合起来让
"明显有害但没有任何条款覆盖"的内容**没有合法输出**:

  - flag  -> 引用门要求 policy_citation,没有条款可引 -> 奖励清零
  - clear -> 它确实有害
  - hold  -> 永远等下去,事件不闭合

模型被逼着**编造引用**(最坏)或**放行**(次坏)。所以需要第四个动作。

    uncovered = 看起来有害,但现行政策没有覆盖它

它不要求引用(所以过得了引用门),会触发政策补缺流程,且在奖励里是独立
的一档 —— 既不当命中也不当误报。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from .policy import ClauseStatus, PolicyClause, PolicyCorpus

ACTION_UNCOVERED = "uncovered"


@dataclass
class UncoveredCase:
    """一次 out-of-policy 观测。积累够了才提议补条款。"""
    clip_id: str
    t_s: float
    description: str
    suggested_category: str | None = None
    confidence: float | None = None
    evidence_frames: tuple[int, ...] = ()


@dataclass
class PolicyGap:
    """一个被识别出的政策缺口。"""
    suggested_category: str
    n_cases: int
    cases: list[UncoveredCase]
    first_seen: str
    last_seen: str

    @property
    def is_actionable(self) -> bool:
        """够不够提议补条款。单个样本不构成缺口 —— 可能只是误判。"""
        return self.n_cases >= PolicyGapTracker.MIN_CASES


class PolicyGapTracker:
    """累积 uncovered 观测,识别缺口。

    ⚠️ 不自动改政策。识别缺口与修改政策是两件事,后者必须经人工。
    """

    MIN_CASES = 5        # 少于这个数不提议 —— 单点很可能是误判

    def __init__(self, corpus: PolicyCorpus):
        self.corpus = corpus
        self._cases: list[UncoveredCase] = []

    def record(self, case: UncoveredCase) -> bool:
        """记录一次 uncovered。

        Returns:
            True 表示确实未覆盖;False 表示该类别其实**有**条款 ——
            那是模型该引用却没引,属于模型错误而非政策缺口。
        """
        if case.suggested_category and self.corpus.covers(case.suggested_category):
            return False
        self._cases.append(case)
        return True

    def gaps(self) -> list[PolicyGap]:
        by_cat: dict[str, list[UncoveredCase]] = {}
        for c in self._cases:
            by_cat.setdefault(c.suggested_category or "<未分类>", []).append(c)
        out = []
        for cat, cases in sorted(by_cat.items(),
                                 key=lambda kv: -len(kv[1])):
            out.append(PolicyGap(
                suggested_category=cat, n_cases=len(cases), cases=cases,
                first_seen=f"{min(c.t_s for c in cases):.1f}s",
                last_seen=f"{max(c.t_s for c in cases):.1f}s"))
        return out

    def actionable_gaps(self) -> list[PolicyGap]:
        return [g for g in self.gaps() if g.is_actionable]

    @property
    def stats(self) -> dict:
        return {"cases": len(self._cases),
                "gaps": len(self.gaps()),
                "actionable": len(self.actionable_gaps()),
                "by_category": dict(Counter(
                    c.suggested_category or "<未分类>" for c in self._cases))}


PROPOSAL_PROMPT = """现有政策语料未覆盖以下观测到的内容。请提议一条新政策条款。

已有类别:
${existing}

未覆盖的观测(${n} 例):
${cases}

输出一个 JSON 对象:
{"id":"<新条款id,形如 X1_slug>","category":"<同 id>","title":"<简短标题>","text":"<一句话的判定标准>"}

要求:判定标准必须可操作(说清楚看到什么算违规),不要与已有类别重叠。"""


def build_proposal_prompt(gap: PolicyGap, corpus: PolicyCorpus,
                          max_cases: int = 8) -> str:
    """构造提议新条款的 prompt。"""
    from string import Template
    existing = "\n".join(f"- [{c.id}] {c.title}" for c in corpus.enforced())
    cases = "\n".join(f"- {c.description}" for c in gap.cases[:max_cases])
    return Template(PROPOSAL_PROMPT).substitute(
        existing=existing, n=gap.n_cases, cases=cases)


def draft_clause_from(proposal: dict, gap: PolicyGap) -> PolicyClause:
    """把模型的提议变成一条 **draft** 条款。

    ⚠️ 一定是 draft。生成的条款必须经 `PolicyCorpus.approve(reviewer=...)`
    才能生效 —— 自动生成并自动启用审核政策,等于让系统在无人知晓的情况下
    改变判定范围。`PolicyClause.__post_init__` 会强制这一点。
    """
    required = ("id", "category", "title", "text")
    missing = [k for k in required if not proposal.get(k)]
    if missing:
        raise ValueError(f"提议缺字段 {missing}: {proposal}")
    return PolicyClause(
        id=str(proposal["id"]), category=str(proposal["category"]),
        title=str(proposal["title"]), text=str(proposal["text"]),
        status=ClauseStatus.DRAFT, provenance="generated",
        added_on=date.today().isoformat(),
        examples=[c.description for c in gap.cases[:5]])


@dataclass
class PolicySwap:
    """一次政策替换的记录。

    政策变更 = **校准纪元边界**(docs/03_ADAPTATION.md §4):
    判定范围变了,旧阈值的 conformal 保证立即失效,必须重新校准。
    """
    old_fingerprint: str
    new_fingerprint: str
    reason: str
    at: str = field(default_factory=lambda: date.today().isoformat())
    requires_recalibration: bool = True
    affected_categories: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (f"政策替换 {self.old_fingerprint} -> {self.new_fingerprint} "
                f"({self.reason});"
                f"{'需重新校准' if self.requires_recalibration else '无需重校准'}")


def swap_policy(old: PolicyCorpus, new: PolicyCorpus, *,
                reason: str) -> PolicySwap:
    """替换政策语料,并判断是否需要重新校准。

    只有**判定范围**变化才需要重校准。仅调整条款顺序(缓解位置偏置)不算 ——
    `fingerprint()` 与顺序无关正是为此。
    """
    of, nf = old.fingerprint(), new.fingerprint()
    changed = sorted((old.categories() | new.categories())
                     - (old.categories() & new.categories()))
    return PolicySwap(old_fingerprint=of, new_fingerprint=nf, reason=reason,
                      requires_recalibration=(of != nf),
                      affected_categories=tuple(changed))
