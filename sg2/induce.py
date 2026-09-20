"""政策归纳:从少量带标样本反推政策定义文本。

**为什么需要它。** 到一个新 benchmark 时,你手里有的是**标签,不是政策
文档** —— 标签里编码着一条从没被写下来的政策。`docs/06 §3.6` 的零日实验
原本假设"有人把政策写出来",这在换 benchmark 时不成立。

**它属于哪一级:都不属于。** 三级(sentinel/中间层/analyst)都是在线组件,
有延迟预算。政策归纳每个新 benchmark 只跑一次、要一次看很多样本、要
迭代、没有延迟约束 —— 它是**离线编译器**。

分工:
    生成(propose/revise) -> frontier,离线,次数少
    评估(score)          -> 中间层,要跑很多次,便宜
    消费                  -> 中间层在线读,**自己从不生成政策**

**必须是循环。** 单轮提议看不见自己的错误。加上循环才从"让 LLM 写条
政策"变成有可测目标的优化 —— 这与 docs/08 §4.3 的离线 prompt 编译是
同一套机械,只是优化的文本换成了政策定义。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Callable, Sequence

from .policy import ClauseStatus, PolicyClause, PolicyCorpus


@dataclass(frozen=True)
class LabeledExample:
    """一条带标样本。`text` 是它的可读描述(感知层产出或数据集自带)。"""
    id: str
    text: str
    label: bool                      # True = 该政策判定为 unsafe


@dataclass
class InductionMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def __str__(self) -> str:
        return (f"F1={self.f1:.3f} P={self.precision:.3f} "
                f"R={self.recall:.3f} (n={self.n})")


@dataclass
class InductionRound:
    round: int
    clause_text: str
    metrics: InductionMetrics
    n_errors: int


@dataclass
class InductionResult:
    clause: PolicyClause
    history: list[InductionRound]
    best_round: int

    @property
    def best(self) -> InductionRound:
        return self.history[self.best_round]

    @property
    def improved(self) -> bool:
        """循环是否真的有用。没有提升就该老实报告单轮就够了。"""
        return len(self.history) > 1 and self.best_round > 0

    def curve(self) -> list[float]:
        return [r.metrics.f1 for r in self.history]


PROPOSE_TMPL = """下面是某条安全政策的正例与负例。请把这条政策**写出来**。

判定为违规的样本:
${positives}

判定为**不**违规的样本:
${negatives}

输出一个 JSON 对象:
{"id":"<条款id,形如 N1_slug>","title":"<简短标题>","text":"<一句话判定标准>"}

要求:
- 判定标准要能把上面两组分开,且能推广到没见过的样本
- 不要罗列具体样本,要写出**共同的判定依据**
- 不要写表面特征(颜色、分辨率之类),除非它确实是判定依据"""

REVISE_TMPL = """当前政策条款:
${clause}

用它判定时出现了以下错误:

误报(判成违规,实际不违规):
${false_positives}

漏报(判成不违规,实际违规):
${false_negatives}

请修订这条条款,使它同时减少这两类错误。输出同样的 JSON 格式。
当前 F1 = ${f1}。"""


def _fmt(examples: Sequence[LabeledExample], limit: int = 10) -> str:
    if not examples:
        return "(无)"
    return "\n".join(f"- {e.text}" for e in examples[:limit])


def build_propose_prompt(examples: Sequence[LabeledExample],
                         limit: int = 10) -> str:
    pos = [e for e in examples if e.label]
    neg = [e for e in examples if not e.label]
    return Template(PROPOSE_TMPL).substitute(
        positives=_fmt(pos, limit), negatives=_fmt(neg, limit))


def build_revise_prompt(clause_text: str, fps: Sequence[LabeledExample],
                        fns: Sequence[LabeledExample], f1: float,
                        limit: int = 6) -> str:
    return Template(REVISE_TMPL).substitute(
        clause=clause_text, false_positives=_fmt(fps, limit),
        false_negatives=_fmt(fns, limit), f1=f"{f1:.3f}")


def score(clause: PolicyClause, dev: Sequence[LabeledExample],
          judge: Callable[[PolicyClause, LabeledExample], bool],
          ) -> tuple[InductionMetrics, list[LabeledExample],
                     list[LabeledExample]]:
    """在 dev 集上评估一条候选条款。

    `judge` 由调用方提供 —— 生产上是中间层,测试里可以是任何函数。
    这一层刻意不绑定模型:评估要跑很多次,该用便宜的那一级。
    """
    m = InductionMetrics()
    fps: list[LabeledExample] = []
    fns: list[LabeledExample] = []
    for e in dev:
        pred = judge(clause, e)
        if pred and e.label:
            m.tp += 1
        elif pred and not e.label:
            m.fp += 1
            fps.append(e)
        elif not pred and e.label:
            m.fn += 1
            fns.append(e)
        else:
            m.tn += 1
    return m, fps, fns


def induce_policy(
    propose_set: Sequence[LabeledExample],
    dev_set: Sequence[LabeledExample],
    *,
    generate: Callable[[str], dict],
    judge: Callable[[PolicyClause, LabeledExample], bool],
    max_rounds: int = 4,
    min_gain: float = 0.01,
    category: str | None = None,
) -> InductionResult:
    """从带标样本归纳出一条政策条款。

    Args:
        propose_set: 展示给生成器的样本(`pool="exemplar"`)
        dev_set:     给候选打分的样本(`pool="compile"`)
        generate:    prompt -> {"id","title","text"};接 frontier
        judge:       (条款, 样本) -> 是否判违规;接中间层
        min_gain:    F1 提升小于此值即停止 —— 避免在噪声上空转

    ⚠️ `propose_set` 与 `dev_set` **必须不相交**,且两者都不能来自最终
    评测集。在评测集上调政策,报出来的 F1 没有意义。`sg2.schema.load()`
    的池守卫就是为了让这件事不可能被意外违反。

    ⚠️ 产出的条款是 **draft**。生成的政策不得自动生效 —— 见
    `PolicyCorpus.approve(reviewer=...)`。
    """
    if not propose_set:
        raise ValueError("propose_set 不能为空")
    if not dev_set:
        raise ValueError("dev_set 不能为空 —— 没有 dev 就无法选候选,"
                         "循环退化成单轮且无从判断好坏")
    overlap = {e.id for e in propose_set} & {e.id for e in dev_set}
    if overlap:
        raise ValueError(
            f"propose_set 与 dev_set 重叠 {sorted(overlap)[:5]};"
            "在同一批样本上既提议又选型,F1 会虚高")

    history: list[InductionRound] = []
    prompt = build_propose_prompt(propose_set)
    best_clause: PolicyClause | None = None
    best_f1 = -1.0
    best_round = 0

    for rnd in range(max_rounds):
        proposal = generate(prompt)
        missing = [k for k in ("id", "title", "text") if not proposal.get(k)]
        if missing:
            raise ValueError(f"第 {rnd} 轮提议缺字段 {missing}: {proposal}")
        cand = PolicyClause(
            id=str(proposal["id"]),
            category=category or str(proposal.get("category", proposal["id"])),
            title=str(proposal["title"]), text=str(proposal["text"]),
            status=ClauseStatus.DRAFT, provenance="generated")

        m, fps, fns = score(cand, dev_set, judge)
        history.append(InductionRound(rnd, cand.text, m, len(fps) + len(fns)))

        if m.f1 > best_f1 + min_gain:
            best_clause, best_f1, best_round = cand, m.f1, rnd
        elif rnd > 0:
            break                      # 不再有实质提升

        if not fps and not fns:
            break                      # dev 上已全对
        prompt = build_revise_prompt(cand.text, fps, fns, m.f1)

    assert best_clause is not None
    return InductionResult(clause=best_clause, history=history,
                           best_round=best_round)
