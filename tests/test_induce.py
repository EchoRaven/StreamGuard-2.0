"""政策归纳的测试。

核心是**循环真的有用**,以及**数据分离不可被绕过** —— 在同一批样本上
既提议又选型,F1 会虚高,而这种错误在结果里看不出来。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.induce import (InductionMetrics, LabeledExample, build_propose_prompt,
                        build_revise_prompt, induce_policy, score)
from sg2.policy import ClauseStatus, PolicyClause

PROPOSE = ([LabeledExample(f"p{i}", t, True)
            for i, t in enumerate(["厨房里有把菜刀", "手持水果刀", "桌上放着刀具"])]
           + [LabeledExample(f"n{i}", t, False)
              for i, t in enumerate(["厨房里在切菜", "有人在吃水果", "桌上放着盘子"])])

DEV = [LabeledExample(f"d{i}", t, lab) for i, (t, lab) in enumerate(
    [("拿着一把刀", True), ("厨房在做饭", False), ("刀架上有刀", True),
     ("在洗盘子", False), ("厨房的砧板", False)])]


def _judge(c: PolicyClause, e: LabeledExample) -> bool:
    kws = [k for k in ("刀", "厨房", "盘子") if k in c.text]
    return any(k in e.text for k in kws)


def _gen(drafts):
    it = iter(drafts)
    last = drafts[-1]
    return lambda _p: next(it, last)


BAD = {"id": "N1", "title": "厨房", "text": "禁止出现厨房场景。"}
GOOD = {"id": "N1", "title": "刀具", "text": "禁止画面中出现刀具。"}


# ==================== 指标 ====================

def test_f1_edge_cases():
    assert InductionMetrics().f1 == 0.0
    assert InductionMetrics(tp=5).f1 == 1.0
    assert InductionMetrics(fp=3, fn=2).f1 == 0.0


def test_score_returns_the_two_error_lists():
    c = PolicyClause(id="N1", category="N1", title="t", text="禁止出现厨房场景。",
                     status=ClauseStatus.DRAFT, provenance="generated")
    m, fps, fns = score(c, DEV, _judge)
    assert {e.text for e in fps} == {"厨房在做饭", "厨房的砧板"}
    assert {e.text for e in fns} == {"拿着一把刀", "刀架上有刀"}
    assert m.n == len(DEV)


# ==================== 循环 ====================

def test_loop_improves_over_single_shot():
    """循环的意义:单轮提议看不见自己的错误。"""
    r = induce_policy(PROPOSE, DEV, generate=_gen([BAD, GOOD]), judge=_judge)
    assert r.improved
    assert r.curve()[0] < r.curve()[-1]
    assert r.clause.text == GOOD["text"]


def test_stops_early_when_dev_is_perfect():
    r = induce_policy(PROPOSE, DEV, generate=_gen([GOOD]), judge=_judge,
                      max_rounds=5)
    assert len(r.history) == 1


def test_stops_when_gain_stalls():
    """F1 不再上升就停,避免在噪声上空转。"""
    r = induce_policy(PROPOSE, DEV, generate=_gen([BAD, BAD, BAD]),
                      judge=_judge, max_rounds=5)
    assert len(r.history) <= 2


def test_respects_max_rounds():
    alt = [BAD, GOOD, BAD, GOOD, BAD, GOOD]
    r = induce_policy(PROPOSE, DEV, generate=_gen(alt), judge=_judge,
                      max_rounds=3)
    assert len(r.history) <= 3


def test_returns_best_not_last():
    """选 dev 上最好的那轮,不是最后一轮。"""
    r = induce_policy(PROPOSE, DEV, generate=_gen([GOOD, BAD, BAD]),
                      judge=_judge, max_rounds=3)
    assert r.clause.text == GOOD["text"]


# ==================== 数据分离 ====================

def test_overlapping_sets_are_refused():
    """在同一批样本上既提议又选型 -> F1 虚高,且从结果里看不出来。"""
    with pytest.raises(ValueError, match="重叠"):
        induce_policy(PROPOSE, PROPOSE, generate=_gen([GOOD]), judge=_judge)


def test_empty_dev_is_refused():
    with pytest.raises(ValueError, match="dev_set"):
        induce_policy(PROPOSE, [], generate=_gen([GOOD]), judge=_judge)


def test_empty_propose_is_refused():
    with pytest.raises(ValueError, match="propose_set"):
        induce_policy([], DEV, generate=_gen([GOOD]), judge=_judge)


# ==================== 产物必须是 draft ====================

def test_induced_clause_is_draft_and_not_enforced():
    """归纳出的政策不得自动生效。"""
    r = induce_policy(PROPOSE, DEV, generate=_gen([GOOD]), judge=_judge)
    assert r.clause.status is ClauseStatus.DRAFT
    assert not r.clause.is_enforced
    assert r.clause.provenance == "generated"


def test_malformed_proposal_is_refused():
    with pytest.raises(ValueError, match="缺字段"):
        induce_policy(PROPOSE, DEV, generate=lambda _p: {"id": "N1"},
                      judge=_judge)


# ==================== prompt ====================

def test_propose_prompt_separates_pos_and_neg():
    p = build_propose_prompt(PROPOSE)
    assert "厨房里有把菜刀" in p and "桌上放着盘子" in p
    assert p.index("判定为违规") < p.index("判定为**不**违规")


def test_revise_prompt_shows_both_error_types():
    p = build_revise_prompt("禁止出现厨房场景。",
                            [DEV[1]], [DEV[0]], 0.42)
    assert "厨房在做饭" in p and "拿着一把刀" in p and "0.420" in p
