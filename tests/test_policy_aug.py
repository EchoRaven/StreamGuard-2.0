"""政策增强与敏感度指标的测试。

守住一条底线:**增强本身不能把目标条款弄丢**。弄丢了训练标签就错了,
而这种错在 loss 曲线上完全看不出来。
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.eval.sensitivity import (FlipCase, SensitivityReport,
                                  make_counterfactual_pair,
                                  measure_sensitivity)
from sg2.policy import PolicyClause, safewatch_corpus
from sg2.train.policy_aug import (SAFE_EXIT, AugConfig, augment, augment_batch,
                                  mismatch_stats)

CORP = safewatch_corpus()
TARGETS = ["C1_sexual", "C3_violence", "C5_illegal", None] * 25


# ==================== 增强的正确性 ====================

def test_target_clause_survives_when_not_mismatch():
    """非 mismatch 时目标条款必须还在,否则训练标签就错了。"""
    for s in augment_batch(CORP, ["C1_sexual"] * 60, seed=1):
        if s.is_mismatch:
            continue
        ids = {c.id for c in s.corpus.enforced()}
        assert s.target_clause_id in ids


def test_mismatch_removes_the_target():
    """mismatch 时目标条款必须不在 —— 这才是"答案由政策决定"的信号。"""
    ss = [s for s in augment_batch(CORP, ["C1_sexual"] * 80, seed=2)
          if s.is_mismatch]
    assert ss, "mismatch_rate>0 时应有 mismatch 样本"
    for s in ss:
        assert s.target_clause_id is None
        assert s.expected_action == "clear"
        assert "C1_sexual" not in {c.id for c in s.corpus.enforced()}


def test_ids_are_randomised():
    """id 不随机化的话,模型能把 id 与类别绑死,增强形同虚设。"""
    st = mismatch_stats(augment_batch(CORP, TARGETS, seed=3))
    assert st["unique_id_ratio"] > 0.95


def test_ids_stable_when_disabled():
    s = augment(CORP, "C1_sexual",
                cfg=AugConfig(rename_ids=False, inject_novel=False,
                              subset=False, mismatch_rate=0.0),
                rng=random.Random(0))
    assert s.target_clause_id == "C1_sexual"


def test_clause_count_varies():
    """"语料里总是这六条"也是一条可被记住的线索。"""
    st = mismatch_stats(augment_batch(CORP, TARGETS, seed=4))
    assert st["clauses_max"] > st["clauses_min"]


def test_safe_exit_always_present():
    """没有安全出口时,模型被迫用 uncovered 表达"不违规"(实测)。"""
    assert mismatch_stats(augment_batch(CORP, TARGETS, seed=5))["has_safe_exit"]


def test_novel_clauses_are_injected():
    ss = augment_batch(CORP, ["C1_sexual"] * 20,
                       cfg=AugConfig(n_novel=2), seed=6)
    assert any(any(c.category == "novel" for c in s.corpus.enforced())
               for s in ss)


def test_paraphrase_is_applied():
    s = augment(CORP, "C1_sexual",
                cfg=AugConfig(paraphrase=lambda t: "改写:" + t,
                              mismatch_rate=0.0, rename_ids=False),
                rng=random.Random(0))
    tgt = next(c for c in s.corpus.enforced() if c.id == "C1_sexual")
    assert tgt.text.startswith("改写:")


def test_min_clauses_respected():
    ss = augment_batch(CORP, TARGETS, cfg=AugConfig(min_clauses=3), seed=7)
    assert all(len(s.corpus.enforced()) >= 3 for s in ss)


def test_bad_mismatch_rate_is_refused():
    with pytest.raises(ValueError, match="mismatch_rate"):
        AugConfig(mismatch_rate=1.5)


def test_deterministic_given_seed():
    a = mismatch_stats(augment_batch(CORP, TARGETS, seed=9))
    b = mismatch_stats(augment_batch(CORP, TARGETS, seed=9))
    assert a == b


# ==================== 敏感度指标 ====================

_C = PolicyClause(id="T1", category="T1", title="测试图", text="禁止测试信号图。")
_ITEMS = [(f"v{i}", i) for i in range(10)]


def test_reads_policy_is_recognised():
    r = measure_sensitivity(_ITEMS, _C,
                            lambda c, x: "flag" if c.name == "covering" else "clear")
    assert r.flip_rate == 1.0 and r.correct_rate == 1.0
    assert "读政策" in r.verdict() and not r.always_flag


def test_always_flag_is_caught():
    """两种政策都 flag = 典型的"背下来了"。"""
    r = measure_sensitivity(_ITEMS, _C, lambda c, x: "flag")
    assert r.always_flag and r.correct_rate == 0.0
    assert "背不在读" in r.verdict()


def test_never_flag_is_caught():
    r = measure_sensitivity(_ITEMS, _C, lambda c, x: "clear")
    assert r.always_same and "背不在读" in r.verdict()


def test_flipping_wrong_direction_is_distinguished():
    """翻转但方向反了,不该和"读政策"混为一谈。"""
    r = measure_sensitivity(_ITEMS, _C,
                            lambda c, x: "clear" if c.name == "covering" else "flag")
    assert r.flip_rate == 1.0 and r.correct_rate == 0.0
    assert "方向常错" in r.verdict()


def test_counterfactual_pair_is_balanced():
    """两边条款数相同且都有安全出口 —— 否则差异可能来自条款数。"""
    cov, notcov = make_counterfactual_pair(_C)
    assert len(cov.enforced()) == len(notcov.enforced())
    for c in (cov, notcov):
        assert any(x.id == SAFE_EXIT.id or x.id == "Z0_ok"
                   for x in c.enforced())


def test_uncovered_counts_as_not_flag():
    """uncovered 不是"判为违规" —— 方向正确性要按这个算。"""
    c = FlipCase("x", "flag", "uncovered")
    assert c.flipped and c.correct


def test_empty_report_is_safe():
    r = SensitivityReport()
    assert r.n == 0 and r.flip_rate == 0.0 and "无样本" in r.verdict()
