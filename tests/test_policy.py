"""政策语料 / out-of-policy / 政策替换的测试。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.outofpolicy import (PolicyGapTracker, UncoveredCase,
                             build_proposal_prompt, draft_clause_from,
                             swap_policy)
from sg2.policy import (SAFEWATCH_CATEGORIES, ClauseStatus, PolicyClause,
                        PolicyCorpus, safewatch_corpus)
from sg2.train.reward import Episode, RewardConfig, Step, compute_reward

FLAG = json.dumps({"action": "flag", "category": "C1", "policy_citation": "P1"})
UNC = json.dumps({"action": "uncovered", "description": "..."})
HOLD = json.dumps({"action": "hold"})
CLEAR = json.dumps({"action": "clear"})


# ==================== 语料 ====================

def test_safewatch_has_six_categories():
    """SafeWatch-Bench 六类(ICLR 2025, arXiv:2412.06878)。"""
    assert len(SAFEWATCH_CATEGORIES) == 6
    assert safewatch_corpus().categories() == set(SAFEWATCH_CATEGORIES)


def test_fingerprint_ignores_clause_order():
    """换顺序缓解位置偏置时,不应被当成政策变更而触发重新校准。"""
    c = safewatch_corpus()
    f = c.fingerprint()
    c.clauses.reverse()
    assert c.fingerprint() == f


def test_fingerprint_changes_with_text():
    c = safewatch_corpus()
    f = c.fingerprint()
    c.clauses[0].text = "改过的判定标准"
    assert c.fingerprint() != f


def test_render_shuffle_changes_order_not_content():
    c = safewatch_corpus()
    a, b = c.render(), c.render(shuffle_seed=7)
    assert a != b
    assert set(a.split("\n")) != set(), "渲染不应为空"
    for cl in c.enforced():
        assert cl.id in a and cl.id in b


def test_draft_clauses_are_not_enforced():
    c = safewatch_corpus()
    c.add(PolicyClause(id="X1", category="X1", title="t", text="x",
                       status=ClauseStatus.DRAFT, provenance="generated"))
    assert not c.covers("X1")
    assert "X1" not in c.render()


def test_retired_clause_keeps_history_but_not_enforced():
    c = safewatch_corpus()
    new = c.amend("C1_sexual", text="更严格的标准")
    assert c.by_id("C1_sexual").status is ClauseStatus.RETIRED
    assert new.supersedes == "C1_sexual" and new.is_enforced
    assert c.by_id("C1_sexual") is not None, "历史条款必须保留以便复现旧判决"


def test_amend_does_not_mutate_old_text():
    """就地改文本会让引用旧 id 的历史判决无法复现。"""
    c = safewatch_corpus()
    old_text = c.by_id("C1_sexual").text
    c.amend("C1_sexual", text="新标准")
    assert c.by_id("C1_sexual").text == old_text


def test_duplicate_id_is_refused():
    c = safewatch_corpus()
    with pytest.raises(ValueError, match="已存在"):
        c.add(PolicyClause(id="C1_sexual", category="C1", title="t", text="x"))


def test_valid_citation_rejects_draft_and_unknown():
    c = safewatch_corpus()
    c.add(PolicyClause(id="X1", category="X1", title="t", text="x",
                       status=ClauseStatus.DRAFT, provenance="generated"))
    assert c.valid_citation("C1_sexual")
    assert not c.valid_citation("X1")
    assert not c.valid_citation("nope")


def test_corpus_roundtrip(tmp_path):
    c = safewatch_corpus()
    p = c.save(tmp_path / "p.yaml")
    assert PolicyCorpus.load(p).fingerprint() == c.fingerprint()


# ==================== 生成条款必须人工批准 ====================

def test_generated_clause_cannot_be_active_at_construction():
    with pytest.raises(ValueError, match="人工复核|active"):
        PolicyClause(id="X1", category="X1", title="t", text="x",
                     provenance="generated", status=ClauseStatus.ACTIVE)


def test_draft_from_proposal_is_draft():
    gap = PolicyGapTracker(safewatch_corpus()).gaps()
    c = draft_clause_from(
        {"id": "X1", "category": "X1", "title": "t", "text": "x"},
        type("G", (), {"cases": []})())
    assert c.status is ClauseStatus.DRAFT and c.provenance == "generated"


def test_approve_requires_reviewer():
    c = safewatch_corpus()
    c.add(PolicyClause(id="X1", category="X1", title="t", text="x",
                       status=ClauseStatus.DRAFT, provenance="generated"))
    with pytest.raises(ValueError, match="复核人"):
        c.approve("X1", reviewer="")
    c.approve("X1", reviewer="alice")
    assert c.covers("X1")


def test_cannot_approve_non_draft():
    c = safewatch_corpus()
    with pytest.raises(ValueError, match="只有 draft"):
        c.approve("C1_sexual", reviewer="alice")


# ==================== 缺口追踪 ====================

def _tracker(n=6, cat="X_child"):
    t = PolicyGapTracker(safewatch_corpus())
    for i in range(n):
        t.record(UncoveredCase(f"v{i}", float(i), f"案例{i}", cat))
    return t


def test_covered_category_is_not_a_gap():
    """有条款却说未覆盖 = 模型该引用没引,不是政策缺口。"""
    t = PolicyGapTracker(safewatch_corpus())
    assert t.record(UncoveredCase("v", 1.0, "暴力", "C3_violence")) is False
    assert t.stats["cases"] == 0


def test_single_case_is_not_actionable():
    """单点很可能是误判,不足以提议改政策。"""
    assert _tracker(n=1).actionable_gaps() == []


def test_enough_cases_become_actionable():
    assert len(_tracker(n=6).actionable_gaps()) == 1


def test_proposal_prompt_lists_existing_and_cases():
    t = _tracker()
    p = build_proposal_prompt(t.actionable_gaps()[0], t.corpus)
    assert "C1_sexual" in p and "案例0" in p


# ==================== 政策替换 ====================

def test_content_change_requires_recalibration():
    a = safewatch_corpus()
    b = PolicyCorpus.from_dict(a.to_dict())
    b.add(PolicyClause(id="X1", category="X1", title="t", text="x"))
    assert swap_policy(a, b, reason="补条款").requires_recalibration


def test_reorder_does_not_require_recalibration():
    """只换顺序判定范围没变,不该触发重校准。"""
    a = safewatch_corpus()
    b = PolicyCorpus.from_dict(a.to_dict())
    b.clauses.reverse()
    assert not swap_policy(a, b, reason="缓解位置偏置").requires_recalibration


# ==================== uncovered 的奖励 ====================

def _ep(raw, *, nu=8.0, cat="C1", covered=True):
    return Episode((Step(10.0, raw, 50),), nu_s=nu, category=cat,
                   covered=covered)


def test_uncovered_ordering_is_correct():
    """七种情形的相对顺序,是整个 out-of-policy 设计的核心。"""
    r = {
        "flag_ok": compute_reward(_ep(FLAG)).total,
        "unc_right": compute_reward(_ep(UNC, cat="X1", covered=False)).total,
        "safe_clear": compute_reward(Episode((Step(10., CLEAR, 50),), nu_s=None)).total,
        "safe_unc": compute_reward(Episode((Step(10., UNC, 50),), nu_s=None)).total,
        "unc_silent": compute_reward(_ep(HOLD, cat="X1", covered=False)).total,
        "fabricate": compute_reward(_ep(FLAG, cat="X1", covered=False)).total,
        "unc_lazy": compute_reward(_ep(UNC)).total,
    }
    assert r["flag_ok"] > r["unc_right"] > r["safe_clear"] > r["safe_unc"]
    assert r["safe_unc"] > r["unc_silent"] > r["fabricate"] > r["unc_lazy"]


def test_uncovered_is_not_penalised_as_a_miss():
    """未覆盖流上正确指出缺口,不该再按漏报罚。"""
    right = compute_reward(_ep(UNC, cat="X1", covered=False))
    silent = compute_reward(_ep(HOLD, cat="X1", covered=False))
    assert right.total > 0 > silent.total


def test_uncovered_passes_citation_gate():
    """uncovered 不要求引用 —— 正因为无条款可引才走这条。"""
    assert compute_reward(_ep(UNC, cat="X1", covered=False)).gate_failed is None


def test_uncovered_reward_must_be_below_hit():
    with pytest.raises(ValueError, match="一律说未覆盖"):
        RewardConfig(gamma_uncovered_right=1.2, gamma_hit=1.0)


# ==================== 引用解析（真实模型实测的三种错法） ====================

@pytest.mark.parametrize("cid,expect,why", [
    ("C1_sexual", "C1_sexual", "exact"),
    ("C1", "C1_sexual", "prefix"),          # 模型截断成前缀
    ("1", None, "ordinal"),                 # 抄了行首序号
    ("C", None, "ambiguous_prefix"),        # 宽容要有边界
    ("nope", None, "unknown"),
    ("", None, "empty"),
    (None, None, "empty"),
])
def test_resolve_citation(cid, expect, why):
    cl, reason = safewatch_corpus().resolve_citation(cid)
    assert (cl.id if cl else None) == expect
    assert reason == why


def test_render_has_no_ordinals_by_default():
    """实测 Qwen3-VL 会把行首序号 '1.' 当成条款 id 抄回来。"""
    r = safewatch_corpus().render()
    assert not any(line.strip().startswith(("1.", "2."))
                   for line in r.split("\n"))


def test_render_does_not_bracket_ids():
    """方括号是渲染格式,实测会被模型连同 id 一起抄进 citation。"""
    assert "[C1_sexual]" not in safewatch_corpus().render()


def test_draft_clause_cannot_be_cited():
    c = safewatch_corpus()
    c.add(PolicyClause(id="X1", category="X1", title="t", text="x",
                       status=ClauseStatus.DRAFT, provenance="generated"))
    cl, why = c.resolve_citation("X1")
    assert cl is None and why == "not_enforced"


# ==================== 渲染格式泄漏：实测四种抄法 ====================

@pytest.mark.parametrize("raw,expect", [
    ("[C1_sexual]", "C1_sexual"),          # 渲染用方括号包 id
    ("id=C1_sexual", "C1_sexual"),         # 渲染写 "id=xxx",连键名抄走
    ("id: C1_sexual", "C1_sexual"),
    ("条款=C1_sexual", "C1_sexual"),
    (" C1_sexual ", "C1_sexual"),
    ("C1_sexual", "C1_sexual"),
])
def test_render_prefix_leaks_are_normalized(raw, expect):
    """四种抄法全是**渲染格式泄漏**,不是模型不听话。

    实测 Qwen3-VL-8B 在匹配条款下判对了内容、引对了条款,却输出
    "id=T1_testpattern" —— 因为渲染里就写着 "id=xxx"。
    """
    from sg2.stream.protocol import normalize_citation
    assert normalize_citation(raw) == expect


def test_id_prefix_leak_resolves_to_real_clause():
    from sg2.stream.protocol import normalize_citation
    c = safewatch_corpus()
    cl, why = c.resolve_citation(normalize_citation("id=C3_violence"))
    assert cl is not None and cl.id == "C3_violence" and why == "exact"
