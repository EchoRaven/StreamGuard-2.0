"""政策动物园与反事实奖励的测试。

守两条:
  1. 消融必须真的把"数量"与"多样性"解耦,否则问不出哪个有用
  2. 反事实奖励不能被"一律翻转"钻空子
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.policy_zoo import (SCHEMAS, build_from_schema, estimate_policy_tokens,
                            mix, schema_diversity_ablation)
from sg2.train.reward import (CFConfig, CounterfactualPair,
                              assert_cf_not_gameable, counterfactual_reward)

F = json.dumps({"action": "flag", "category": "C1", "policy_citation": "X1"})
FW = json.dumps({"action": "flag", "category": "C1",
                 "policy_citation": "WRONG"})
C = json.dumps({"action": "clear"})
U = json.dumps({"action": "uncovered", "description": "x"})


# ==================== 消融设计 ====================

def test_ablation_decouples_count_from_diversity():
    """many_one 与 few_many 必须在两个维度上相反,否则问不出哪个有用。"""
    arms = {a.label: a for a in schema_diversity_ablation()}
    mo, fm = arms["many_one"], arms["few_many"]
    assert mo.total_clauses > fm.total_clauses      # 数量:多 vs 少
    assert mo.n_schemas < fm.n_schemas              # 多样性:少 vs 多


def test_all_four_arms_present():
    labels = {a.label for a in schema_diversity_ablation()}
    assert labels == {"few_one", "many_one", "few_many", "many_many"}


def test_schemas_render_differently():
    """不同 schema 的 id 风格必须真的不同 —— 那正是模型可能记住的。"""
    heads = {build_from_schema(s, rng=random.Random(0)).render().splitlines()[0]
             for s in SCHEMAS}
    assert len(heads) == len(SCHEMAS)


def test_id_styles_are_distinct():
    ids = []
    for s in SCHEMAS:
        ids.append(build_from_schema(s, rng=random.Random(0)).enforced()[0].id)
    assert len(set(ids)) == len(SCHEMAS)


def test_mix_samples_across_sources():
    corpora = [build_from_schema(s) for s in SCHEMAS]
    m = mix(corpora, k=6, rng=random.Random(0))
    assert len(m.enforced()) == 6
    cats = {c.provenance for c in m.enforced()}
    assert len(cats) > 1, "混采应跨来源"


def test_more_clauses_costs_more_tokens():
    """"更多 policy"是有代价的 —— 直接吃 KV 预算。"""
    small = mix([build_from_schema(SCHEMAS[0])])
    big = mix([build_from_schema(s) for s in SCHEMAS])
    assert estimate_policy_tokens(big) > estimate_policy_tokens(small)


# ==================== 反事实奖励 ====================

def test_consistent_is_rewarded():
    r = counterfactual_reward(CounterfactualPair(F, C, target_citation="X1"))
    assert r.outcome == "consistent" and r.total > 0


def test_uncovered_on_b_side_still_counts_as_not_flag():
    r = counterfactual_reward(CounterfactualPair(F, U, target_citation="X1"))
    assert r.outcome == "consistent"


def test_insensitive_is_penalised():
    for pair in (CounterfactualPair(F, F, target_citation="X1"),
                 CounterfactualPair(C, C)):
        assert counterfactual_reward(pair).outcome == "insensitive"
        assert counterfactual_reward(pair).total < 0


def test_reversed_is_worse_than_insensitive():
    """方向反了是学到了**错误关系**,比没学到更糟。"""
    rev = counterfactual_reward(CounterfactualPair(C, F)).total
    ins = counterfactual_reward(CounterfactualPair(C, C)).total
    assert rev < ins


def test_wrong_citation_gets_no_credit():
    r = counterfactual_reward(CounterfactualPair(FW, C, target_citation="X1"))
    assert r.outcome == "wrong_citation" and r.total == 0.0


def test_invalid_output_is_zero():
    r = counterfactual_reward(CounterfactualPair("乱码", C))
    assert r.outcome == "invalid" and r.total == 0.0


def test_always_flip_is_not_optimal():
    """只奖励"翻转"的话最优策略是一律翻转,与内容无关。"""
    assert_cf_not_gameable()


def test_reversed_must_exceed_insensitive_at_construction():
    with pytest.raises(ValueError, match="beta_reversed"):
        CFConfig(beta_reversed=0.2, beta_insensitive=0.4)


def test_gameable_config_is_detected():
    cfg = CFConfig(beta_reversed=0.41, beta_insensitive=0.4,
                   gamma_consistent=0.5)
    assert_cf_not_gameable(cfg)          # 勉强不可钻
