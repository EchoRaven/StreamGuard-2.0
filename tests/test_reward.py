"""RL 奖励的反钻空子测试。

docs/08_SFT_RL.md §3.3 列了五条钻法,这里逐条锁死。奖励能被钻空子是
RL 跑砸的头号原因,而且钻成功时训练曲线往往看起来很正常。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.train.reward import (Episode, RewardConfig, Step,
                              assert_not_degenerate, compute_reward,
                              fa_exchange_rate,
                              flag_breakeven_precision)

NU, CAT = 8.0, "C1"


def flag(cat=CAT, cite="P1"):
    return json.dumps({"action": "flag", "category": cat,
                       "policy_citation": cite})


HOLD = json.dumps({"action": "hold"})
CLEAR = json.dumps({"action": "clear"})


def ep(*steps, nu=NU, cat=CAT):
    return Episode(steps=tuple(steps), nu_s=nu, category=cat)


def R(e, cfg=None):
    return compute_reward(e, cfg).total


# ---------- 五条钻法 ----------

def test_flag_everything_scores_worse():
    """钻法 1:一开始就全 flag。误报惩罚必须让它不划算。"""
    spam = ep(*[Step(t, flag(), 50) for t in (1., 3., 5., 7., 10.)])
    good = ep(Step(10.0, flag(), 50))
    assert R(spam) < R(good)


def test_flag_without_citation_is_zero():
    """钻法 2:flag 但不给引用。引用门直接归零。"""
    e = ep(Step(10.0, json.dumps({"action": "flag", "category": CAT}), 50))
    b = compute_reward(e)
    assert b.total == 0.0 and b.gate_failed == "引用门"


def test_wrong_citation_counts_as_false_alarm():
    """钻法 3:引用随便填。类别门把它计为误报,且不算命中。"""
    b = compute_reward(ep(Step(10.0, flag(cat="C9"), 50)))
    assert not b.hit and b.n_false_alarms == 1 and b.total < 0


def test_never_flag_scores_worst():
    """钻法 4(最危险):永远 hold。必须严格差于正确 flag。"""
    silent = ep(Step(10.0, HOLD, 50), Step(12.0, HOLD, 50))
    good = ep(Step(10.0, flag(), 50))
    assert R(silent) < R(good)


def test_token_penalty_applies():
    """钻法 5:灌长推理。同样结果下 token 多的必须更差。"""
    short = ep(Step(10.0, flag(), 50))
    longer = ep(Step(10.0, flag(), 5000))
    assert R(longer) < R(short)


# ---------- 格式门 ----------

@pytest.mark.parametrize("raw", ["yes it is unsafe", "", "{broken",
                                 json.dumps({"action": "escalate"}),
                                 json.dumps(["flag"])])
def test_malformed_output_is_zero(raw):
    b = compute_reward(ep(Step(10.0, raw, 50)))
    assert b.total == 0.0 and b.gate_failed == "格式门"


# ---------- 延迟 ----------

def test_earlier_correct_flag_scores_higher():
    early = ep(Step(9.0, flag(), 50))
    late = ep(Step(20.0, flag(), 50))
    assert R(early) > R(late)


def test_flag_before_nu_is_false_alarm_not_hit():
    """ν 之前的 flag 不可能是命中 —— 内容还没出现。"""
    b = compute_reward(ep(Step(5.0, flag(), 50)))
    assert not b.hit and b.n_false_alarms == 1


def test_miss_is_worse_than_one_false_alarm():
    """漏报必须比一次误报更糟,否则模型学会用沉默换安全。"""
    miss = ep(Step(10.0, HOLD, 50))
    one_fa = ep(Step(5.0, flag(), 50), Step(10.0, flag(), 50))
    assert R(miss) < R(one_fa)


def test_safe_stream_flag_is_pure_false_alarm():
    b = compute_reward(Episode((Step(10.0, flag(), 50),), nu_s=None))
    assert not b.hit and b.n_false_alarms == 1 and b.total < 0


def test_safe_stream_clear_is_best():
    clear = Episode((Step(10.0, CLEAR, 50),), nu_s=None)
    flagged = Episode((Step(10.0, flag(), 50),), nu_s=None)
    assert R(clear) > R(flagged)


# ---------- 结构性约束 ----------

def test_beta_ge_gamma_is_refused_at_construction():
    """beta >= gamma 会让沉默成为最优。构造时就拒绝,不等训练跑完才发现。"""
    with pytest.raises(ValueError, match="退化|beta_fa"):
        RewardConfig(beta_fa=1.0, gamma_hit=1.0)


def test_default_config_is_not_degenerate():
    assert_not_degenerate()


def test_degenerate_config_is_detected():
    """误报惩罚过大 -> 盈亏平衡精度过高 -> 模型会学成沉默。"""
    bad = RewardConfig(gamma_hit=1.0, beta_fa=0.9, alpha_delay=0.1)
    assert flag_breakeven_precision(bad) > 0.3
    with pytest.raises(ValueError, match="退化|盈亏平衡"):
        assert_not_degenerate(bad)


def test_breakeven_precision_is_negative_for_default():
    """默认配置下 q* <= 0:任何精度的 flag 都优于沉默。"""
    assert flag_breakeven_precision() <= 0.0


def test_breakeven_rises_with_fa_penalty():
    q_low = flag_breakeven_precision(RewardConfig(beta_fa=0.1))
    q_high = flag_breakeven_precision(RewardConfig(beta_fa=0.9,
                                                   alpha_delay=0.1))
    assert q_high > q_low


def test_fa_exchange_rate_below_one():
    """一次误报应少于一次正确检测的价值,否则沉默更划算。"""
    assert fa_exchange_rate() < 1.0


def test_first_correct_flag_wins_not_last():
    """多次 flag 时,延迟应按**第一次正确的**算,后续计为误报。"""
    b = compute_reward(ep(Step(9.0, flag(), 50), Step(15.0, flag(), 50)))
    assert b.hit and b.delay_s == pytest.approx(1.0) and b.n_false_alarms == 1
