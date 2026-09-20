"""SafeLens 基线反推的回归测试。这些数字会进论文对比栏,必须钉住。"""
import pytest

from sg2.aci import abstention_floor
from sg2.baselines import SAFELENS, SafeLensReported


def test_implied_escalation_rate():
    """1.76 = 0.04 + r*5.02 -> 34.3%。原文没给这个数。"""
    assert SAFELENS.implied_escalation_rate == pytest.approx(0.3426, abs=1e-3)


def test_overall_below_s2_rules_out_parallel():
    """串行反推的前提:整体延迟必须低于 S2 单独,否则是并行不是级联。"""
    assert SAFELENS.overall_latency_s < SAFELENS.s2_latency_s


def test_inconsistent_numbers_raise_not_silently_return():
    """三个数不自洽时必须抛异常 —— 静默返回 r>1 会流进论文表格。"""
    bad = SafeLensReported(s1_latency_s=0.04, s2_latency_s=1.0,
                           overall_latency_s=1.76)
    with pytest.raises(ValueError, match="不在"):
        _ = bad.implied_escalation_rate


def test_escalation_above_kotte_floor():
    """34.3% 必须在 Kotte 下界之上,否则其报告数字互相矛盾。"""
    for alpha in (0.05, 0.10, 0.20):
        assert SAFELENS.implied_escalation_rate >= abstention_floor(
            SAFELENS.base_risk, alpha)


@pytest.mark.parametrize("fps", [0.5, 1.0, 2.0])
def test_s2_exceeds_every_streaming_deadline(fps):
    """5.02s 的 CoT 超过所有流式截止期 —— 不可迁移的核心论据。"""
    assert not SAFELENS.slow_tier_feasible(fps)


def test_whole_system_infeasible_at_1fps_and_above():
    """即使升级率为 0,整体 1.76s 在 ≥1fps 也排不开。"""
    assert SAFELENS.deadline_feasible(0.5)
    assert not SAFELENS.deadline_feasible(1.0)
    assert not SAFELENS.deadline_feasible(2.0)


def test_escalation_far_above_our_cost_crossover():
    assert SAFELENS.implied_escalation_rate > 10 * 0.0253


def test_per_category_weakest_are_temporal_ones():
    """Abuse / Violence 是最弱两类 —— 恰是最依赖时序上下文的。"""
    ranked = sorted(SAFELENS.per_category_acc, key=lambda kv: kv[1])
    assert {ranked[0][0], ranked[1][0]} == {"C2_abuse", "C3_violence"}
