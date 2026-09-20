"""能力路由与实时可行性的测试。

两条核心:
  1. 路由按**能力**不按风险 —— "低分是因为看不懂"必须上送
  2. 实时可行性与成本是**两条独立约束** —— 算不过来钱也救不了
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.realtime import (RealtimeBudget, min_batch_for_throughput,
                          streams_to_fill_batch)
from sg2.router import CompetenceRouter, Route, compare_routing


def _r(**kw):
    return CompetenceRouter(tau=0.0, margin=0.15, **kw)


# ==================== 四象限 ====================

def test_low_score_high_ood_escalates():
    """纯风险路由永远抓不到这一格 —— 分数低正因为看不懂。"""
    assert _r().route(-1.5, ood=0.9).route is Route.ESCALATE


def test_low_score_low_ood_is_cleared():
    assert _r().route(-1.5, ood=0.02).route is Route.DECIDE_SAFE


def test_high_score_escalates_because_of_citation_gate():
    """sentinel 产不出 policy_citation,任何'不安全'判定必须经中间层。"""
    d = _r().route(1.5, ood=0.02)
    assert d.route is Route.ESCALATE and "引用" in d.reason


def test_high_score_can_self_decide_only_if_allowed():
    d = _r(can_decide_unsafe=True).route(1.5, ood=0.02)
    assert d.route is Route.DECIDE_UNSAFE


def test_near_threshold_escalates():
    assert _r().route(0.01, ood=0.02).route is Route.ESCALATE


# ==================== OOD 否决放行 ====================

@pytest.mark.parametrize("score", [-1.5, -0.5, -0.2])
def test_ood_vetoes_clearing_at_any_low_score(score):
    """回归:只按分数放行时,分布外帧因相似度低被大量放行。"""
    assert _r().route(score, ood=0.5).route is Route.ESCALATE


def test_veto_threshold_is_respected():
    r = _r(ood_safe_veto=0.8)
    assert r.route(-1.5, ood=0.5).route is Route.DECIDE_SAFE
    assert r.route(-1.5, ood=0.9).route is Route.ESCALATE


# ==================== CUSUM 报警是输入不是命令 ====================

def test_alarm_raises_voi_but_does_not_force():
    r = _r()
    a = r.route(-1.5, ood=0.0, alarmed=False)
    b = r.route(-1.5, ood=0.0, alarmed=True)
    assert b.voi > a.voi
    assert b.route is Route.DECIDE_SAFE     # 提高倾向 ≠ 强制上送


def test_forced_overrides_everything():
    d = _r().route(-1.5, ood=0.0, forced=True)
    assert d.route is Route.FORCED and d.escalated


# ==================== 与纯风险路由的对照 ====================

def test_competence_catches_blindspots_risk_misses():
    import numpy as np
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(-1, .5, 800), rng.normal(-.8, .3, 200)]
    o = np.r_[rng.uniform(0, .3, 800), rng.uniform(.7, 1., 200)]
    res = compare_routing(s, o, tau=0.0, risk_threshold=0.5)
    assert res["blindspot_caught"] > 0
    assert res["competence_escalation_rate"] > res["risk_escalation_rate"]


# ==================== 实时可行性 ====================

def test_single_stream_is_feasible_at_default():
    b = RealtimeBudget()
    assert b.feasible and all(t.headroom >= 1.0 for t in b.tiers())


def test_high_sampling_rate_reduces_stream_capacity():
    lo = RealtimeBudget(sentinel_fps=0.5).max_streams_per_gpu()
    hi = RealtimeBudget(sentinel_fps=5.0).max_streams_per_gpu()
    assert hi < lo


def test_high_escalation_rate_saturates_midtier():
    b = RealtimeBudget(escalation_rate=0.9)
    assert b.tiers()[1].utilization > RealtimeBudget(
        escalation_rate=0.05).tiers()[1].utilization


def test_infeasible_when_utilization_exceeds_one():
    """利用率 >1 时队列无界增长,排队延迟 = ∞。"""
    import math
    b = RealtimeBudget(sentinel_fps=2.0, escalation_rate=1.0, n_streams=50)
    assert not b.feasible
    assert math.isinf(b.total_added_delay_s())


def test_queue_delay_counts_toward_detection_delay():
    """系统算不过来造成的延迟,和模型看晚了是同一个量。"""
    light = RealtimeBudget(escalation_rate=0.01).total_added_delay_s()
    heavy = RealtimeBudget(escalation_rate=0.5).total_added_delay_s()
    assert heavy > light


# ==================== batching 的代价 ====================

def test_single_stream_cannot_fill_a_large_batch():
    """实测 SigLIP2 batch=32 才有 46 帧/秒,而单流 0.5fps 攒不满 ——
    只能跨流攒批,多流调度是吞吐的前提不是可选优化。"""
    assert streams_to_fill_batch(32, sentinel_fps=0.5, window_s=1.0) >= 32


def test_batch_one_needs_few_streams():
    assert streams_to_fill_batch(1, sentinel_fps=0.5, window_s=2.0) == 1


def test_min_batch_returns_one_when_already_fast_enough():
    assert min_batch_for_throughput(1.0, 0.02, 0.02) == 1
