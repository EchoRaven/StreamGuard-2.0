"""评测指标的测试。

最重要的是两条**防自欺**性质:
  1. 聚合数字会盖住弱分层 —— 必须逐层看
  2. 延迟与误报要联合报告 —— 单看一个能被另一个换来
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.buffer import RingBuffer
from sg2.eval import (OperatingPoint, StreamOutcome, cost_crossover, dominates,
                      graceful_degradation_check, operating_point,
                      pareto_curve, stratified_roc)
import numpy as np


def _o(sid, nu, tau, fa=0, cost=1.0, st=None):
    return StreamOutcome(sid, nu_s=nu, tau_s=tau, n_false_alarms=fa,
                         cost=cost, stratum=st)


# ==================== StreamOutcome ====================

def test_detection_requires_tau_after_nu():
    """ν 之前的告警不是检出 —— 内容还没出现。"""
    assert _o("a", 10.0, 15.0).detected
    assert not _o("b", 10.0, 5.0).detected
    assert not _o("c", 10.0, None).detected


def test_missed_only_for_unsafe_streams():
    assert _o("a", 10.0, None).missed
    assert not _o("b", None, None).missed


def test_delay_is_none_when_not_detected():
    assert _o("a", 10.0, None).delay_s is None
    assert _o("b", 10.0, 13.0).delay_s == pytest.approx(3.0)


# ==================== 操作点 ====================

def test_operating_point_reports_all_four():
    oc = [_o("a", 10.0, 13.0), _o("b", 10.0, None), _o("c", None, None, fa=2)]
    p = operating_point(oc, label="x", hours=1.0)
    assert p.recall == pytest.approx(0.5)
    assert p.fa_per_hour == pytest.approx(2.0)
    assert p.mean_delay_s == pytest.approx(3.0)
    assert p.cost > 0


def test_p90_delay_is_reported():
    oc = [_o(f"a{i}", 0.0, float(i)) for i in range(1, 11)]
    p = operating_point(oc, label="x", hours=1.0)
    assert p.p90_delay_s is not None and p.p90_delay_s > p.mean_delay_s


# ==================== 帕累托 ====================

def test_pareto_drops_dominated_points():
    pts = [OperatingPoint("a", 0.9, 1.0, 5.0, 8.0, cost=1.0),
           OperatingPoint("b", 0.8, 1.0, 5.0, 8.0, cost=2.0),   # 被支配
           OperatingPoint("c", 0.95, 1.0, 5.0, 8.0, cost=3.0)]
    front = [p.label for p in pareto_curve(pts)]
    assert "a" in front and "c" in front and "b" not in front


def test_dominates_needs_all_four_no_worse():
    a = OperatingPoint("a", 0.9, 1.0, 5.0, 8.0, cost=1.0)
    b = OperatingPoint("b", 0.8, 2.0, 9.0, 12.0, cost=2.0)
    assert dominates(a, b) and not dominates(b, a)


def test_better_recall_worse_cost_is_not_domination():
    """一个变好一个变差不算支配 —— 这正是必须四量联报的理由。"""
    a = OperatingPoint("a", 0.95, 1.0, 5.0, 8.0, cost=9.0)
    b = OperatingPoint("b", 0.80, 1.0, 5.0, 8.0, cost=1.0)
    assert not dominates(a, b) and not dominates(b, a)


# ==================== 分层（防自欺） ====================

def test_aggregate_hides_weak_strata():
    """聚合召回 0.7 看着还行,拆开 B/C 几乎失效。"""
    oc = ([_o(f"A{i}", 0.0, 1.0, st="A") for i in range(60)]
          + [_o(f"B{i}", 0.0, 1.0 if i < 10 else None, st="B")
             for i in range(30)]
          + [_o(f"C{i}", 0.0, None, st="C") for i in range(10)])
    agg = operating_point(oc, label="agg", hours=1.0)
    per = {r.stratum: r.recall for r in stratified_roc(oc)}
    assert agg.recall > 0.65
    assert per["C"] == 0.0 and per["B"] < 0.4
    assert agg.recall > per["B"] and agg.recall > per["C"]


def test_stratified_roc_separates_pos_and_neg():
    oc = [_o("p", 0.0, 1.0, st="A"), _o("n", None, None, fa=1, st="A")]
    r = stratified_roc(oc)[0]
    assert r.n_pos == 1 and r.n_neg == 1 and r.fpr == 1.0


# ==================== 优雅退化 ====================

def test_degradation_holds_when_above_floor():
    oc = [_o(f"a{i}", 0.0, 1.0 if i < 50 else None) for i in range(100)]
    assert graceful_degradation_check(oc, rho=0.05).holds


def test_degradation_fails_when_below_floor():
    """穿透下界 = 保底覆盖没有真正旁路,是实现 bug 的检测器。"""
    oc = [_o(f"a{i}", 0.0, None) for i in range(200)]
    assert not graceful_degradation_check(oc, rho=0.30).holds


def test_tolerance_widens_with_small_n():
    big = graceful_degradation_check(
        [_o(f"a{i}", 0.0, 1.0) for i in range(500)], rho=0.1)
    small = graceful_degradation_check(
        [_o(f"a{i}", 0.0, 1.0) for i in range(5)], rho=0.1)
    assert small.tolerance > big.tolerance


# ==================== 成本交叉点 ====================

def test_crossover_shrinks_when_audit_is_charged():
    """审计是保证的价格,入账后能赢的升级率上界必然变小。"""
    no_audit = cost_crossover(1.0, 0.05, 0.3, 3.0)
    with_audit = cost_crossover(1.0, 0.05, 0.3, 3.0,
                                c_audit=2.0, audit_rate=0.01)
    assert with_audit < no_audit


def test_crossover_nonpositive_means_premise_fails():
    """sentinel 本身就比基线贵时,任何升级率都赢不了。"""
    assert cost_crossover(0.1, 0.5, 0.3, 3.0) <= 0


# ==================== Ring buffer ====================

def _img(n=32):
    return np.zeros((n, n, 3), np.float32)


def test_buffer_evicts_by_time():
    b = RingBuffer(window_s=5.0, max_bytes=10 ** 9)
    for i in range(200):
        b.append(i * 0.1, _img())
    assert b.span_s <= 5.0 + 1e-6 and b.stats["dropped_time"] > 0


def test_buffer_evicts_by_bytes():
    """只按帧数限制不够 —— 分辨率一变占用差几十倍。"""
    b = RingBuffer(window_s=10 ** 6, max_bytes=256 * 1024)
    for i in range(200):
        b.append(i * 0.1, _img())
    assert b.stats["dropped_bytes"] > 0 and b.stats["mb"] <= 0.26


def test_buffer_rejects_out_of_order():
    b = RingBuffer()
    b.append(5.0, _img())
    with pytest.raises(ValueError, match="时间戳回退"):
        b.append(4.0, _img())


def test_covers_is_false_after_rolloff():
    """滚出去的取不回来 —— 那是硬 miss 不是延迟。"""
    b = RingBuffer(window_s=2.0)
    for i in range(100):
        b.append(i * 0.1, _img())
    assert b.covers(9.5) and not b.covers(0.5)


def test_resample_hits_target_rate():
    b = RingBuffer(window_s=100.0)
    for i in range(300):
        b.append(i * 0.1, _img())      # 10 fps 原生
    got = b.resample(10.0, 20.0, fps=2.0)
    assert 18 <= len(got) <= 23        # 约 21 帧
    assert all(10.0 <= f.t_s <= 20.0 for f in got)


def test_rewind_empty_outside_window():
    b = RingBuffer(window_s=1.0)
    for i in range(50):
        b.append(i * 0.1, _img())
    assert b.rewind(0.0, 0.5) == []


def test_resample_never_repeats_a_frame():
    """回归:目标帧率高于原生帧率时,同一帧会被**非相邻地**重复取到。
    原实现只查 out[-1],漏掉这种情况 —— 实测 burst 4fps 对 0.5fps 的流,
    某一帧被取了两次,窗口里出现重复时间戳。
    """
    b = RingBuffer(window_s=100.0)
    for i in range(10):
        b.append(i * 2.0, _img())          # 0.5 fps 原生
    got = b.resample(0.0, 18.0, fps=4.0)   # 目标远高于原生
    ts = [f.t_s for f in got]
    assert len(ts) == len(set(ts)), f"出现重复时间戳: {ts}"
