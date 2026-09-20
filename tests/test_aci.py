"""ACI 在线校准的测试。

三条最容易写错、且写错时**不报任何错**的性质:
  1. 更新方向(符号写反 -> 错误率低反而放宽,高反而收紧)
  2. 无反馈时不更新(把"没反馈"当"没漏报" -> 阈值单调收紧)
  3. 饱和检测(撞死边界后自适应失效,系统照跑)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.aci import AdaptiveConformal, DtACI, audit_signal_cost


def _run(a, rate, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a.update(bool(rng.random() < rate))
    return a


# ==================== 方向 ====================

def test_miss_loosens_and_hit_tightens():
    """err=1 应上调 alpha_t(放宽),err=0 应下调。写反时不报错。"""
    a = AdaptiveConformal(alpha=0.05, gamma=0.01)
    base = a.alpha_t
    a.update(True)
    assert a.alpha_t > base, "漏报后应放宽"
    b = AdaptiveConformal(alpha=0.05, gamma=0.01)
    base = b.alpha_t
    b.update(False)
    assert b.alpha_t < base, "未漏报应收紧"


def test_low_error_rate_drives_alpha_down():
    a = _run(AdaptiveConformal(alpha=0.05, gamma=0.002), rate=0.01)
    assert a.alpha_t < 0.05


def test_high_error_rate_drives_alpha_up():
    a = _run(AdaptiveConformal(alpha=0.05, gamma=0.002), rate=0.20)
    assert a.alpha_t > 0.05


# ==================== 无反馈 ====================

def test_none_does_not_update():
    """绝大多数 tick 没有审计信号。把它当成'没漏报'会让阈值单调收紧。"""
    a = AdaptiveConformal(alpha=0.05, gamma=0.02)
    before = a.alpha_t
    for _ in range(500):
        a.update(None)
    assert a.alpha_t == before
    assert a.stats["fed"] == 0 and a.stats["t"] == 500


def test_sparse_feedback_still_tracks():
    rng = np.random.default_rng(2)
    a = AdaptiveConformal(alpha=0.05, gamma=0.02)
    for _ in range(20000):
        a.update(bool(rng.random() < 0.05) if rng.random() < 0.01 else None)
    assert 0 < a.stats["fed"] < 20000
    assert a.coverage_gap < 0.05


# ==================== 饱和 ====================

def test_saturation_is_detected_when_gamma_too_large():
    """撞死边界后自适应失效,而系统照跑不报错。"""
    a = _run(AdaptiveConformal(alpha=0.05, gamma=0.32), rate=0.50, n=2000)
    assert a.saturation > 0.0


def test_well_matched_gamma_does_not_saturate():
    """默认 gamma 在目标附近的漂移下应零饱和 —— 这是默认值的依据。"""
    rng = np.random.default_rng(0)
    a = AdaptiveConformal(alpha=0.05)
    for t in range(3200):
        a.update(bool(rng.random() < (0.03 if (t // 400) % 2 == 0 else 0.08)))
    assert a.saturation == 0.0, f"默认 gamma 饱和了 {a.saturation:.2f}"
    assert 0.01 < a.alpha_t < 0.20


def test_clip_is_respected():
    a = _run(AdaptiveConformal(alpha=0.05, gamma=0.3, clip=(0.01, 0.2)),
             rate=1.0, n=500)
    assert 0.01 <= a.alpha_t <= 0.2


# ==================== 界与统计 ====================

def test_bound_shrinks_with_more_feedback():
    a = AdaptiveConformal(alpha=0.05, gamma=0.02)
    a.update(True)
    early = a.coverage_bound()
    _run(a, rate=0.05, n=2000)
    assert a.coverage_bound() < early


def test_warm_start_sets_alpha():
    a = AdaptiveConformal(alpha=0.05)
    a.warm_start(0.11)
    assert a.alpha_t == pytest.approx(0.11)


def test_warm_start_respects_clip():
    a = AdaptiveConformal(alpha=0.05, clip=(0.01, 0.2))
    a.warm_start(0.9)
    assert a.alpha_t == 0.2


# ==================== DtACI ====================

def test_dtaci_weights_sum_to_one():
    d = _run(DtACI(alpha=0.05), rate=0.1, n=500)
    assert sum(d.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_dtaci_tracks_without_picking_gamma():
    """合适的 gamma 取决于漂移速率,而漂移速率事先不知道。"""
    rng = np.random.default_rng(0)
    d = DtACI(alpha=0.05)
    for t in range(3200):
        d.update(bool(rng.random() < (0.03 if (t // 400) % 2 == 0 else 0.08)))
    assert abs(d.empirical_error - 0.055) < 0.03


def test_dtaci_ignores_none():
    d = DtACI(alpha=0.05)
    before = d.alpha_t
    for _ in range(100):
        d.update(None)
    assert d.alpha_t == pytest.approx(before)


# ==================== 审计价格 ====================

def test_audit_cost_matches_documented_2000():
    """eps=1%, alpha=5% -> 2000 条未升级内容换 1 个漏报信号。"""
    assert audit_signal_cost(0.01, 0.05) == pytest.approx(2000.0)


def test_audit_cost_rejects_bad_rates():
    for bad in ((0.0, 0.05), (0.01, 0.0), (1.5, 0.05)):
        with pytest.raises(ValueError):
            audit_signal_cost(*bad)


# ==================== 构造校验 ====================

@pytest.mark.parametrize("kw", [{"alpha": 0.0}, {"alpha": 1.0},
                                {"gamma": 0.0}, {"clip": (0.5, 0.1)}])
def test_bad_config_is_refused(kw):
    with pytest.raises(ValueError):
        AdaptiveConformal(**kw)
