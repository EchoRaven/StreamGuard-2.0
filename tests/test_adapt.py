"""少样本边界与校准状态的回归测试。

最重要的一条:边界一动,保证必须失效。这是 docs/03_ADAPTATION.md §4 的
核心不变量,静默保留旧阈值会让整个 conformal 层变成装饰。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.adapt import (CalibrationStatus, FewShotBoundary, UncalibratedClaim,
                       min_calibration_n, recall_lower_bound)

RNG = np.random.default_rng(0)
D = 32


def _blob(center, n, scale=0.25):
    return center + RNG.normal(0, scale, size=(n, D))


@pytest.fixture
def data():
    pos_c = np.zeros(D); pos_c[0] = 1.0
    neg_c = np.zeros(D); neg_c[1] = 1.0
    return _blob(pos_c, 200), _blob(neg_c, 200)


# ---------- 校准数学 ----------

def test_n_min_matches_documented_table():
    assert min_calibration_n(0.95, 0.10, 0) == 45
    assert min_calibration_n(0.95, 0.10, 1) == 77
    assert min_calibration_n(0.99, 0.10, 0) == 230


def test_recall_bound_is_monotone_in_n():
    bounds = [recall_lower_bound(n, 0) for n in (5, 10, 20, 45, 100)]
    assert bounds == sorted(bounds)
    assert recall_lower_bound(45, 0) >= 0.95
    assert recall_lower_bound(20, 0) < 0.95      # few-shot 不够


def test_more_misses_lowers_the_bound():
    assert recall_lower_bound(100, 3) < recall_lower_bound(100, 0)


# ---------- 核心不变量 ----------

def test_boundary_change_invalidates_calibration(data):
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.fit_head(np.vstack([pos[:100], neg[:100]]),
               np.r_[np.ones(100), -np.ones(100)])
    assert b.calibrate(pos[100:160]) is CalibrationStatus.CALIBRATED

    b.add_prototype(pos[160])                    # 只加一个样例
    assert b.status is CalibrationStatus.UNCALIBRATED, \
        "边界动了但校准状态没失效 —— conformal 保证会变成装饰"


def test_cache_add_also_invalidates(data):
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.fit_head(np.vstack([pos[:100], neg[:100]]),
               np.r_[np.ones(100), -np.ones(100)])
    b.calibrate(pos[100:160])
    b.add_cache(neg[170], label=-1)
    assert b.status is CalibrationStatus.UNCALIBRATED


# ---------- 不掩盖失败 ----------

def test_uncalibrated_claim_raises_not_returns_default(data):
    pos, _ = data
    b = FewShotBoundary(dim=D)
    b.add_prototype(pos[:3])
    d = b.decide(pos[10])
    assert d.calibration is CalibrationStatus.UNCALIBRATED
    with pytest.raises(UncalibratedClaim):
        d.claim_recall_bound()


def test_few_shot_calibration_is_estimated_not_calibrated(data):
    """n=20 < 45:必须是 ESTIMATED,不能冒充 CALIBRATED。"""
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.fit_head(np.vstack([pos[:50], neg[:50]]), np.r_[np.ones(50), -np.ones(50)])
    assert b.calibrate(pos[100:120]) is CalibrationStatus.ESTIMATED
    with pytest.raises(UncalibratedClaim):
        b.decide(pos[150]).claim_recall_bound()


def test_calibrated_path_returns_a_real_bound(data):
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.fit_head(np.vstack([pos[:80], neg[:80]]), np.r_[np.ones(80), -np.ones(80)])
    b.calibrate(pos[100:180])
    d = b.decide(pos[190])
    assert d.claim_recall_bound() > 0.9


# ---------- 少样本确实改善边界 ----------

def test_one_shot_prototype_separates_better_than_chance(data):
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.add_prototype(pos[0])                      # 1 例
    sp = np.mean([b.score(x) for x in pos[100:150]])
    sn = np.mean([b.score(x) for x in neg[100:150]])
    assert sp > sn, "单样例原型应已能分开两类"


def test_more_shots_widen_the_margin(data):
    pos, neg = data
    margins = []
    for k in (1, 5, 30):
        b = FewShotBoundary(dim=D)
        b.add_prototype(pos[:k])
        margins.append(np.mean([b.score(x) for x in pos[100:150]]) -
                       np.mean([b.score(x) for x in neg[100:150]]))
    assert margins[2] > margins[0], f"样例增多间隔未变大: {margins}"


def test_epoch_increments_on_each_calibration(data):
    pos, neg = data
    b = FewShotBoundary(dim=D)
    b.fit_head(np.vstack([pos[:80], neg[:80]]), np.r_[np.ones(80), -np.ones(80)])
    b.calibrate(pos[100:180])
    e1 = b.epoch
    b.add_prototype(pos[185])
    b.calibrate(pos[100:180])
    assert b.epoch == e1 + 1


# ---------- 回归:阈值必须按置信下界选,不按经验召回 ----------

def test_calibrated_bound_always_meets_target(data):
    """CALIBRATED 状态下,声称的下界必须真的 >= target。

    回归:初版按经验召回选阈值,n=120 且经验召回恰好 95% 时下界远低于 95%,
    等于声称一个没挣到的保证。
    """
    pos, neg = data
    for n in (45, 80, 150):
        b = FewShotBoundary(dim=D, target_recall=0.95, delta=0.10)
        b.fit_head(np.vstack([pos[:60], neg[:60]]),
                   np.r_[np.ones(60), -np.ones(60)])
        if b.calibrate(pos[100:100 + n]) is CalibrationStatus.CALIBRATED:
            assert b.decide(pos[190]).claim_recall_bound() >= 0.95, \
                f"n={n} 声称 CALIBRATED 但下界不达标"


def test_below_n_min_never_reaches_calibrated(data):
    """n < 45 时,即使校准集上零漏也只能是 ESTIMATED。"""
    pos, neg = data
    for n in (5, 10, 20, 44):
        b = FewShotBoundary(dim=D)
        b.fit_head(np.vstack([pos[:60], neg[:60]]),
                   np.r_[np.ones(60), -np.ones(60)])
        assert b.calibrate(pos[100:100 + n]) is CalibrationStatus.ESTIMATED, \
            f"n={n} 不应达到 CALIBRATED"


def test_allowed_misses_grow_with_n(data):
    """n 越大,能容忍的漏报越多 —— 这正是文档表格 45/77/105 的结构。"""
    pos, neg = data
    misses = []
    for n in (45, 120, 300):
        b = FewShotBoundary(dim=D)
        b.fit_head(np.vstack([pos[:60], neg[:60]]),
                   np.r_[np.ones(60), -np.ones(60)])
        b.calibrate(pos[100:100 + n])
        misses.append(b._n_miss)
    assert misses[0] == 0, "n=45 应恰好只允许零漏"
    assert misses == sorted(misses), f"允许漏报数未随 n 单调增: {misses}"
