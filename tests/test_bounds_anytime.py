"""风险上界层级 + anytime-valid 监控的测试。

两条核心:
  1. 精确二项在稀缺区间大幅优于集中不等式 —— 原文没列这个界
  2. **认证必须可撤销**,否则漂移后保证是假的
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.aci import abstention_floor
from sg2.anytime import AnytimeRiskMonitor
from sg2.bounds import (bernstein_ucb, compare_bounds, ecrc_certifies,
                        exact_binomial_ucb, hoeffding_ucb, min_n_to_certify)


# ==================== 界的层级 ====================

def test_hierarchy_holds_at_strict_target():
    """Kotte 2026 Prop.2:严格目标下 Hoeffding ⊆ Bernstein。"""
    r = np.zeros(200)
    r[:4] = 1.0
    assert bernstein_ucb(r) < hoeffding_ucb(r)


def test_exact_binomial_dominates_in_the_scarce_regime():
    """★ n=45 零漏报时**只有精确二项**能认证 alpha=0.05。

    原文的层级里没有精确二项,而我们的损失是 0/1,它对伯努利是紧的。
    """
    res = {b.name: b.certifies for b in compare_bounds(np.zeros(45), 0.05)}
    assert res["ExactBinom"]
    assert not res["Hoeffding"] and not res["Bernstein"]


def test_exact_binomial_needs_far_fewer_samples():
    """实测:认证 alpha=0.05(真实漏报 2%)所需最小 n
    Hoeffding 1657 / Bernstein 466 / e-CRC 265 / 精确二项 105。
    """
    ns = {b: min_n_to_certify(b, 0.02, 0.05)
          for b in ("Hoeffding", "Bernstein", "ExactBinom")}
    assert ns["ExactBinom"] < ns["Bernstein"] < ns["Hoeffding"]


def test_exact_binomial_matches_documented_45():
    """n=45 零漏报的 UCB 应恰好压在 0.05 线下(docs/03 的依据)。"""
    assert exact_binomial_ucb(np.zeros(45)) <= 0.05
    assert exact_binomial_ucb(np.zeros(44)) > 0.05


def test_exact_binomial_falls_back_on_non_binary_loss():
    """非 0/1 损失上硬套精确二项会给出无效的界,必须退回。"""
    r = np.array([0.3, 0.7, 0.2, 0.9])
    assert exact_binomial_ucb(r) == hoeffding_ucb(r)


def test_ecrc_certifies_clean_data_eventually():
    assert ecrc_certifies(np.zeros(300), alpha=0.05)


def test_no_bound_certifies_when_risk_exceeds_target():
    """µ > α 时任何界都不该认证 —— 那是不可行区间。"""
    r = np.zeros(500); r[:150] = 1.0          # 30% 风险
    for b in compare_bounds(r, alpha=0.05):
        assert not b.certifies, b.name


# ==================== anytime-valid 监控 ====================

def _stream(n, mu, seed=0, shift_at=None, mu2=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = mu2 if (shift_at and i >= shift_at and mu2 is not None) else mu
        risk = float(rng.random() < m)
        out.append((float(rng.normal(0.8 if risk else -0.8, 1.0)), risk))
    return out


def test_monitor_certifies_in_the_feasible_regime():
    m = AnytimeRiskMonitor(alpha=0.05)
    data = _stream(3000, 0.02)
    for s, r in data:
        m.update(s, r)
    for s, _ in data[-500:]:
        m.should_emit(s)
    assert m.certified_tau() is not None
    assert m.abstention_rate < 0.2


def test_certification_is_revocable_under_shift():
    """★ 回归:上鞅只涨不跌时,阈值一旦认证就永久认证,
    漂移后弃权率只有 3.8%,远低于 Prop.3 的 36.8% —— 保证是假的。
    """
    m = AnytimeRiskMonitor(alpha=0.05)
    data = _stream(4000, 0.02, shift_at=2000, mu2=0.40)
    for s, r in data:
        m.update(s, r)
    for s, _ in data[-800:]:
        m.should_emit(s)
    assert m.abstention_rate > 0.5
    assert m.respects_floor(0.40)


def test_monitor_respects_the_prop3_floor():
    m = AnytimeRiskMonitor(alpha=0.05)
    data = _stream(3000, 0.40)
    for s, r in data:
        m.update(s, r)
    for s, _ in data[-500:]:
        m.should_emit(s)
    assert m.abstention_rate >= abstention_floor(0.40, 0.05) - 1e-9


def test_none_feedback_does_not_update():
    """没有反馈就不更新 —— full feedback 是 Prop.8 要求的前提。"""
    m = AnytimeRiskMonitor(alpha=0.05)
    for _ in range(500):
        m.update(0.0, None)
    assert m.certified_tau() is None and m.stats["seen"] == 500


def test_abstains_when_nothing_certified():
    m = AnytimeRiskMonitor(alpha=0.05)
    assert not m.should_emit(-10.0)
    assert m.abstention_rate == 1.0


@pytest.mark.parametrize("kw", [{"alpha": 0.0}, {"alpha": 1.0},
                                {"delta": 0.0}, {"taus": ()}])
def test_bad_config_is_refused(kw):
    with pytest.raises(ValueError):
        AnytimeRiskMonitor(**kw)
