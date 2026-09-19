"""合成流水线测试。

重点是**统计性**的:拼接痕迹与标签独立。单看几个样本看不出泄漏,
必须用大样本检验(第 11 轮教训:逐项性质不能用少数样本去测)。
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.synth.plan import (EDGE_MARGIN_S, MIN_SEPARATION_S,
                            duration_pool_from_events, plan_cuts)

POOL = [1.2, 2.5, 0.9, 3.1, 1.8, 0.6, 4.0]
DUR = 1800.0
N = 3000


@pytest.fixture(scope="module")
def plans():
    rng = random.Random(42)
    out = {True: [], False: []}
    for i in range(N):
        pos = i % 2 == 0
        out[pos].append(plan_cuts(f"h{i}", DUR, is_positive=pos,
                                  duration_pool=POOL, rng=rng))
    return out


def _tvd(a, b, bins, rng_):
    ha, _ = np.histogram(a, bins=bins, range=rng_)
    hb, _ = np.histogram(b, bins=bins, range=rng_)
    return 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum()


# ---------- 三处必须与标签无关的采样 ----------

def test_cut_count_independent_of_label(plans):
    a = [p.n_cuts for p in plans[True]]
    b = [p.n_cuts for p in plans[False]]
    assert _tvd(a, b, 7, (1.5, 6.5)) < 0.06, "切点数泄漏标签"


def test_cut_position_independent_of_label(plans):
    a = [c.t_s for p in plans[True] for c in p.cuts]
    b = [c.t_s for p in plans[False] for c in p.cuts]
    assert _tvd(a, b, 10, (0, DUR)) < 0.06, "切点位置泄漏标签"


def test_insert_duration_independent_of_label(plans):
    """最容易漏的一处:needle 总是 1-3s 而 benign 总是 10s 的话时长即标签。"""
    a = [c.duration_s for p in plans[True] for c in p.cuts]
    b = [c.duration_s for p in plans[False] for c in p.cuts]
    assert _tvd(a, b, 7, (0.5, 4.5)) < 0.06, "插入时长泄漏标签"


def test_needle_position_matches_benign_position(plans):
    """最严格的检验:needle 切点的位置分布要和 benign 切点一致。"""
    n = [c.t_s for p in plans[True] for c in p.cuts if c.kind == "needle"]
    b = [c.t_s for p in plans[True] + plans[False]
         for c in p.cuts if c.kind == "benign"]
    assert _tvd(n, b, 10, (0, DUR)) < 0.06


def test_naive_cut_count_attack_is_chance(plans):
    """用切点数当分类器,准确率应接近 0.5。"""
    a, b = [p.n_cuts for p in plans[True]], [p.n_cuts for p in plans[False]]
    best = max((sum(1 for x in a if x >= k) + sum(1 for x in b if x < k))
               / (len(a) + len(b)) for k in range(2, 7))
    assert best < 0.56, f"仅凭切点数即可猜出标签, 准确率 {best:.3f}"


# ---------- 结构约束 ----------

def test_exactly_one_needle_per_positive(plans):
    assert all(len(p.needle_cuts) == 1 for p in plans[True])


def test_no_needle_in_negatives(plans):
    assert all(len(p.needle_cuts) == 0 for p in plans[False])


def test_cuts_respect_min_separation(plans):
    for p in plans[True][:200]:
        ts = sorted(c.t_s for c in p.cuts)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        assert all(g >= MIN_SEPARATION_S - 1e-6 for g in gaps), gaps


def test_cuts_respect_edge_margin(plans):
    for p in plans[True][:200]:
        for c in p.cuts:
            assert EDGE_MARGIN_S <= c.t_s <= DUR - EDGE_MARGIN_S


# ---------- 防御性 ----------

def test_empty_duration_pool_is_refused():
    with pytest.raises(ValueError, match="duration_pool"):
        plan_cuts("h", DUR, is_positive=True, duration_pool=[],
                  rng=random.Random(0))


def test_too_short_host_is_refused():
    with pytest.raises(ValueError, match="太短"):
        plan_cuts("h", 20.0, is_positive=True, duration_pool=POOL,
                  rng=random.Random(0))


def test_duration_pool_helper_rejects_empty():
    with pytest.raises(ValueError):
        duration_pool_from_events([])
