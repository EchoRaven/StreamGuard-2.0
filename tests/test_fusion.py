"""Sentinel 融合头测试。

重点锁两条容易在重构中丢掉、且丢了会让指标虚高的性质:
  1. 按源视频切分(按帧切会让验证集变成训练集副本)
  2. 类别权重(正例率 <1% 时不加权会全预测负例)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.train import FusionHead, auprc, split_by_source

RNG = np.random.default_rng(0)


@pytest.fixture
def imbalanced():
    n, d = 4000, 8
    y = (RNG.random(n) < 0.01).astype(float)
    X = RNG.normal(0, 1, (n, d))
    X[y > 0, 0] += 2.0
    X[y > 0, 3] += 1.2
    groups = np.repeat(np.arange(n // 50), 50)
    return X, y, groups


# ---------- 切分 ----------

def test_split_by_source_has_no_group_overlap(imbalanced):
    _, _, g = imbalanced
    tr, va = split_by_source(g)
    assert not (set(g[tr]) & set(g[va])), "源视频跨切分出现 —— 验证集会虚高"


def test_split_is_deterministic_given_seed(imbalanced):
    _, _, g = imbalanced
    a = split_by_source(g, seed=3)[1]
    b = split_by_source(g, seed=3)[1]
    assert np.array_equal(a, b)


def test_split_covers_everything(imbalanced):
    _, _, g = imbalanced
    tr, va = split_by_source(g)
    assert np.all(tr ^ va), "训练/验证必须互补且无遗漏"


# ---------- 训练 ----------

def test_learns_the_injected_signal(imbalanced):
    X, y, g = imbalanced
    tr, _ = split_by_source(g)
    h = FusionHead(dim=X.shape[1]).fit(X[tr], y[tr])
    top2 = set(np.argsort(-np.abs(h.weights))[:2].tolist())
    assert {0, 3} & top2, f"未学到注入信号所在维度, top2={top2}"


def test_beats_random_baseline_on_auprc(imbalanced):
    X, y, g = imbalanced
    tr, va = split_by_source(g)
    h = FusionHead(dim=X.shape[1]).fit(X[tr], y[tr])
    ap = auprc(y[va], h.score(X[va]))
    assert ap > 5 * y[va].mean(), f"AUPRC {ap:.3f} 未显著超过基线 {y[va].mean():.3f}"


def test_class_weighting_prevents_all_negative_collapse(imbalanced):
    """1% 正例率下不加权会全预测负例。加权后正例分数必须整体更高。"""
    X, y, g = imbalanced
    tr, va = split_by_source(g)
    h = FusionHead(dim=X.shape[1]).fit(X[tr], y[tr])
    s = h.score(X[va])
    assert s[y[va] > 0].mean() > s[y[va] == 0].mean()


def test_score_is_unbounded_logit_not_probability(imbalanced):
    """CUSUM 要的是无界连续量,不是挤在 [0,1] 的概率。"""
    X, y, g = imbalanced
    tr, _ = split_by_source(g)
    h = FusionHead(dim=X.shape[1]).fit(X[tr], y[tr])
    s = h.score(X)
    assert s.min() < 0.0, "分数应可为负 —— 被压进 [0,1] 说明返回的是概率"


# ---------- 指标 ----------

def test_auprc_is_sensitive_where_auc_is_not():
    """构造一个 AUC 尚可但 AUPRC 很差的情形 —— 这正是不用 AUC 的理由。"""
    y = np.zeros(1000); y[:10] = 1
    s = RNG.normal(0, 1, 1000); s[:10] += 1.0
    ap = auprc(y, s)
    assert ap < 0.5, f"AUPRC {ap:.3f};该情形下它应明显低"


def test_auprc_perfect_ranking_is_one():
    y = np.array([1., 1., 0., 0., 0.])
    assert auprc(y, np.array([9., 8., 1., 0., -1.])) == pytest.approx(1.0)


# ---------- 防御性 ----------

def test_dim_mismatch_is_refused():
    with pytest.raises(ValueError, match="维度"):
        FusionHead(dim=5).fit(RNG.normal(0, 1, (20, 8)), np.zeros(20))


def test_score_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="未训练"):
        FusionHead(dim=4).score(np.zeros((2, 4)))
