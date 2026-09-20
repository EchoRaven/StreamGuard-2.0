"""帧驱逐的测试。

核心是两条:
  1. 驱逐的监督信号**自监督**,不需要人工标注"哪帧重要"
  2. FIFO 对流式检测特别糟 —— needle 常在刚过去那段,FIFO 丢的恰恰是那儿
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.evict import (POLICIES, FrameRef, agreement_at_budget,
                       hindsight_targets, leave_one_out_importance)
from sg2.stream.context import StreamContext

RNG = np.random.default_rng(0)
D = 32


def _stream(n=30, needle_at=18):
    base = RNG.normal(0, 1, D); base /= np.linalg.norm(base)
    nd = RNG.normal(0, 1, D); nd /= np.linalg.norm(nd)
    out = []
    for i in range(n):
        e = nd if i == needle_at else base + RNG.normal(0, 0.05, D)
        e = e / np.linalg.norm(e)
        out.append(FrameRef(t_s=i * 2.0, emb=e, is_keyframe=(i % 7 == 0),
                            motion=float(RNG.random()),
                            score=float(e @ nd)))
    return out, nd


def _judge(nd):
    return lambda fs: "flag" if any(f.emb @ nd > 0.8 for f in fs) else "clear"


# ==================== 自监督重要性 ====================

def test_importance_needs_no_human_labels():
    """"标注"来自模型自己的输出,不需要人看任何一帧。"""
    fr, nd = _stream()
    imp = leave_one_out_importance(fr, _judge(nd))
    assert imp.ranking[0] == 18
    assert imp.per_frame.sum() == 1.0


def test_concentration_tells_whether_optimising_is_worth_it():
    """集中度接近 0 = 丢哪帧都一样,再优化也没收益。"""
    fr, nd = _stream()
    assert leave_one_out_importance(fr, _judge(nd)).concentrated > 0.9

    flat = [FrameRef(t_s=i * 2.0, emb=np.ones(D) / np.sqrt(D))
            for i in range(10)]
    assert leave_one_out_importance(flat, lambda fs: "clear").concentrated == 0.0


def test_leave_one_out_costs_n_plus_one_calls():
    fr, nd = _stream(n=12)
    assert leave_one_out_importance(fr, _judge(nd)).n_calls == 13


def test_hindsight_targets_keep_the_needle_and_the_latest():
    fr, nd = _stream()
    y = hindsight_targets(fr, _judge(nd), keep=5)
    assert y[18] == 1.0 and y[-1] == 1.0


# ==================== FIFO 的缺陷 ====================

@pytest.mark.parametrize("keep", [3, 5, 8])
def test_fifo_loses_the_needle(keep):
    """回归依据:FIFO 是 RingBuffer 与 StreamContext 原本的策略。"""
    fr, nd = _stream()
    assert not agreement_at_budget(fr, _judge(nd), keep, "fifo")["agree"]


@pytest.mark.parametrize("policy", ["coverage", "salience", "hybrid"])
@pytest.mark.parametrize("keep", [3, 5, 8])
def test_smart_policies_keep_the_decision(policy, keep):
    fr, nd = _stream()
    assert agreement_at_budget(fr, _judge(nd), keep, policy)["agree"]


def test_all_policies_agree_when_budget_is_full():
    fr, nd = _stream()
    for p in POLICIES:
        assert agreement_at_budget(fr, _judge(nd), len(fr), p)["agree"]


def test_policies_respect_the_budget():
    fr, _ = _stream()
    for p, fn in POLICIES.items():
        assert len(fn(fr, 7)) <= 7, p


def test_coverage_prefers_diverse_frames():
    """冗余帧该先丢 —— 两帧几乎一样时留一帧就够。"""
    fr, _ = _stream(n=20, needle_at=10)
    sel = POLICIES["coverage"](fr, 4)
    assert 10 in sel


# ==================== StreamContext 集成 ====================

def test_context_keeps_important_frame_under_budget():
    c = StreamContext(max_vision_tokens=300, vision_window_s=1e6)
    for i, imp in enumerate([0.1, 0.9, 0.2, 0.3, 0.15]):
        c.append_frame(100, t_s=float(i), importance=imp)
    tags = [b.tag for b in c._vision]
    assert any("0.900" in t for t in tags), "最重要的帧被丢了"
    assert any("0.150" in t for t in tags), "最新帧被丢了"


def test_context_falls_back_to_fifo_without_importance():
    c = StreamContext(max_vision_tokens=300, vision_window_s=1e6)
    for i in range(5):
        c.append_frame(100, t_s=float(i))
    assert c.n_frames == 3


def test_time_window_evicts_regardless_of_importance():
    """超出时间窗的帧本来就该走,与重要性无关。"""
    c = StreamContext(max_vision_tokens=10 ** 6, vision_window_s=3.0)
    for i in range(10):
        c.append_frame(10, t_s=float(i), importance=1.0 if i == 0 else 0.0)
    assert all(9.0 - b.t_s <= 3.0 for b in c._vision)


def test_latest_frame_is_never_evicted_by_importance():
    c = StreamContext(max_vision_tokens=200, vision_window_s=1e6)
    for i in range(6):
        c.append_frame(100, t_s=float(i), importance=0.0)
    assert max(b.t_s for b in c._vision) == 5.0
