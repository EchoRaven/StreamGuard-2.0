"""流式 KV 分段 + SFT 数据构造的测试。

两个模块都是**纯逻辑**,出错时表现隐蔽(模型照跑,只是学错东西或注意力
悄悄坏掉),所以在没有 GPU 的情况下就要测透。
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.stream.context import Segment, SinkEvicted, StreamContext
from sg2.train.sft import (DEFAULT_MIX, Window, resample_to_mix, windows_for)


# ==================== KV 分段 ====================

def test_policy_header_cache_hit():
    c = StreamContext()
    assert c.set_policy(500, "v1") is False      # 首次:需编码
    assert c.set_policy(500, "v1") is True       # 再次:命中
    assert c.set_policy(500, "v2") is False      # 换版本:重编码


def test_sink_is_never_evicted():
    """attention sink 被驱逐会让注意力分布崩坏 —— 必须抛异常而非悄悄执行。"""
    c = StreamContext(sink_tokens=4)
    c.set_policy(1000, "v1")
    with pytest.raises(SinkEvicted):
        c.evict_to(100)


def test_policy_survives_aggressive_eviction():
    c = StreamContext(sink_tokens=4)
    c.set_policy(500, "v1")
    for i in range(20):
        c.append_frame(100, t_s=float(i))
    c.evict_to(c.sink_tokens + c.policy_tokens)
    assert c.policy_tokens == 500, "政策头不应被驱逐"
    assert c.vision_tokens == 0


def test_vision_window_evicts_by_time():
    c = StreamContext(max_vision_tokens=10**6, vision_window_s=5.0)
    for i in range(10):
        c.append_frame(10, t_s=float(i))
    assert all(9.0 - b.t_s <= 5.0 for b in c._vision)


def test_vision_window_evicts_by_token_budget():
    c = StreamContext(max_vision_tokens=300, vision_window_s=10**6)
    for i in range(10):
        c.append_frame(100, t_s=float(i))
    assert c.vision_tokens <= 300


def test_close_event_truncates_only_event():
    """事件闭合 = 截断 S2,这是 1.0 context reset 的缓存层对应物。"""
    c = StreamContext()
    c.set_policy(400, "v1")
    c.append_frame(200, t_s=0.0)
    c.append_event(150, "证据")
    freed = c.close_event()
    assert freed == 150
    assert c.event_tokens == 0
    assert c.policy_tokens == 400 and c.vision_tokens == 200
    assert not c.event_open


def test_event_context_has_its_own_budget():
    c = StreamContext(max_event_tokens=200)
    for i in range(5):
        c.append_event(100, f"e{i}")
    assert c.event_tokens <= 200


def test_total_always_includes_sink():
    c = StreamContext(sink_tokens=4)
    assert c.total_tokens == 4
    c.set_policy(100, "v")
    assert c.total_tokens == 104


def test_oversized_policy_is_refused():
    with pytest.raises(ValueError, match="超出上限"):
        StreamContext(max_policy_tokens=100).set_policy(500, "v")


# ==================== SFT 窗口 ====================

def _rec(events, dur=120.0):
    from sg2.schema import ClipRecord
    return ClipRecord(
        id="c1",
        source={"kind": "synthetic"},
        media={"duration_s": dur, "fps": 30.0, "codec": "h264"},
        label={"safe": not events, "categories": sorted({e["category"] for e in events}),
               "events": events},
        axes={}, metadata_adversarial={"condition": "aligned"},
        splice=None, split="test", pool="eval")


EV = [{"event_id": "e0", "category": "C1", "t_start_s": 60.0, "t_end_s": 64.0,
       "frame_start": 1800, "frame_end": 1920, "severity": "high",
       "evidence_modality": ["pixel"]}]


def test_windows_are_causal():
    """窗口绝不能含未来帧 —— 否则模型学到真实流上不存在的能力。"""
    for w in windows_for(_rec(EV)):
        assert w.t_end_s > w.t_start_s
        if w.kind == "pos_partial":
            assert w.t_end_s < EV[0]["t_end_s"], "部分窗口不应看到 event 结尾"


def test_partial_window_targets_hold():
    """只看到 event 前一部分时,正确动作是 hold 不是 flag。"""
    ws = [w for w in windows_for(_rec(EV)) if w.kind == "pos_partial"]
    assert ws and all(w.action == "hold" for w in ws)


def test_full_window_targets_flag_with_citation():
    ws = [w for w in windows_for(_rec(EV)) if w.kind == "pos_full"]
    assert ws
    for w in ws:
        assert w.action == "flag" and w.policy_citation
        assert json.loads(w.target())["policy_citation"] == w.policy_citation


def test_negative_windows_do_not_overlap_events():
    for w in windows_for(_rec(EV)):
        if w.kind.startswith("neg"):
            assert w.t_end_s <= EV[0]["t_start_s"] or w.t_start_s >= EV[0]["t_end_s"]


def test_flag_without_citation_is_refused_at_construction():
    with pytest.raises(ValueError, match="citation"):
        Window("c", 0.0, 1.0, "pos_full", "flag", category="C1")


def test_non_flag_with_citation_is_refused():
    with pytest.raises(ValueError, match="citation"):
        Window("c", 0.0, 1.0, "pos_partial", "hold", policy_citation="P1")


# ==================== 配比重采样 ====================

def _mkw(kind, i):
    act = {"pos_full": "flag", "pos_partial": "hold"}.get(kind, "clear")
    kw = dict(category="C1", policy_citation="SW-C1") if act == "flag" else {}
    return Window("c", float(i), float(i) + 1, kind, act, **kw)


def test_resample_hits_target_mix():
    """不重采样时配比由数据集正负比决定 —— 实测 neg_cross 会占到 47%。"""
    ws = ([_mkw("pos_full", i) for i in range(20)]
          + [_mkw("pos_partial", i) for i in range(20)]
          + [_mkw("neg_same", i) for i in range(200)]
          + [_mkw("neg_cross", i) for i in range(200)])
    out = resample_to_mix(ws, rng=random.Random(0))
    total = len(out)
    for k, target in DEFAULT_MIX.items():
        got = sum(1 for w in out if w.kind == k) / total
        assert abs(got - target) < 0.05, f"{k}: {got:.1%} vs {target:.0%}"


def test_resample_never_oversamples():
    """不做有放回过采样 —— 会让模型记住少数几个窗口。"""
    ws = [_mkw("pos_full", i) for i in range(3)] + \
         [_mkw("neg_same", i) for i in range(100)]
    out = resample_to_mix(ws, rng=random.Random(0))
    ids = [(w.kind, w.t_start_s) for w in out]
    assert len(ids) == len(set(ids)), "出现重复窗口"


def test_resample_with_missing_kind_returns_original():
    ws = [_mkw("neg_same", i) for i in range(10)]
    assert len(resample_to_mix(ws)) == 10
