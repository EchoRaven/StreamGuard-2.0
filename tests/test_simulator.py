"""流式训练模拟器的测试。

核心是**训练与运行时用同一份采样逻辑**。不一致时不会报错,
只会让线上指标莫名低于离线 —— 最难查的那种。
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import SG2Config
from sg2.train.simulator import (SimConfig, StreamSimulator, resample_ticks,
                                 sequence_stats, simulate_many)


def _sim(**kw):
    return StreamSimulator(SimConfig(**kw), seed=0)


EV = dict(duration_s=120.0, nu_s=60.0, nu_end_s=64.0)


# ==================== 与运行时一致 ====================

def test_config_derives_from_the_same_sg2config():
    """训练参数必须从**同一份**配置派生,不另起一套。"""
    cfg = SG2Config()
    cfg.override("sentinel.sample_fps", 1.5)
    cfg.override("cusum.jitter_s", 0.9)
    sc = SimConfig.from_config(cfg)
    assert sc.sentinel_fps == 1.5 and sc.jitter_s == 0.9


def test_sampling_has_jitter_not_uniform():
    """部署带抖动,训练不带的话模型会以为帧是等距的。"""
    ts = _sim().sample_times(120.0)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert len(set(round(g, 3) for g in gaps)) > 5


def test_dedup_changes_the_gap_distribution():
    a = sequence_stats(_sim(dedup_skip_rate=0.0).simulate(**EV))
    b = sequence_stats(_sim(dedup_skip_rate=0.3).simulate(**EV))
    assert b["n_ticks"] < a["n_ticks"]
    assert b["gap_std"] > a["gap_std"]


def test_burst_sampling_after_alarm():
    """告警后突发采样 —— 间隔应变密。"""
    ticks = _sim().simulate(alarm_at=70.0, **EV)
    before = [t.t_s for t in ticks if t.t_s < 70]
    after = [t.t_s for t in ticks if t.t_s > 71]
    g_before = [b - a for a, b in zip(before, before[1:])]
    g_after = [b - a for a, b in zip(after, after[1:])]
    assert sum(g_after) / len(g_after) < sum(g_before) / len(g_before)


def test_eviction_actually_happens():
    """训练序列必须长到**真的触发驱逐** ——
    否则模型从没见过"驱逐之后"的状态。"""
    assert sequence_stats(_sim().simulate(**EV))["evictions"] > 0


def test_window_respects_hardware_cap():
    """real 分辨率下 11GB 卡实测最多 2 帧。"""
    ticks = _sim(max_window_frames=2).simulate(alarm_at=70.0, **EV)
    assert max(len(t.window_t_s) for t in ticks) <= 2


def test_window_has_no_duplicate_timestamps():
    """回归:resample 曾把同一帧非相邻地重复取到。"""
    for t in _sim().simulate(alarm_at=70.0, **EV):
        assert len(t.window_t_s) == len(set(t.window_t_s))


# ==================== 因果与标签 ====================

def test_windows_are_causal():
    """窗口只含 [t-w, t] —— **含当前帧是对的**,那正是要判的帧。

    容差用 1e-3 而非 1e-6:tick 的 t_s 存的是 round(t,4),窗口帧存的是
    原始浮点,同一帧两处会差到 1e-5 量级。用过紧的容差会把"当前帧"
    误判成"未来帧"。
    """
    for t in _sim().simulate(alarm_at=70.0, **EV):
        assert all(w <= t.t_s + 1e-3 for w in t.window_t_s), (
            f"t={t.t_s} 窗口={t.window_t_s}")


def test_window_never_contains_a_genuinely_future_frame():
    """真正的未来帧(超出一个采样间隔)绝不能出现。"""
    sim = _sim()
    step = 1.0 / sim.cfg.sentinel_fps
    for t in sim.simulate(alarm_at=70.0, **EV):
        assert all(w < t.t_s + step * 0.5 for w in t.window_t_s)


def test_partial_coverage_targets_hold():
    """事件刚开始、证据不全时正确动作是**等**,不是猜。"""
    assert sequence_stats(_sim().simulate(**EV))["has_hold"]


def test_safe_stream_has_no_flag():
    st = sequence_stats(_sim().simulate(duration_s=60.0, nu_s=None))
    assert st["labels"].get("flag", 0) == 0


def test_before_event_is_clear():
    for t in _sim().simulate(**EV):
        if t.t_s < 60.0:
            assert t.label == "clear"


# ==================== 正例稀缺 ====================

def test_positives_are_scarce_in_natural_sampling():
    """这是流式训练的**根本困难**,不是实现问题。

    实测 0.5fps 下 4 秒的事件只落到 2 个 tick,而 120s 流有 200+ tick。
    """
    st = sequence_stats(_sim().simulate(**EV))
    pos = st["labels"].get("flag", 0) + st["labels"].get("hold", 0)
    assert pos / st["n_ticks"] < 0.05


def test_resample_hits_target_mix():
    specs = [dict(duration_s=120.0, nu_s=20.0 + i * 3, nu_end_s=24.0 + i * 3)
             for i in range(30)]
    bal = resample_ticks(simulate_many(_sim(), specs), rng=random.Random(0))
    st = sequence_stats(bal)
    tot = st["n_ticks"]
    assert abs(st["labels"].get("flag", 0) / tot - 0.25) < 0.08
    assert abs(st["labels"].get("hold", 0) / tot - 0.25) < 0.08


def test_resample_never_oversamples():
    """重复同一个 tick 会让模型记住那几帧。"""
    specs = [dict(duration_s=120.0, nu_s=20.0 + i * 3, nu_end_s=24.0 + i * 3)
             for i in range(20)]
    bal = resample_ticks(simulate_many(_sim(), specs), rng=random.Random(0))
    key = [(t.t_s, t.frame_idx, t.label) for t in bal]
    assert len(key) == len(set(key))


def test_scarcity_needs_many_streams_not_oversampling():
    one = resample_ticks(_sim().simulate(**EV))
    specs = [dict(duration_s=120.0, nu_s=20.0 + i * 3, nu_end_s=24.0 + i * 3)
             for i in range(30)]
    many = resample_ticks(simulate_many(_sim(), specs), rng=random.Random(0))
    assert len(many) > len(one) * 5
