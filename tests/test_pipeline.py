"""流式后端 / 协议解析 / 端到端编排的测试。

编排逻辑出错时系统照跑,只是悄悄少看了帧、或者事件永远不闭合 ——
正是最该脱离 GPU 测透的部分。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import SG2Config
from sg2.models import build_midtier, build_sentinel
from sg2.models.base import StreamingVLM
from sg2.pipeline import Pipeline
from sg2.stream.protocol import ParseError, extract_json, parse_step

FLAG = '{"action":"flag","category":"C1","policy_citation":"SW-C1"}'
HOLD = '{"action":"hold"}'
CLEAR = '{"action":"clear"}'


# ==================== 协议解析 ====================

@pytest.mark.parametrize("raw,expect", [
    (HOLD, "hold"),
    (CLEAR, "clear"),
    (FLAG, "flag"),
    (f"```json\n{CLEAR}\n```", "clear"),
    (f"Looking at the frames, {FLAG} is my verdict.", "flag"),
    ('{"action":"flag","category":"C1"}', "invalid"),      # 无引用
    ('{"action":"escalate"}', "invalid"),
    ("this looks unsafe", "invalid"),
    ("", "invalid"),
    ("{broken", "invalid"),
    ('["flag"]', "invalid"),                                # 顶层不是对象
])
def test_parse_step(raw, expect):
    assert parse_step(raw).action == expect


def test_parser_does_not_repair_malformed_json():
    """只定位不修补。修补会让 RL 的格式门形同虚设。"""
    with pytest.raises(ParseError):
        extract_json('{"action": "flag"')


def test_evidence_frames_are_ints():
    s = parse_step('{"action":"flag","category":"C1","policy_citation":"P",'
                   '"evidence_frames":[1,2.0,"x",null]}')
    assert s.evidence_frames == (1, 2)


# ==================== Mock 后端 ====================

def _mock(script=None, **kw):
    cfg = SG2Config().midtier
    cfg.name = "mock"
    return build_midtier(cfg, script=script or [HOLD], **kw)


def test_mock_satisfies_protocol():
    assert isinstance(_mock(), StreamingVLM)


def test_policy_header_is_cached():
    m = _mock()
    assert m.set_policy("policy " * 100, "v1") is False
    assert m.set_policy("policy " * 100, "v1") is True
    assert m.set_policy("policy " * 100, "v2") is False


def test_vision_window_evicts_under_budget():
    m = _mock(tokens_per_frame=512)
    m.cfg.context.max_vision_tokens = 1024
    m.reset()
    for t in range(10):
        m.ingest(np.zeros((4, 4, 3), np.float32), float(t))
    assert m.ctx.vision_tokens <= 1024


def test_flag_writes_event_and_clear_truncates():
    m = _mock(script=[FLAG, CLEAR])
    m.ingest(np.zeros((4, 4, 3), np.float32), 0.0)
    assert m.step().action == "flag"
    assert m.ctx.event_tokens > 0
    m.step()
    assert m.close_event() > 0 and m.ctx.event_tokens == 0


def test_reset_clears_everything():
    m = _mock(script=[FLAG])
    m.set_policy("p" * 40, "v1")
    m.ingest(np.zeros((4, 4, 3), np.float32), 0.0)
    m.step()
    m.reset()
    assert m.ctx.event_tokens == 0 and m.ctx.vision_tokens == 0


# ==================== 端到端编排 ====================

def _pipe(rho=0.0, script=None, seed=0):
    cfg = SG2Config()
    cfg.override("sentinel.encoder.name", "random")
    cfg.override("sentinel.channels.codec", False)
    cfg.override("midtier.name", "mock")
    cfg.override("runtime.seed", seed)
    if rho > 0:
        cfg.override("coverage.rho", rho)
    else:
        cfg.coverage.rho = 1e-9          # 绕过 rho>0 的构造约束
    return Pipeline(cfg, midtier=build_midtier(cfg.midtier,
                                               script=script or [HOLD]))


def _frames(n=40, size=8):
    rng = np.random.default_rng(0)
    return [(float(t), rng.random((size, size, 3)).astype(np.float32))
            for t in range(n)]


def test_pipeline_records_every_tick():
    r = _pipe().run(_frames(40))
    assert len(r.ticks) == 40


def test_coverage_floor_escalates_without_any_alarm():
    """ρ 是旁路:sentinel 完全无判别力时仍有流量进中间层。

    这是优雅退化定理(docs/06 §6.1)成立的机制。
    """
    r = _pipe(rho=0.5, script=[FLAG]).run(_frames(60))
    assert r.alarms == 0, "本用例不应触发 CUSUM"
    assert r.coverage_forced > 0 and r.escalations > 0


def test_zero_coverage_means_no_forced_escalation():
    r = _pipe(rho=0.0, script=[FLAG]).run(_frames(60))
    assert r.coverage_forced == 0


def test_detection_delay_is_none_when_never_flagged():
    r = _pipe(rho=0.5, script=[HOLD]).run(_frames(40))
    assert r.first_flag_t_s is None
    assert r.detection_delay(10.0) is None


def test_detection_delay_is_nonnegative():
    r = _pipe(rho=1.0, script=[HOLD, HOLD, FLAG]).run(_frames(40))
    d = r.detection_delay(0.0)
    assert d is not None and d >= 0.0


def test_escalation_rate_is_a_fraction():
    r = _pipe(rho=0.3).run(_frames(50))
    assert 0.0 <= r.escalation_rate <= 1.0


def test_pipeline_is_deterministic_given_seed():
    a = _pipe(rho=0.3, seed=7).run(_frames(40))
    b = _pipe(rho=0.3, seed=7).run(_frames(40))
    assert [t.forced_by_coverage for t in a.ticks] == \
           [t.forced_by_coverage for t in b.ticks]


def test_clear_closes_an_open_event():
    r = _pipe(rho=1.0, script=[FLAG, CLEAR, HOLD]).run(_frames(6))
    assert r.events_closed == 1


# ==================== 接线审计的回归 ====================

def test_uncovered_fields_are_parsed():
    """回归:parse_step 漏解析 description/suggested_category,
    导致缺口追踪把所有案例归到 <未分类>,永远聚不出可行动缺口 ——
    而 StreamStep 里明明有这两个字段,静默丢失。
    """
    from sg2.stream.protocol import parse_step
    s = parse_step(json.dumps({"action": "uncovered", "description": "描述",
                               "suggested_category": "X_new"}))
    assert s.description == "描述" and s.suggested_category == "X_new"


def _wired_pipe(script, rho=0.5, audit=0.5, seed=0):
    from sg2.pipeline import Pipeline
    cfg = SG2Config()
    cfg.override("sentinel.encoder.name", "random")
    cfg.override("sentinel.channels.codec", False)
    cfg.override("midtier.name", "mock")
    cfg.override("coverage.rho", rho)
    cfg.override("coverage.audit_rate", audit)
    cfg.override("runtime.seed", seed)
    return Pipeline(cfg, midtier=build_midtier(cfg.midtier, script=script))


UNC = json.dumps({"action": "uncovered", "description": "描述",
                  "suggested_category": "X_child"})


def test_pipeline_fills_ring_buffer():
    """回归:buffer 曾实现完整却从未被 Pipeline 调用。"""
    r = _wired_pipe([HOLD]).run(_frames(40))
    assert r.buffer_stats["appended"] == 40


def test_pipeline_feeds_aci():
    """回归:aci 曾实现完整却从未被 Pipeline 调用。"""
    r = _wired_pipe([HOLD], audit=1.0).run(_frames(40), nu_s=10.0)
    assert r.audited == 40 and r.aci_stats["fed"] == 40


def test_aci_gets_no_feedback_without_ground_truth():
    """没有真值就不该喂 ACI —— 把'没反馈'当'没漏报'会让阈值单调收紧。"""
    r = _wired_pipe([HOLD], audit=1.0).run(_frames(40), nu_s=None)
    assert r.aci_stats["fed"] == 0


def test_pipeline_tracks_policy_gaps():
    """回归:outofpolicy 曾实现完整却从未被 Pipeline 调用。"""
    p = _wired_pipe([UNC], rho=1.0)
    r = p.run(_frames(30))
    assert r.uncovered_cases > 0
    assert p.gaps.stats["by_category"].get("X_child", 0) > 0


def test_gap_tracking_does_not_change_policy():
    """识别缺口与修改政策是两件事,后者必须经人工。"""
    p = _wired_pipe([UNC], rho=1.0)
    before = p.corpus.fingerprint()
    p.run(_frames(30))
    assert p.corpus.fingerprint() == before
