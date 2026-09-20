"""Analyst 工具集的测试。

两条关键性质:
  1. 预算必须真的能拦住 —— agentic 是多轮的,不设上限单次升级可无限膨胀
  2. 滚出 buffer 的内容取不回来 —— 那是硬 miss 不是延迟,不能静默返回空
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.analyst.tools import (BudgetExceeded, ToolBox, ToolBudget,
                               render_tool_specs)
from sg2.buffer import RingBuffer
from sg2.policy import safewatch_corpus

RNG = np.random.default_rng(0)


def _box(window_s=30.0, n=300, **bkw):
    b = RingBuffer(window_s=window_s)
    for i in range(n):
        b.append(i * 0.1, RNG.random((64, 64, 3)).astype(np.float32))
    return ToolBox(buffer=b, corpus=safewatch_corpus(),
                   budget=ToolBudget(**bkw))


# ==================== 预算 ====================

def test_budget_blocks_after_max_calls():
    tb = _box(max_calls=3)
    for _ in range(3):
        assert tb.policy_lookup().ok
    r = tb.policy_lookup()
    assert not r.ok and "工具调用" in r.error


def test_budget_blocks_on_tokens():
    tb = _box(max_tokens=300)
    assert tb.policy_lookup().ok           # 200
    assert not tb.policy_lookup().ok       # 400 > 300


def test_budget_blocks_on_decoded_frames():
    tb = _box(max_decoded_frames=5)
    assert not tb.rewind(10.0, 20.0, fps=4.0).ok


def test_budget_failure_returns_result_not_raises():
    """预算耗尽要返回当前最佳判断,不是抛异常炸掉整个升级。"""
    tb = _box(max_calls=1)
    tb.policy_lookup()
    r = tb.policy_lookup()
    assert not r.ok and r.cost["calls"] > 1


def test_budget_charge_raises_directly():
    b = ToolBudget(max_calls=1)
    b.charge()
    with pytest.raises(BudgetExceeded):
        b.charge()


# ==================== 滚出窗口 ====================

def test_rolled_off_zoom_fails_loudly():
    """取不回来就要说,不能静默返回空。"""
    tb = _box(window_s=5.0)
    r = tb.zoom(0.5, (0, 0, 1, 1))
    assert not r.ok and "滚出" in r.error


def test_rolled_off_rewind_fails_loudly():
    tb = _box(window_s=5.0)
    assert not tb.rewind(0.0, 1.0).ok


def test_in_window_zoom_succeeds():
    tb = _box(window_s=30.0)
    r = tb.zoom(20.0, (0.25, 0.25, 0.75, 0.75))
    assert r.ok and r.value.shape == (32, 32, 3)


# ==================== 参数校验 ====================

@pytest.mark.parametrize("bbox", [(0.5, 0, 0.2, 1), (0, 0, 1.5, 1),
                                  (-0.1, 0, 1, 1), (0, 0.9, 1, 0.1)])
def test_bad_bbox_is_refused(bbox):
    assert not _box().zoom(20.0, bbox).ok


def test_diff_is_cheaper_than_rewind():
    """diff 只解 2 帧,rewind 解一整段 —— 应先用 diff 定位。"""
    a, b = _box(), _box()
    a.diff(10.0, 20.0)
    b.rewind(10.0, 20.0, fps=4.0)
    assert a.spent["frames"] < b.spent["frames"]


# ==================== 未接后端 ====================

def test_missing_backend_reports_clearly():
    tb = _box()
    for r in (tb.ocr(20.0), tb.transcribe(10.0, 20.0),
              tb.case_lookup(np.zeros(8))):
        assert not r.ok and "未接" in r.error


# ==================== 检索 ====================

def test_policy_lookup_filters():
    tb = _box()
    allc = tb.policy_lookup().value
    hit = tb.policy_lookup("暴力").value
    assert len(hit) < len(allc) and all("C3" in c["id"] for c in hit)


def test_trace_records_every_call():
    tb = _box()
    tb.policy_lookup(); tb.diff(10.0, 12.0); tb.ocr(20.0)
    assert len(tb.trace) == 3
    assert [r.ok for r in tb.trace] == [True, True, False]


def test_tool_specs_render():
    s = render_tool_specs()
    for name in ("zoom", "rewind", "diff", "policy_lookup", "case_lookup"):
        assert name in s
