"""SG2 格式与校验器的回归测试。

重点锁住两件容易在重构中丢掉的性质:
  1. loader 的池守卫
  2. 拼接泄漏检查在干净数据上不误报、在脏数据上必报
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.schema import ALL_POOLS, ClipRecord, Event, PoolViolation, Splice, load
from sg2 import validate as V

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "examples/manifest.clean.jsonl"
DIRTY = ROOT / "examples/manifest.dirty.jsonl"


@pytest.fixture(scope="module", autouse=True)
def _samples():
    if not CLEAN.exists() or not DIRTY.exists():
        subprocess.run([sys.executable, "examples/make_sample.py"],
                       cwd=ROOT, check=True)


# ---------- 池守卫 ----------

def test_load_without_pool_is_refused():
    with pytest.raises(PoolViolation):
        list(load(CLEAN))


def test_load_with_pool_filters():
    cal = list(load(CLEAN, pool="calibration"))
    assert cal and all(r.pool == "calibration" for r in cal)


def test_all_pools_reads_everything():
    assert len(list(load(CLEAN, pool=ALL_POOLS))) == 120


# ---------- 逐条不变量 ----------

def test_unsafe_without_events_is_refused():
    """没有 events 就没有 ν,检测延迟无法测量。"""
    r = json.loads(next(l for l in CLEAN.read_text().splitlines()[1:]))
    r["label"] = {"safe": False, "categories": ["C1_violence"], "events": []}
    with pytest.raises(ValueError, match="ν"):
        ClipRecord.from_json(r)


def test_event_requires_positive_duration():
    with pytest.raises(ValueError, match="t_end_s"):
        Event(event_id="e", category="c", t_start_s=5.0, t_end_s=5.0,
              frame_start=0, frame_end=0, severity="high",
              evidence_modality=["pixel"])


def test_non_uniform_reencode_is_refused():
    """压缩域 sentinel 会读编码器指纹,不统一重编码等于自埋捷径。"""
    with pytest.raises(ValueError, match="统一重编码"):
        Splice(host_id="h", n_cuts=0, cuts=[], reencoded_uniformly=False)


def test_fp_bait_must_be_benign():
    r = json.loads(CLEAN.read_text().splitlines()[1])
    r["label"] = {"safe": False, "categories": ["C1_violence"],
                  "events": r["label"]["events"] or [{
                      "event_id": "e0", "category": "C1_violence",
                      "t_start_s": 1.0, "t_end_s": 2.0, "frame_start": 30,
                      "frame_end": 60, "severity": "high",
                      "evidence_modality": ["pixel"]}]}
    r["metadata_adversarial"]["condition"] = "fp_bait"
    with pytest.raises(ValueError, match="fp_bait"):
        ClipRecord.from_json(r)


# ---------- 数据集级校验 ----------

def test_clean_manifest_has_no_errors():
    errs = [f for f in V.validate(str(CLEAN)) if f.level == "error"]
    assert errs == [], f"clean 不应有 error: {[str(e) for e in errs]}"


def test_dirty_manifest_is_caught():
    checks = {f.check for f in V.validate(str(DIRTY)) if f.level == "error"}
    assert "splice_balanced" in checks
    assert "cut_position_leakage" in checks


def test_leakage_check_does_not_cry_wolf_on_clean():
    """回归:固定 TVD 阈值曾在 clean 上误报,已改为置换检验。"""
    recs = list(load(CLEAN, pool=ALL_POOLS))
    out = V.check_cut_position_leakage(recs)
    assert not [f for f in out if f.level == "error"]


def test_calibration_sample_size_uses_exact_binomial():
    """45 = ceil(ln 0.10 / ln 0.95),recall>=95% @ 90% 置信、零失败。"""
    recs = list(load(CLEAN, pool=ALL_POOLS))
    out = V.check_calibration_sample_size(recs)
    assert out and "45" in str(out[0])
