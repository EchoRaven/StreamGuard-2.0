"""合成流水线端到端测试(需要 ffmpeg)。

这些测试真的渲染视频并**抽帧比对像素** —— 时间轴算术正确不等于 ν 指向
正确的画面。第 13 轮教训:算对不等于像素对。
"""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.synth import InsertClip, build_clip, plan_cuts
from sg2.synth import ffmpeg as F

pytestmark = pytest.mark.slow

W = H = 160


def _have_ffmpeg() -> bool:
    try:
        F.ffmpeg_exe()
        return True
    except Exception:
        return False


skip_no_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="缺 ffmpeg")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    host = F.make_test_video(d / "host.mp4", duration_s=60.0,
                             pattern="testsrc2", freq=200, size=f"{W}x{H}")
    needle = F.make_test_video(d / "n.mp4", duration_s=3.0,
                               pattern="smptebars", freq=900, size=f"{W}x{H}")
    benign = F.make_test_video(d / "b.mp4", duration_s=3.0,
                               pattern="testsrc", freq=300, size=f"{W}x{H}")
    rng = random.Random(11)
    plan = plan_cuts("host", 60.0, is_positive=True, duration_pool=[3.0],
                     rng=rng, n_cuts_range=(2, 2))
    rec = build_clip(plan, host,
                     [InsertClip("n", str(needle), 3.0, category="C1_violence")],
                     [InsertClip("b", str(benign), 3.0)],
                     d / "out.mp4", clip_id="t-0001", rng=rng,
                     workdir=d / "w")
    return d, rec


@skip_no_ffmpeg
def test_record_passes_schema(built):
    _, rec = built
    assert not rec.is_safe and len(rec.events) == 1
    assert rec.splice.reencoded_uniformly


@skip_no_ffmpeg
def test_output_duration_accounts_for_inserts(built):
    d, rec = built
    info = F.probe(d / "out.mp4")
    expected = 60.0 + rec.splice.n_cuts * 3.0
    assert abs(info.duration_s - expected) < 1.5, \
        f"输出 {info.duration_s:.2f}s vs 期望 {expected:.2f}s"


@skip_no_ffmpeg
def test_ground_truth_points_at_needle_pixels(built):
    """ν 必须指向 needle 的画面,不能只是算术上对。"""
    d, rec = built
    e = rec.events[0]

    def frame(t):
        r = subprocess.run(
            [F.ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}",
             "-i", str(d / "out.mp4"), "-frames:v", "1", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], capture_output=True)
        a = np.frombuffer(r.stdout, np.uint8)
        need = W * H * 3
        return a[:need].reshape(H, W, 3).astype(float) if a.size >= need else None

    mid = (e.t_start_s + e.t_end_s) / 2
    f_needle = frame(mid)
    f_host1 = frame(max(1.0, e.t_start_s - 8))
    f_host2 = frame(min(rec.media["duration_s"] - 1, e.t_end_s + 5))
    assert all(x is not None for x in (f_needle, f_host1, f_host2))

    d_needle = np.abs(f_needle - f_host1).mean()
    d_host = np.abs(f_host1 - f_host2).mean()
    assert d_needle > 3 * d_host, \
        f"ν 处画面与宿主无区别 (needle差={d_needle:.1f} 宿主差={d_host:.1f})"


@skip_no_ffmpeg
def test_cut_timestamps_are_on_output_timeline(built):
    """切点时间戳必须在输出时间轴上,不是宿主时间轴。"""
    _, rec = built
    ts = [c.t_s for c in rec.splice.cuts]
    assert ts == sorted(ts)
    assert max(ts) <= rec.media["duration_s"]
    needle_cut = next(c for c in rec.splice.cuts if c.kind == "needle")
    assert abs(needle_cut.t_s - rec.events[0].t_start_s) < 0.1


@skip_no_ffmpeg
def test_probe_fallback_without_ffprobe(built):
    """ffprobe 缺失时回退解析必须给出正确的时长与帧率。"""
    d, _ = built
    info = F._probe_via_ffmpeg(str(d / "out.mp4"))
    assert info.fps == pytest.approx(30.0, abs=0.5)
    assert info.duration_s > 60.0
    assert info.width == W and info.height == H
