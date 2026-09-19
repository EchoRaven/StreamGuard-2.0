"""ffmpeg 薄封装。

统一重编码是硬性要求,不是优化项:压缩域 sentinel 读的就是编码器指纹,
各片段保留各自的量化参数与 GOP 结构会成为与标签相关的捷径。
因此本模块**不提供 `-c copy` 的路径**。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# 归一化与重编码的统一参数(docs/02_DATASET.md §3.2)
NORM = dict(width=1280, height=720, fps=30, crf_norm=20, crf_final=23,
            preset="medium", gop=250, loudness_lufs=-23.0, crossfade_ms=40)


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    for getter in (
        lambda: shutil.which("ffmpeg"),
        lambda: __import__("imageio_ffmpeg").get_ffmpeg_exe(),
    ):
        try:
            p = getter()
            if p:
                return p
        except Exception:
            continue
    raise FFmpegMissing("找不到 ffmpeg。pip install imageio-ffmpeg")


def ffprobe_exe() -> str | None:
    """ffprobe 可能缺失(imageio-ffmpeg 只带 ffmpeg)。缺失时回退到 ffmpeg 解析。"""
    return shutil.which("ffprobe")


def _run(args: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "")[-800:]
        raise RuntimeError(f"ffmpeg 失败 (rc={r.returncode}):\n{tail}")
    return r


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    has_audio: bool


def probe(path: str | Path) -> MediaInfo:
    """读媒体信息。优先 ffprobe,缺失时从 ffmpeg 的 stderr 解析。"""
    path = str(path)
    fp = ffprobe_exe()
    if fp:
        r = subprocess.run(
            [fp, "-v", "error", "-show_format", "-show_streams",
             "-of", "json", path], capture_output=True, text=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
            if v is None:
                raise RuntimeError(f"{path}: 无视频流")
            num, den = (v.get("r_frame_rate") or "30/1").split("/")
            return MediaInfo(
                duration_s=float(d["format"]["duration"]),
                fps=float(num) / float(den or 1),
                width=int(v["width"]), height=int(v["height"]),
                codec=v.get("codec_name", "?"),
                has_audio=any(s["codec_type"] == "audio" for s in d["streams"]))
    return _probe_via_ffmpeg(path)


def _probe_via_ffmpeg(path: str) -> MediaInfo:
    r = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    err = r.stderr
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    if not m:
        raise RuntimeError(f"{path}: 无法解析时长\n{err[-400:]}")
    h, mi, s = m.groups()
    dur = int(h) * 3600 + int(mi) * 60 + float(s)
    v = re.search(r"Video:\s*(\w+).*?(\d{2,5})x(\d{2,5}).*?([\d.]+)\s*fps", err, re.S)
    if not v:
        raise RuntimeError(f"{path}: 无法解析视频流")
    codec, w, hgt, fps = v.groups()
    return MediaInfo(duration_s=dur, fps=float(fps), width=int(w),
                     height=int(hgt), codec=codec,
                     has_audio="Audio:" in err)


def normalize(src: str | Path, dst: str | Path, *, with_audio: bool = True) -> Path:
    """统一分辨率/帧率/响度。宿主与插入片段在拼接前都要过这一步。"""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"scale={NORM['width']}:{NORM['height']}:force_original_aspect_ratio=decrease,"
          f"pad={NORM['width']}:{NORM['height']}:(ow-iw)/2:(oh-ih)/2,"
          f"fps={NORM['fps']},setsar=1")
    args = [ffmpeg_exe(), "-y", "-i", str(src), "-vf", vf,
            "-c:v", "libx264", "-crf", str(NORM["crf_norm"]),
            "-preset", NORM["preset"], "-g", str(NORM["gop"]),
            "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-af", f"loudnorm=I={NORM['loudness_lufs']}:LRA=7:TP=-2",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    _run(args + [str(dst)])
    return dst


def cut(src: str | Path, dst: str | Path, start_s: float, dur_s: float) -> Path:
    """裁剪片段。重编码而非 -c copy —— 关键帧对齐会让 -c copy 的实际时长漂移。"""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_exe(), "-y", "-ss", f"{start_s:.3f}", "-i", str(src),
          "-t", f"{dur_s:.3f}",
          "-c:v", "libx264", "-crf", str(NORM["crf_norm"]),
          "-preset", NORM["preset"], "-g", str(NORM["gop"]),
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
          "-avoid_negative_ts", "make_zero", str(dst)])
    return dst


def concat_reencode(parts: list[str | Path], dst: str | Path) -> Path:
    """按顺序拼接并**统一重编码**。

    严禁 -c copy:那会保留各片段的编码器指纹,成为与标签相关的捷径。
    """
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in parts:
            fh.write(f"file '{Path(p).resolve()}'\n")
        listfile = fh.name
    try:
        _run([ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", listfile,
              "-c:v", "libx264", "-crf", str(NORM["crf_final"]),
              "-preset", NORM["preset"], "-g", str(NORM["gop"]),
              "-pix_fmt", "yuv420p",
              "-af", f"loudnorm=I={NORM['loudness_lufs']}:LRA=7:TP=-2",
              "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
              str(dst)])
    finally:
        Path(listfile).unlink(missing_ok=True)
    return dst


def make_test_video(dst: str | Path, *, duration_s: float, pattern: str = "testsrc2",
                    freq: int = 440, size: str | None = None) -> Path:
    """用 lavfi 生成测试视频。让合成流水线无需真实素材即可端到端验证。"""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    size = size or f"{NORM['width']}x{NORM['height']}"
    _run([ffmpeg_exe(), "-y",
          "-f", "lavfi", "-i", f"{pattern}=size={size}:rate={NORM['fps']}:duration={duration_s}",
          "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration_s}",
          "-c:v", "libx264", "-crf", str(NORM["crf_norm"]),
          "-preset", "ultrafast", "-g", str(NORM["gop"]), "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
          "-shortest", str(dst)])
    return dst
