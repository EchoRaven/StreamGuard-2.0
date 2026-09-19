"""执行拼接规划:产出视频 + 符合 SPEC v1.0 的 ClipRecord。"""
from __future__ import annotations

import hashlib
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .. import bins
from ..schema import ClipRecord, Cut, Splice
from . import ffmpeg as F
from .plan import SplicePlan


@dataclass(frozen=True)
class InsertClip:
    """一个待插入片段。needle 带类别与证据属性;benign 不带。"""
    clip_id: str
    path: str
    duration_s: float
    category: str | None = None
    evidence_modality: tuple[str, ...] = ("pixel",)
    pixel_area_frac: float = 0.05
    requires_ocr: bool = False
    window_required_s: float = 0.0


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def build_clip(plan: SplicePlan, host_path: str | Path,
               needle_pool: list[InsertClip], benign_pool: list[InsertClip],
               out_video: str | Path, *, clip_id: str, rng: random.Random,
               pool: str = "eval", split: str = "test",
               condition: str = "aligned",
               workdir: str | Path | None = None) -> ClipRecord:
    """把一个 SplicePlan 渲染成视频,并返回对应的 ClipRecord。

    宿主被切成若干段,插入片段夹在中间,整体统一重编码。
    事件的 t_start_s 按**输出时间轴**计算 —— 那才是 ν。
    """
    host_path = Path(host_path)
    out_video = Path(out_video)
    tmp = Path(workdir or tempfile.mkdtemp(prefix="sg2synth_"))
    tmp.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    events: list[dict] = []
    cuts_meta: list[Cut] = []
    out_t = 0.0          # 输出时间轴游标
    host_t = 0.0         # 宿主时间轴游标

    for i, c in enumerate(sorted(plan.cuts, key=lambda x: x.t_s)):
        # 宿主段:从 host_t 到本切点
        seg_dur = c.t_s - host_t
        if seg_dur > 0.05:
            seg = F.cut(host_path, tmp / f"{clip_id}_h{i}.mp4", host_t, seg_dur)
            parts.append(seg)
            out_t += seg_dur
        host_t = c.t_s

        # 插入片段:按规划的时长从对应池里挑一个最接近的
        src_pool = needle_pool if c.kind == "needle" else benign_pool
        if not src_pool:
            raise ValueError(f"{c.kind} 池为空")
        ins = min(src_pool, key=lambda x: abs(x.duration_s - c.duration_s))
        use_dur = min(ins.duration_s, c.duration_s)
        piece = F.cut(ins.path, tmp / f"{clip_id}_i{i}.mp4", 0.0, use_dur)
        parts.append(piece)

        cuts_meta.append(Cut(t_s=round(out_t, 3), kind=c.kind,
                             insert_id=ins.clip_id))
        if c.kind == "needle":
            events.append({
                "event_id": f"e{len(events)}",
                "category": ins.category or "UNKNOWN",
                "t_start_s": round(out_t, 3),
                "t_end_s": round(out_t + use_dur, 3),
                "frame_start": int(out_t * F.NORM["fps"]),
                "frame_end": int((out_t + use_dur) * F.NORM["fps"]),
                "severity": "high",
                "evidence_modality": list(ins.evidence_modality),
                "sufficient_alone": list(ins.evidence_modality),
                "annotator_ids": ["synthetic"],
                "agreement": 1.0,
            })
        out_t += use_dur

    # 宿主尾段
    host_info = F.probe(host_path)
    tail = host_info.duration_s - host_t
    if tail > 0.05:
        parts.append(F.cut(host_path, tmp / f"{clip_id}_htail.mp4", host_t, tail))
        out_t += tail

    F.concat_reencode(parts, out_video)
    info = F.probe(out_video)

    needle_total = sum(e["t_end_s"] - e["t_start_s"] for e in events)
    ratio = needle_total / info.duration_s if info.duration_s else 0.0
    area = min((ins.pixel_area_frac for ins in needle_pool), default=0.05) \
        if events else 0.0
    ocr_req = any(e for e in events if "ocr" in e["evidence_modality"])
    decisive = sorted({m for e in events for m in e["evidence_modality"]}) or ["pixel"]
    win = max((ins.window_required_s for ins in needle_pool), default=0.0) \
        if events else 0.0

    return ClipRecord(
        id=clip_id,
        source={"kind": "synthetic", "provenance": plan.host_id,
                "license": "derived"},
        media={"duration_s": round(info.duration_s, 3), "fps": info.fps,
               "resolution": [info.width, info.height], "codec": info.codec,
               "sha256": _sha256(out_video),
               "reencode": {"crf": F.NORM["crf_final"],
                            "preset": F.NORM["preset"], "gop": F.NORM["gop"]}},
        label={"safe": not events,
               "categories": sorted({e["category"] for e in events}),
               "events": events},
        axes={
            "temporal_sparsity": {
                "needle_total_s": round(needle_total, 3),
                "video_duration_s": round(info.duration_s, 3),
                "ratio": round(ratio, 8),
                "bin": bins.temporal_sparsity_bin(ratio)},
            "perceptual_subtlety": {
                "min_evidence_pixel_area_frac": area,
                "evidence_contrast": 0.3,
                "evidence_span_frames": int(needle_total * F.NORM["fps"]),
                "requires_ocr": ocr_req,
                "bin": bins.perceptual_subtlety_bin(area, ocr_req)},
            "modality_locus": {
                "decisive": decisive,
                "pixel_only_sufficient": decisive == ["pixel"],
                "bin": bins.modality_locus_bin(decisive)},
            "context_dependence": {
                "frame_alone_sufficient": win == 0.0,
                "window_required_s": win,
                "bin": bins.context_dependence_bin(win)},
        },
        metadata_adversarial={"title": clip_id, "description": "",
                              "asr_path": None, "chat_path": None,
                              "condition": condition},
        splice=Splice(host_id=plan.host_id, n_cuts=len(cuts_meta),
                      cuts=cuts_meta,
                      audio_crossfade_ms=F.NORM["crossfade_ms"],
                      loudness_lufs=F.NORM["loudness_lufs"],
                      reencoded_uniformly=True),
        split=split, pool=pool,
    )
