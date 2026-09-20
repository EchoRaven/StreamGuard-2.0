"""SG2 ClipRecord 数据结构与 loader。

设计要点:loader 强制三池分离。校准集被当成 exemplar 使用会使 conformal
保证失效,而这种错误写在文档里一定会被违反,所以写进类型层。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal, Sequence

SPEC_VERSION = "1.0"

Modality = Literal["pixel", "speech", "ocr", "motion", "composition"]
Pool = Literal["exemplar", "compile", "calibration", "eval"]
Split = Literal["train", "calib", "test", "zeroday_holdout"]
# 标注粒度。**不是可选的元信息** —— 它决定这条记录能不能进检测延迟的
# 统计。SafeWatch-Bench 实测 1400 条全是 video_level(带时序字段的 0 条),
# 只能用于"整条视频安全与否",不能用于流式定位。
Granularity = Literal["temporal", "video_level"]
MetaCondition = Literal["aligned", "fp_bait", "camouflage"]


class PoolViolation(RuntimeError):
    """试图跨池读取。这不是警告,是错误——它会静默毁掉校准。"""


@dataclass
class Event:
    """一个不安全事件。t_start_s 就是 quickest-detection 里的 ν。"""
    event_id: str
    category: str
    t_start_s: float
    t_end_s: float
    frame_start: int
    frame_end: int
    severity: Literal["low", "medium", "high"]
    evidence_modality: list[Modality]
    sufficient_alone: list[Modality] = field(default_factory=list)
    bbox_track: list | None = None
    annotator_ids: list[str] = field(default_factory=list)
    agreement: float | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    def __post_init__(self):
        if self.t_end_s <= self.t_start_s:
            raise ValueError(f"{self.event_id}: t_end_s 必须 > t_start_s")
        if not self.evidence_modality:
            raise ValueError(f"{self.event_id}: evidence_modality 不能为空")


@dataclass
class Cut:
    t_s: float
    kind: Literal["benign", "needle"]
    insert_id: str | None = None


@dataclass
class Splice:
    host_id: str
    n_cuts: int
    cuts: list[Cut]
    audio_crossfade_ms: int = 40
    loudness_lufs: float = -23.0
    reencoded_uniformly: bool = True

    def __post_init__(self):
        if self.n_cuts != len(self.cuts):
            raise ValueError(f"{self.host_id}: n_cuts 与 cuts 长度不一致")
        if not self.reencoded_uniformly:
            raise ValueError(
                f"{self.host_id}: 必须统一重编码。压缩域 sentinel 会读编码器指纹,"
                "不统一等于给检测器留了一条与标签相关的捷径")


@dataclass
class ClipRecord:
    id: str
    source: dict
    media: dict
    label: dict
    axes: dict
    metadata_adversarial: dict
    splice: Splice | None
    split: Split
    pool: Pool
    granularity: Granularity = "temporal"

    @property
    def events(self) -> list[Event]:
        return [Event(**e) if isinstance(e, dict) else e
                for e in self.label.get("events", [])]

    @property
    def is_safe(self) -> bool:
        return bool(self.label.get("safe", False))

    @property
    def has_nu(self) -> bool:
        """是否有变点真值。**算检测延迟前必须先问这个。**

        视频级标注的数据(如 SafeWatch-Bench)拿不到 ν,把它算进
        E[(τ−ν)⁺] 会得到一个没有意义的数。
        """
        return self.granularity == "temporal" and bool(self.label.get("events"))

    @property
    def needle_total_s(self) -> float:
        return sum(e.duration_s for e in self.events)

    def __post_init__(self):
        if self.is_safe and self.label.get("events"):
            raise ValueError(f"{self.id}: safe=true 但有 events")
        if (not self.is_safe and not self.label.get("events")
                and self.granularity == "temporal"):
            raise ValueError(
                f"{self.id}: safe=false 但没有 events —— 没有 ν 就测不了"
                "检测延迟。若数据本身就只有视频级标注,请显式设 "
                'granularity="video_level"(它会被排除出延迟统计)')
        if self.granularity == "video_level" and self.label.get("events"):
            raise ValueError(
                f"{self.id}: granularity=video_level 却带 events —— "
                "有 ν 就该标成 temporal")
        if self.source.get("kind") == "native" and self.splice is not None:
            raise ValueError(f"{self.id}: native 来源不应有 splice")
        cond = self.metadata_adversarial.get("condition")
        if cond == "fp_bait" and not self.is_safe:
            raise ValueError(f"{self.id}: fp_bait 必须是 benign 视频")
        if cond == "camouflage" and self.is_safe:
            raise ValueError(f"{self.id}: camouflage 必须是 unsafe 视频")

    @classmethod
    def from_json(cls, d: dict) -> "ClipRecord":
        sp = d.get("splice")
        if sp is not None:
            sp = Splice(**{**sp, "cuts": [Cut(**c) for c in sp["cuts"]]})
        return cls(**{**d, "splice": sp})

    def to_json(self) -> dict:
        return asdict(self)


def load(manifest: str | Path, *, pool: Pool | Sequence[Pool] | None = None,
         split: Split | Sequence[Split] | None = None) -> Iterator[ClipRecord]:
    """读 manifest。

    pool 必须显式指定。这是故意的:不写 pool 就想读全量数据的代码,
    正是会把校准集喂进 exemplar 的那种代码。
    """
    if pool is None:
        raise PoolViolation(
            "load() 必须显式指定 pool=。三池分离是 conformal 保证的前提,"
            "见 SPEC.md §3。确实要全量请写 pool=ALL_POOLS。")
    wanted_pool = {pool} if isinstance(pool, str) else set(pool)
    wanted_split = ({split} if isinstance(split, str)
                    else set(split) if split else None)

    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "__meta__" in d:
                if d["__meta__"].get("spec_version") != SPEC_VERSION:
                    raise ValueError(f"spec_version 不匹配: {d['__meta__']}")
                continue
            if d["pool"] not in wanted_pool:
                continue
            if wanted_split and d["split"] not in wanted_split:
                continue
            yield ClipRecord.from_json(d)


ALL_POOLS: tuple[Pool, ...] = ("exemplar", "compile", "calibration", "eval")
