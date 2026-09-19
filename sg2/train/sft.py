"""SFT 数据构造:从 manifest 展开成**因果窗口**。

设计见 docs/08_SFT_RL.md §2。两条最容易做错的:

1. **窗口必须因果** —— 只含 [t-w, t],绝不含未来帧。训练数据里混进未来帧,
   模型会学到一个在真实流上根本不存在的能力,而离线指标看不出来。
2. **必须有"部分覆盖"窗口** —— 只有"证据齐全"和"完全没有"两种,模型学不会
   `hold`,在真实流上被迫在证据不全时硬猜。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterator, Literal

from ..schema import ClipRecord, Event

WindowKind = Literal["pos_full", "pos_partial", "neg_same", "neg_cross"]

# docs/08_SFT_RL.md §2.1 的目标配比
DEFAULT_MIX: dict[WindowKind, float] = {
    "pos_full": 0.25, "pos_partial": 0.25,
    "neg_same": 0.35, "neg_cross": 0.15,
}


@dataclass(frozen=True)
class Window:
    clip_id: str
    t_start_s: float
    t_end_s: float
    kind: WindowKind
    action: str                      # hold | flag | clear
    category: str | None = None
    evidence_frames: tuple[int, ...] = ()
    policy_citation: str | None = None

    def __post_init__(self):
        if self.t_end_s <= self.t_start_s:
            raise ValueError("窗口时长必须为正")
        if self.action == "flag" and not self.policy_citation:
            raise ValueError(f"{self.clip_id}: flag 目标必须带 policy_citation")
        if self.action != "flag" and self.policy_citation:
            raise ValueError(f"{self.clip_id}: 非 flag 不应带 citation")

    def target(self) -> str:
        d: dict = {"action": self.action}
        if self.action == "flag":
            d.update(category=self.category,
                     evidence_frames=list(self.evidence_frames),
                     policy_citation=self.policy_citation)
        return json.dumps(d, ensure_ascii=False)


def _citation_for(category: str) -> str:
    """类别 -> 政策条款 id。真实实现查 spec/taxonomy;这里用确定性映射。"""
    return f"SW-{category}"


def windows_for(rec: ClipRecord, *, window_s: float = 8.0,
                partial_frac: float = 0.4, rng: random.Random | None = None,
                max_neg_per_clip: int = 4) -> Iterator[Window]:
    """把一条记录展开成若干因果窗口。

    Args:
        partial_frac: "部分覆盖"窗口覆盖 event 的比例。0.4 表示只看到
            event 的前 40% —— 此时正确动作是 `hold` 而非 `flag`。
    """
    rng = rng or random.Random(0)
    dur = float(rec.media["duration_s"])
    events: list[Event] = rec.events

    for e in events:
        # 正例-完整:窗口右端在 event 结束之后,完整证据可见
        end = min(dur, e.t_end_s + 0.5)
        start = max(0.0, end - window_s)
        if end > start:
            yield Window(rec.id, start, end, "pos_full", "flag",
                         category=e.category,
                         evidence_frames=(e.frame_start, e.frame_end),
                         policy_citation=_citation_for(e.category))

        # 正例-部分:窗口右端落在 event 内部,只看到前 partial_frac
        cut = e.t_start_s + (e.t_end_s - e.t_start_s) * partial_frac
        pstart = max(0.0, cut - window_s)
        if cut > pstart:
            yield Window(rec.id, pstart, cut, "pos_partial", "hold")

    # 负例-同视频:与任何 event 都不相交。同场景同编码,是有价值的难例
    occupied = [(e.t_start_s, e.t_end_s) for e in events]
    cands = []
    t = window_s
    while t <= dur:
        if not any(a - window_s < t < b + window_s for a, b in occupied):
            cands.append(t)
        t += window_s
    rng.shuffle(cands)
    kind: WindowKind = "neg_cross" if not events else "neg_same"
    for t in cands[:max_neg_per_clip]:
        yield Window(rec.id, t - window_s, t, kind, "clear")


def resample_to_mix(windows: list[Window], mix: dict[str, float] | None = None,
                    *, rng: random.Random | None = None) -> list[Window]:
    """按目标配比重采样。

    不重采样会让配比被数据集的正负比直接决定 —— 实测 40 条 demo 上
    neg_cross 占到 47%,把正例稀释到 8.8%,而 pos_partial 正是模型学
    `hold` 的唯一来源。

    以**最受限的那一类**定总量:对每一类取 n_k/target_k 的最小值,
    保证不对任何一类做有放回过采样(过采样会让模型记住少数几个窗口)。
    """
    mix = mix or DEFAULT_MIX
    rng = rng or random.Random(0)
    by_kind: dict[str, list[Window]] = {}
    for w in windows:
        by_kind.setdefault(w.kind, []).append(w)

    feasible = [len(by_kind.get(k, [])) / t
                for k, t in mix.items() if t > 0]
    if not feasible or min(feasible) == 0:
        return list(windows)          # 某类为空,无法配比,原样返回
    total = int(min(feasible))

    out: list[Window] = []
    for k, t in mix.items():
        pool = by_kind.get(k, [])
        want = min(int(round(total * t)), len(pool))
        out.extend(rng.sample(pool, want))
    rng.shuffle(out)
    return out


def build_sft_jsonl(records: list[ClipRecord], out_path: str, *,
                    video_dir: str, window_s: float = 8.0,
                    seed: int = 0, balance: bool = True) -> dict[str, int]:
    """写出 ms-swift 可直接消费的 SFT jsonl。

    Returns:
        各窗口类型的计数,用于核对是否接近 DEFAULT_MIX。
    """
    rng = random.Random(seed)
    allw: list[Window] = []
    for rec in records:
        allw.extend(windows_for(rec, window_s=window_s, rng=rng))
    if balance:
        allw = resample_to_mix(allw, rng=rng)

    counts: dict[str, int] = {}
    with open(out_path, "w") as fh:
        for w in allw:
            counts[w.kind] = counts.get(w.kind, 0) + 1
            fh.write(json.dumps({
                "messages": [
                    {"role": "user",
                     "content": "<video>根据政策判断该片段。"
                                "证据不足时回 hold。只输出 JSON。"},
                    {"role": "assistant", "content": w.target()},
                ],
                "videos": [f"{video_dir}/{w.clip_id}.mp4"],
                "_window": {"start_s": w.t_start_s, "end_s": w.t_end_s,
                            "kind": w.kind},
            }, ensure_ascii=False) + "\n")
    return counts


def mix_report(counts: dict[str, int]) -> str:
    """把实际配比与目标配比对照,偏离过大时应重采样。"""
    total = sum(counts.values()) or 1
    lines = ["窗口类型        实际      目标"]
    for k, target in DEFAULT_MIX.items():
        got = counts.get(k, 0) / total
        flag = "" if abs(got - target) < 0.12 else "  ← 偏离"
        lines.append(f"  {k:<12} {got:6.1%}   {target:5.0%}{flag}")
    return "\n".join(lines)
