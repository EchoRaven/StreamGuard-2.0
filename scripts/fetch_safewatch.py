#!/usr/bin/env python3
"""拉取 SafeWatch-Bench 并转成 SG2 格式。

⚠️ 这是 **gated 仓库**,需要先在 HF 页面申请并等作者批准,再提供 token:
    export HF_TOKEN=hf_...
    python scripts/fetch_safewatch.py --meta-only

⚠️ **SafeWatch-Bench 的标注是视频级的,没有任何时间戳。**
全库 1400 条实测:带时序字段的条目 = 0。字段只有
`video_path / labels / subcategories / video_content / violate_reason`。

后果(见 docs/02_DATASET.md §2.2):
  - 拿不到 ν,**检测延迟 E[(τ−ν)⁺] 在这个数据集上无真值**
  - 只能用于"整条视频安全与否"的判定,不能用于流式定位
  - 要测延迟必须把它的片段**当作 needle 插进长宿主视频**(合成流水线),
    此时 ν 由我们自己的拼接规划给出

所以本脚本的产物默认 `axes.temporal_sparsity.ratio = 1.0`(整条都是
needle),并把 `source.kind` 标为 `native` —— 它不是合成的,也没有 ν。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "Virtue-AI-HUB/SafeWatch-Bench"

# 实测确认的官方类别名(从 violate_reason 的 "category Cn (...)" 抽取)
SAFEWATCH_LABELS = {
    1: "Sexual Content",
    2: "Harassment & Bullying",
    3: "Threats, Violence & Harm",
    4: "False & Deceptive Information",     # C4 的 reason 不用括号格式
    5: "Illegal/Regulated Activities",
    6: "Hateful Content & Extremism",
}

# 子任务 -> 我们的难度轴。SafeWatch 自己就区分了明显与隐晦,
# 这给 A/B/C 分层一个免费的初始划分(仍需 2x2 自动分层校准)。
SUBTLETY_HINT = {
    "evident": "salient", "Evident": "salient",
    "subtle": "subtle", "implication": "subtle",
    "Extremely Disturbing Content": "salient",
}


def _hub():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit(
            "需要 HF_TOKEN。SafeWatch-Bench 是 gated 仓库:\n"
            "  1. 在 https://huggingface.co/datasets/" + REPO + " 申请访问\n"
            "  2. export HF_TOKEN=hf_...")
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi(), hf_hub_download, tok


def load_annotations() -> list[dict]:
    api, dl, tok = _hub()
    out = []
    for f in api.list_repo_files(REPO, repo_type="dataset", token=tok):
        if not f.endswith(".json"):
            continue
        for e in json.load(open(dl(REPO, f, repo_type="dataset", token=tok))):
            e["_src"] = f
            out.append(e)
    return out


def summarise(anns: list[dict]) -> dict:
    keys, labels, subs = Counter(), Counter(), Counter()
    temporal = 0
    per_split = Counter()
    for e in anns:
        keys.update(k for k in e if not k.startswith("_"))
        for lab in e.get("labels") or []:
            labels[lab] += 1
        for s in e.get("subcategories") or []:
            subs[s] += 1
        if any(t in k.lower() for k in e
               for t in ("time", "frame", "start", "end", "stamp")):
            temporal += 1
        per_split[e["_src"].split("/")[0]] += 1
    return {"n": len(anns), "fields": dict(keys),
            "temporal_entries": temporal,
            "labels": dict(sorted(labels.items())),
            "n_subcategories": len(subs),
            "top_subcategories": dict(subs.most_common(10)),
            "splits": dict(per_split),
            "n_benign": sum(1 for e in anns if not e.get("labels"))}


def to_sg2(e: dict) -> dict:
    """转成 SPEC v1.0 的 ClipRecord。

    ⚠️ 没有 ν,所以 events 用**整条视频**作为区间,并在 provenance 里
    标明 `no_temporal_groundtruth` —— 下游算延迟时必须据此排除。
    """
    labs = e.get("labels") or []
    cats = [f"C{l}_" + SAFEWATCH_LABELS.get(l, "unknown").split()[0].lower()
            for l in labs]
    subs = e.get("subcategories") or []
    bin_ = next((SUBTLETY_HINT[s] for s in subs if s in SUBTLETY_HINT),
                "moderate")
    return {
        "id": "sw-" + e["video_path"].replace("/", "_").rsplit(".", 1)[0],
        # 溯源放进 source 而不是另开顶层字段 —— schema 不接受未知顶层键,
        # 而 video_path 是下载视频的唯一依据,不能丢。
        "source": {"kind": "native", "provenance": "safewatch-bench",
                   "license": "gated:manual",
                   "note": "no_temporal_groundtruth",
                   "video_path": e["video_path"],
                   "safewatch_labels": labs,
                   "safewatch_subcategories": subs,
                   "safewatch_json": e["_src"]},
        # ⚠️ 未下视频时时长未知,用 None **而不是 0**。填 0 是在假装知道
        # 一个不知道的值,下游按时长归一化(如 temporal_sparsity.ratio)
        # 会除零或算出无意义的数。
        "media": {"duration_s": None, "fps": None, "codec": "h264"},
        "label": {"safe": not labs, "categories": cats, "events": []},
        "axes": {
            "temporal_sparsity": {"needle_total_s": None,
                                  "video_duration_s": None,
                                  "ratio": 1.0 if labs else 0.0,
                                  "bin": "dense"},
            "perceptual_subtlety": {"min_evidence_pixel_area_frac": 0.1,
                                    "requires_ocr": False, "bin": bin_},
            "modality_locus": {"decisive": ["pixel"], "bin": "visual"},
            "context_dependence": {"window_required_s": 0.0,
                                   "bin": "self_contained"},
        },
        "metadata_adversarial": {"title": e.get("video_content", "")[:200],
                                 "description": e.get("violate_reason", "")[:400],
                                 "asr_path": None, "chat_path": None,
                                 "condition": "aligned"},
        "splice": None, "split": "test", "pool": "eval",
        # ★ SafeWatch-Bench 没有时间戳,必须标成 video_level ——
        # 它会被排除出检测延迟的统计。
        "granularity": "video_level",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-only", action="store_true", help="只拉标注,不下视频")
    ap.add_argument("--out", default="/data/common/haibotong/safewatch")
    ap.add_argument("--n-videos", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    anns = load_annotations()
    s = summarise(anns)

    print(f"标注 {s['n']} 条")
    print(f"字段: {s['fields']}")
    print(f"⚠️ 带时序字段的条目: {s['temporal_entries']}  "
          f"{'<- 视频级标注,无 ν' if not s['temporal_entries'] else ''}")
    print(f"类别分布: {s['labels']}")
    print(f"benign: {s['n_benign']}  子任务种类: {s['n_subcategories']}")
    print(f"split: {s['splits']}")

    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    recs = [to_sg2(e) for e in anns]
    with open(out / "manifest_safewatch.jsonl", "w") as fh:
        fh.write(json.dumps({"__meta__": {"spec_version": "1.0",
                                          "source": REPO,
                                          "no_temporal_groundtruth": True}}) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n写入 {out}/manifest_safewatch.jsonl ({len(recs)} 条)")

    if a.n_videos:
        api, dl, tok = _hub()
        got = 0
        for r in recs[:a.n_videos]:
            try:
                dl(REPO, r["source"]["video_path"], repo_type="dataset",
                   token=tok)
                got += 1
            except Exception as e:                    # noqa: BLE001
                print(f"  {r['id']}: {type(e).__name__}")
        print(f"下载 {got}/{a.n_videos} 个视频")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
