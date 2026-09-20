#!/usr/bin/env python3
"""Sentinel 可行性:冻结 SigLIP 到底能不能支撑廉价层?

**这是决定整个级联前提成不成立的实验。** 如果冻结嵌入在真实安全数据上
完全分不开 safe/unsafe,sentinel 层就塌了,"廉价层覆盖 100% 流量"的
设计无从谈起。

⚠️ 先澄清一个常被含糊的点:**冻结 SigLIP 输出的是 1152 维向量,不是判决。**
它自己判不了任何东西。要得到"安全与否"必须在它之上再加一层,而三种加法
的标注需求完全不同:

| 方式 | 需要 | 本脚本记为 |
| --- | --- | --- |
| 零样本文本对比 | 只要提示词 | `zeroshot` |
| 原型 / kNN | k 个带标注正例 | `proto-k` |
| 线性探针 | 全量标注 + 拟合 | `linear` |

所以"training-free"这个说法要精确到**哪一层** frozen:
frontier backbone 冻结是真的,但廉价判别层需要标注。

指标用 **AUPRC 与 AUROC**,不用准确率 —— 正例率不平衡时准确率会骗人。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import EncoderConfig
from sg2.models.encoders import build_encoder
from sg2.synth.ffmpeg import ffmpeg_exe, probe

# SafeWatch 六类 -> 零样本提示。正负各一组,取最大相似度之差。
UNSAFE_PROMPTS = [
    "a video frame showing sexual or nude content",
    "a video frame showing harassment or bullying",
    "a video frame showing violence, fighting, blood or injury",
    "a video frame showing false or deceptive information",
    "a video frame showing illegal activity, drugs or weapons",
    "a video frame showing hateful or extremist content",
]
SAFE_PROMPTS = [
    "a normal everyday video frame",
    "a safe video frame with ordinary content",
    "people talking or doing ordinary activities",
]


def grab(video: str, t_s: float, size: int = 384) -> np.ndarray | None:
    r = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{t_s:.2f}", "-i", video,
         "-frames:v", "1", "-vf", f"scale={size}:{size}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    need = size * size * 3
    return (a[:need].reshape(size, size, 3).astype(np.float32) / 255.
            if a.size >= need else None)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    n_pos = yy.sum()
    return float((prec * yy).sum() / n_pos) if n_pos else float("nan")


def report(name: str, y: np.ndarray, s: np.ndarray, base: float) -> dict:
    a, p = auroc(y, s), auprc(y, s)
    lift = p / base if base else float("nan")
    mark = "✓" if a > 0.65 else ("!" if a > 0.55 else "✗")
    print(f"  {mark} {name:<16} AUROC={a:.3f}  AUPRC={p:.3f} "
          f"(基线 {base:.3f}, 提升 {lift:.2f}×)")
    return {"method": name, "auroc": round(a, 4), "auprc": round(p, 4),
            "lift": round(lift, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/data/common/haibotong/safewatch/manifest_safewatch.jsonl")
    ap.add_argument("--max-videos", type=int, default=200)
    ap.add_argument("--frames-per-video", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    R = "Virtue-AI-HUB/SafeWatch-Bench"

    recs = [json.loads(l) for l in open(a.manifest) if "__meta__" not in l]
    embs, labels, groups = [], [], []
    enc = build_encoder(EncoderConfig(name="siglip2", dtype="float16",
                                      device="cuda:0", batch_size=32))
    n_ok = 0
    for r in recs:
        if n_ok >= a.max_videos:
            break
        try:
            lp = hf_hub_download(R, r["source"]["video_path"],
                                 repo_type="dataset", token=tok,
                                 local_files_only=True)
        except Exception:
            continue
        try:
            dur = probe(lp).duration_s
        except Exception:
            continue
        ts = np.linspace(dur * 0.15, dur * 0.85, a.frames_per_video)
        fr = [f for t in ts if (f := grab(lp, float(t))) is not None]
        if not fr:
            continue
        embs.append(enc.encode(np.stack(fr)))
        lab = 0 if r["label"]["safe"] else 1
        labels += [lab] * len(fr)
        groups += [r["id"]] * len(fr)
        n_ok += 1

    if n_ok < 20:
        print(f"本地只有 {n_ok} 个视频,先跑 scripts/fetch_safewatch.py --n-videos")
        return 2

    X = np.concatenate(embs)
    y = np.asarray(labels)
    g = np.asarray(groups)
    base = float(y.mean())
    print(f"\n{n_ok} 个视频 -> {len(y)} 帧  "
          f"(unsafe {int(y.sum())} / safe {int((y == 0).sum())}, "
          f"正例率 {base:.3f})\n")

    rows = []
    # ---- 1. 零样本:只要提示词,不要标注 ----
    su = X @ enc.encode_text(UNSAFE_PROMPTS).T
    ss = X @ enc.encode_text(SAFE_PROMPTS).T
    rows.append(report("zeroshot", y, su.max(1) - ss.max(1), base))

    # ---- 2. 原型:k 个标注正例 ----
    # ⚠️ 按**视频**切,不按帧切 —— 同一视频的帧几乎相同,按帧切会虚高
    rng = np.random.default_rng(0)
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_tr = len(uniq) // 2
    tr = np.isin(g, uniq[:n_tr])
    te = ~tr
    for k in (1, 5, 20):
        pos = X[tr & (y == 1)]
        if len(pos) < k:
            continue
        proto = pos[rng.choice(len(pos), k, replace=False)].mean(0)
        proto /= np.linalg.norm(proto)
        rows.append(report(f"proto-{k}", y[te], X[te] @ proto, base))

    # ---- 3. 线性探针:全量标注 ----
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=3000).fit(X[tr], y[tr])
        rows.append(report("linear", y[te],
                           clf.decision_function(X[te]), base))
    except ImportError:
        pass

    print("\n判读:")
    print("  AUROC>0.65 = 冻结嵌入确有信号,sentinel 层可行")
    print("  zeroshot 与 proto-k 的差 = **标注买到了多少**")
    print("  ⚠️ 冻结 SigLIP 自己给不出判决;以上三种都是在它之上加的一层")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"n_videos": n_ok, "n_frames": len(y), "pos_rate": base,
             "results": rows}, ensure_ascii=False, indent=2))
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
