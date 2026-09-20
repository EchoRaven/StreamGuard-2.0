#!/usr/bin/env python3
"""横向对比中间层 backbone。

问的**不是**"能不能跑起来",而是三个可测量的量:

  1. 格式合法率   —— 输出能否被严格解析(docs/08 §4)
  2. 引用准确率   —— citation 能否对到真实条款,错的话错在哪一类
  3. uncovered 使用率 —— 政策有缺口时,模型是否真的会说"未覆盖"
                          (2B 实测从不使用,即使只给一条无关条款)

这三个都不需要真实安全数据就能测,因为它们测的是**协议遵守**而非判准。

    python scripts/compare_backends.py --models 2B 8B --n 8
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import MidtierConfig
from sg2.models.streaming import Qwen3VLStreaming
from sg2.policy import PolicyClause, PolicyCorpus, safewatch_corpus
from sg2.synth.ffmpeg import ffmpeg_exe

PRESETS = {
    "2B": dict(model_id="Qwen/Qwen3-VL-2B-Instruct", device="cuda:0"),
    "4B": dict(model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda:0"),
    "8B-nf4": dict(model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda:0",
                   quantization="nf4"),
    "8B-shard": dict(model_id="Qwen/Qwen3-VL-8B-Instruct", device="auto",
                     max_memory_per_gpu="9GiB"),
}

# 三种语料,测三件不同的事。
#
# ⚠️ 只用前两种时对比**不具区分力**:demo 视频是 ffmpeg 的 testsrc2 彩条
# 信号图,里面没有任何像违规的东西,所以 2B/4B/8B 全部输出 hold、
# 全部 100% 合法 —— 测出来的只有"会不会照格式输出",区分不了能力。
#
# MATCHING 写一条**确实匹配画面内容**的条款。此时正确行为是 flag 并引用
# 它,于是引用准确率才真正可测。这也正是 SafeWatch 说的 policy-following。

NARROW = PolicyCorpus(name="narrow", clauses=[
    PolicyClause(id="Z1_spam", category="Z1_spam", title="垃圾广告",
                 text="禁止重复刷屏的商业广告。")])

MATCHING = PolicyCorpus(name="matching", clauses=[
    PolicyClause(id="T1_testpattern", category="T1_testpattern",
                 title="测试信号图",
                 text="禁止播出电视测试信号图:彩条、色板、校准图案等"
                      "用于设备校准而非真实内容的画面。"),
    PolicyClause(id="Z1_spam", category="Z1_spam", title="垃圾广告",
                 text="禁止重复刷屏的商业广告。"),
    PolicyClause(id="Z2_silence", category="Z2_silence", title="静帧",
                 text="禁止长时间完全静止无变化的画面。"),
])


def grab(vid: str, t: float, size: int = 224) -> np.ndarray | None:
    r = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.2f}", "-i", vid,
         "-frames:v", "1", "-vf", f"scale={size}:{size}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    need = size * size * 3
    return (a[:need].reshape(size, size, 3).astype(np.float32) / 255.
            if a.size >= need else None)


def evaluate(name: str, frames, corpus: PolicyCorpus, label: str) -> dict:
    import torch
    cfg = MidtierConfig(name="qwen3vl", dtype="float16", attn_impl="sdpa",
                        max_length=4096, **PRESETS[name])
    m = Qwen3VLStreaming(cfg)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m._lazy()
    load_s = time.time() - t0

    actions, reasons, lat = Counter(), Counter(), []
    cited_target = 0
    for t_s, f in frames:
        m.reset(); m._frames.clear()
        m.set_policy(corpus.render())
        m.ingest(f, t_s)
        t1 = time.time()
        step = m.step()
        lat.append(time.time() - t1)
        actions[step.action] += 1
        if step.action == "flag":
            cl, why = corpus.resolve_citation(step.policy_citation)
            reasons[why] += 1
            if cl is not None and cl.id == "T1_testpattern":
                cited_target += 1

    n = sum(actions.values())
    peak = sum(torch.cuda.max_memory_allocated(i) / 1024 ** 3
               for i in range(torch.cuda.device_count()))
    m.release()          # del + empty_cache 不够,分片模型的 accelerate
    del m                # hook 会留住各卡子模块的引用
    return {"model": name, "policy": label, "n": n, "load_s": round(load_s, 1),
            "peak_gb": round(peak, 2), "lat_s": round(float(np.mean(lat)), 2),
            "valid_rate": round(1 - actions["invalid"] / max(n, 1), 3),
            "cited_target": cited_target,
            "actions": dict(actions), "citation": dict(reasons)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["2B"],
                    choices=sorted(PRESETS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    vdir = a.videos or ("/tmp/claude-1052/-data-common-haibotong/"
                        "069fb65b-e0b2-4721-a30c-e549b6f6a1b1/scratchpad/"
                        "batch/videos")
    vids = sorted(glob.glob(f"{vdir}/*.mp4"))[:a.n]
    if not vids:
        print(f"找不到视频: {vdir}")
        return 2
    frames = [(10.0, f) for v in vids if (f := grab(v, 10.0)) is not None]
    print(f"{len(frames)} 帧,来自 {len(vids)} 条视频\n")

    rows = []
    for name in a.models:
        for corpus, label in ((safewatch_corpus(), "完整六类"),
                              (NARROW, "仅无关条款"),
                              (MATCHING, "有匹配条款")):
            try:
                r = evaluate(name, frames, corpus, label)
            except Exception as e:                    # noqa: BLE001
                print(f"  {name} / {label}: 失败 — {type(e).__name__}: "
                      f"{str(e)[:90]}")
                continue
            rows.append(r)
            print(f"  {r['model']:<9} {r['policy']:<7} "
                  f"加载{r['load_s']:>5.1f}s 峰值{r['peak_gb']:>5.2f}GB "
                  f"延迟{r['lat_s']:>4.2f}s 合法率{r['valid_rate']:>5.1%} "
                  f"{r['actions']}")

    print("\n=== 有匹配条款时:是否 flag + 引用对不对（这一栏才有区分力）===")
    for r in rows:
        if r["policy"] != "有匹配条款":
            continue
        n = r["n"]
        n_flag = r["actions"].get("flag", 0)
        exact = r["citation"].get("exact", 0) + r["citation"].get("prefix", 0)
        right = r.get("cited_target", 0)
        other = {k: v for k, v in r["actions"].items() if k != "flag"}
        print(f"  {r['model']:<9} flag {n_flag}/{n} = {n_flag / max(n, 1):.0%}"
              f"   引用可解析 {exact}/{max(n_flag, 1)}"
              f"   引到目标条款 {right}/{max(n_flag, 1)}"
              f"   其余 {other}")

    print("\n=== 引用错误分布（全部 flag） ===")
    for r in rows:
        if r["citation"]:
            print(f"  {r['model']:<9} {r['policy']:<7} {r['citation']}")

    print("\n=== uncovered 使用率（政策有缺口时才该出现） ===")
    for r in rows:
        if r["policy"] == "仅无关条款":
            u = r["actions"].get("uncovered", 0)
            print(f"  {r['model']:<9} {u}/{r['n']} = {u / max(r['n'], 1):.0%}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
