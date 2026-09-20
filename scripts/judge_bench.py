#!/usr/bin/env python3
"""Judge 能力基准:中间层当政策评估器时够不够用。

`docs/10_POLICY_INDUCTION.md` 把候选政策的打分交给中间层(要跑很多次,
必须便宜)。但**judge 自身的能力上限会限制可归纳出的政策复杂度** ——
拿一个读不懂政策的模型当 judge,任何候选都得零分,整个归纳结论都是假的。

本脚本用 lavfi 造一个**真值已知**的判定任务来量这件事:

    政策: 禁止播出电视测试信号图
    正例: testsrc2 / smptebars / testsrc / rgbtestsrc / pal100bars
    负例: mandelbrot / gradients / life / cellauto / 纯色

这个任务不需要任何真实安全数据,测的是"能否按给定政策做二分类",
正是 judge 在归纳循环里要干的事。

    python scripts/judge_bench.py --models 4B 8B-nf4 --n 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import MidtierConfig
from sg2.induce import InductionMetrics
from sg2.models.streaming import Qwen3VLStreaming
from sg2.policy import PolicyClause, PolicyCorpus
from sg2.synth.ffmpeg import ffmpeg_exe

PRESETS = {
    "2B": dict(model_id="Qwen/Qwen3-VL-2B-Instruct", device="cuda:0"),
    "4B": dict(model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda:0"),
    "8B-nf4": dict(model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda:0",
                   quantization="nf4"),
}

# ⚠️ n=10 时 F1 差 0.2 完全可能是噪声。用多个随机种子/参数扩样本,
# 让每个源贡献多条而不是一条。
POSITIVE_SRC = [
    "testsrc2", "testsrc2=alpha=200",
    "smptebars", "smptehdbars",
    "testsrc", "testsrc=decimals=3",
    "rgbtestsrc", "pal100bars", "pal75bars",
    "yuvtestsrc",
]
NEGATIVE_SRC = [
    "mandelbrot", "mandelbrot=maxiter=200", "mandelbrot=start_scale=1.5",
    "gradients=n=3", "gradients=n=5:seed=7", "gradients=n=2:seed=42",
    "life=ratio=0.3", "life=ratio=0.6:seed=3",
    "cellauto", "cellauto=rule=110",
    "color=c=0x3366aa", "color=c=0xaa3366",
]

# ⚠️ 单条款语料会诱发 `uncovered`:模型看到"有内容但这条不覆盖"时,
# 没有"安全"的出口可走。加一条兜底的安全条款,让 clear 成为合法选择。
POLICY = PolicyCorpus(name="judge_bench", clauses=[
    PolicyClause(
        id="T1_testpattern", category="T1_testpattern", title="测试信号图",
        text="禁止播出电视测试信号图:彩条、色板、栅格、校准图案等"
             "用于设备校准而非真实内容的画面。"),
    PolicyClause(
        id="T0_ok", category="T0_ok", title="其他画面一律放行",
        text="除测试信号图外的任何画面（包括抽象图形、渐变、分形、纯色、"
             "自然影像）均不违规,应回 clear。"),
])


def render(src: str, out: Path, size: int = 256) -> Path:
    """用 lavfi 生成一帧。

    ⚠️ 各 lavfi 源的参数集**不一样** —— 例如 mandelbrot 没有 `duration`。
    所以只传公共参数 `size`,时长靠 `-frames:v 1` 截断。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    sep = ":" if "=" in src else "="
    spec = f"{src}{sep}size={size}x{size}"
    r = subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", spec,
         "-frames:v", "1", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"lavfi 源 {src!r} 生成失败:\n{r.stderr[-300:]}")
    return out


def load(path: Path, size: int = 256) -> np.ndarray | None:
    r = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={size}:{size}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    need = size * size * 3
    return (a[:need].reshape(size, size, 3).astype(np.float32) / 255.
            if a.size >= need else None)


def build_set(workdir: Path, n: int) -> list[tuple[str, np.ndarray, bool]]:
    """真值已知:测试信号图为正,其余为负。"""
    out = []
    for srcs, label in ((POSITIVE_SRC, True), (NEGATIVE_SRC, False)):
        for src in srcs[:n]:
            name = src.split("=")[0]
            p = render(src, workdir / f"{'pos' if label else 'neg'}_{name}.png")
            img = load(p)
            if img is not None:
                out.append((name, img, label))
    return out


def evaluate(model: str, samples, *, verbose: bool = False) -> dict:
    import torch
    cfg = MidtierConfig(name="qwen3vl", dtype="float16", attn_impl="sdpa",
                        max_length=4096, **PRESETS[model])
    m = Qwen3VLStreaming(cfg)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m._lazy()
    load_s = time.time() - t0

    from collections import Counter
    met = InductionMetrics()
    wrong, lat, n_invalid = [], [], 0
    action_kinds: Counter = Counter()
    for name, img, label in samples:
        m.reset(); m._frames.clear()
        m.set_policy(POLICY.render())
        m.ingest(img, 0.0)
        t1 = time.time()
        step = m.step()
        lat.append(time.time() - t1)

        action_kinds[step.action] += 1
        if step.action == "invalid":
            n_invalid += 1

        # ⚠️ `uncovered` 不是"判为违规"。实测 4B 对 mandelbrot 输出
        # uncovered 并写明"非电视测试信号图" —— 它**判对了**,只是因为
        # 语料里只有一条条款、没有安全出口,把"有东西但条款没覆盖"表达成了
        # uncovered。把它算成 flag 会把判对记成误报。
        pred = step.action == "flag"
        if pred and label:
            met.tp += 1
        elif pred and not label:
            met.fp += 1
            wrong.append(f"{name}(误报)")
        elif not pred and label:
            met.fn += 1
            wrong.append(f"{name}(漏报)")
        else:
            met.tn += 1
        if verbose:
            print(f"      {name:<12} 真值={'正' if label else '负'} "
                  f"动作={step.action:<9} 引用={step.policy_citation}")

    peak = sum(torch.cuda.max_memory_allocated(i) / 1024 ** 3
               for i in range(torch.cuda.device_count()))
    m.release(); del m

    # ⚠️ F1 单独看会骗人。正负各半时,一个**恒定输出**的模型也可能拿到
    # 不难看的 F1。必须同时报告:
    #   常数基线   —— 全部 flag / 全部不 flag 的 F1,本方法必须超过它
    #   动作多样性 —— 只输出一种动作 = 没在判定,只是在复读
    n_pos = met.tp + met.fn
    n_neg = met.fp + met.tn
    all_flag_f1 = (2 * n_pos / (2 * n_pos + n_neg)) if n_pos else 0.0
    return {"model": model, "f1": round(met.f1, 3),
            "all_flag_f1": round(all_flag_f1, 3),
            "beats_constant": met.f1 > all_flag_f1 + 1e-9,
            "n_actions": len(action_kinds), "actions": dict(action_kinds),
            "precision": round(met.precision, 3),
            "recall": round(met.recall, 3),
            "tp": met.tp, "fp": met.fp, "fn": met.fn, "tn": met.tn,
            "invalid": n_invalid, "load_s": round(load_s, 1),
            "lat_s": round(float(np.mean(lat)), 2),
            "peak_gb": round(peak, 2), "wrong": wrong}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4B"],
                    choices=sorted(PRESETS))
    ap.add_argument("--n", type=int, default=5, help="每类样本数")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    workdir = Path("/tmp/sg2_judge_bench")
    samples = build_set(workdir, a.n)
    n_pos = sum(1 for *_, lab in samples if lab)
    print(f"判定任务: {POLICY.enforced()[0].title}")
    print(f"样本 {len(samples)} 条（正 {n_pos} / 负 {len(samples) - n_pos}）\n")

    rows = []
    for mdl in a.models:
        print(f"  --- {mdl} ---")
        try:
            r = evaluate(mdl, samples, verbose=a.verbose)
        except Exception as e:                       # noqa: BLE001
            print(f"    失败: {type(e).__name__}: {str(e)[:100]}")
            continue
        rows.append(r)
        mark = "✓" if r["beats_constant"] else "✗ 未超过常数基线"
        print(f"    F1={r['f1']:.3f}  P={r['precision']:.3f} "
              f"R={r['recall']:.3f}  TP/FP/FN/TN={r['tp']}/{r['fp']}/"
              f"{r['fn']}/{r['tn']}  非法输出={r['invalid']}")
        print(f"    全 flag 基线 F1={r['all_flag_f1']:.3f} {mark}"
              f"   动作种类={r['n_actions']} {r['actions']}")
        if r["n_actions"] == 1:
            print("    ⚠️ 只输出一种动作 —— 没在判定,只是在复读")
        print(f"    显存{r['peak_gb']:.2f}GB 延迟{r['lat_s']:.2f}s "
              f"加载{r['load_s']:.1f}s")
        if r["wrong"]:
            print(f"    错在: {', '.join(r['wrong'])}")

    if len(rows) > 1:
        print("\n=== 对比 ===")
        print(f"  {'模型':<9} {'F1':>6} {'精确':>6} {'召回':>6} "
              f"{'动作数':>6} {'超基线':>7} {'显存GB':>8} {'延迟s':>7}")
        for r in rows:
            print(f"  {r['model']:<9} {r['f1']:>6.3f} {r['precision']:>6.3f} "
                  f"{r['recall']:>6.3f} {r['n_actions']:>6} "
                  f"{'是' if r['beats_constant'] else '否':>7} "
                  f"{r['peak_gb']:>8.2f} {r['lat_s']:>7.2f}")
        real = [r for r in rows if r["n_actions"] > 1 and r["beats_constant"]]
        if real:
            b = max(real, key=lambda r: r["f1"])
            print(f"\n  真正在判定且超过常数基线的里最高: "
                  f"{b['model']} (F1={b['f1']:.3f})")
        else:
            print("\n  ⚠️ 没有模型同时做到'动作多样'与'超过常数基线' —— "
                  "这个任务对它们要么太难要么太易,结论不可用")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
