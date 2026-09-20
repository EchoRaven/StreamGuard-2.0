#!/usr/bin/env python3
"""吞吐基准:成本模型需要的是 **GPU-秒/帧**,不是单次调用的延迟。

⚠️ 这是对先前成本模型的修正。那一版把「一次前向的延迟」当成「每帧成本」,
而且测的时候每次只送 1 帧、224x224。三处都错:

1. **延迟 ≠ 成本。** 成本由吞吐决定(一张卡能服务多少流),
   吞吐受益于 batching 与多帧窗口,延迟不反映这个。
2. **每次 1 帧不是真实用法。** 真实基线把帧打包成窗口批量送。
   7200 帧可能只是 450 次调用,先前等于把基线高估了十几倍。
3. **224x224 不是真实分辨率。** SafeWatch 实测中位 720x480,
   是它的 6.9 倍像素 —— 视觉 token 数随分辨率增长。

本脚本量三个量:
    每帧 GPU-秒(随窗口帧数、分辨率变化)
    每次调用的视觉 token 数
    单卡可并发的流数
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import EncoderConfig, MidtierConfig
from sg2.models.encoders import build_encoder
from sg2.models.streaming import Qwen3VLStreaming
from sg2.policy import safewatch_corpus

PRESETS = {
    "2B": dict(model_id="Qwen/Qwen3-VL-2B-Instruct", device="cuda:0"),
    "4B": dict(model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda:0"),
    "8B-nf4": dict(model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda:0",
                   quantization="nf4"),
}

# SafeWatch 实测:中位 720x480。224 是先前(错误的)测量分辨率。
RESOLUTIONS = {"224": (224, 224), "480p": (640, 480), "real": (720, 480)}


def _frames(n: int, wh: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = wh
    return rng.random((n, h, w, 3)).astype(np.float32)


def bench_midtier(model: str, n_frames: int, res: str, *,
                  reps: int = 3) -> dict:
    import torch
    cfg = MidtierConfig(name="qwen3vl", dtype="float16", attn_impl="sdpa",
                        max_length=8192, **PRESETS[model])
    m = Qwen3VLStreaming(cfg)
    m._lazy()
    pol = safewatch_corpus().render()
    fr = _frames(n_frames, RESOLUTIONS[res])

    # 预热
    m.reset(); m._frames.clear(); m.set_policy(pol)
    for i, f in enumerate(fr):
        m.ingest(f, float(i))
    m.step()

    lat, toks = [], []
    for _ in range(reps):
        m.reset(); m._frames.clear(); m.set_policy(pol)
        for i, f in enumerate(fr):
            m.ingest(f, float(i))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = m.step()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        toks.append(s.tokens)

    peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3
    m.release(); del m
    import gc; gc.collect(); torch.cuda.empty_cache()
    call_s = float(np.median(lat))
    return {"model": model, "res": res, "n_frames": n_frames,
            "call_s": round(call_s, 3),
            "gpu_s_per_frame": round(call_s / n_frames, 4),
            "out_tokens": int(np.median(toks)),
            "peak_gb": round(peak, 2)}


def bench_encoder(res: str, batch: int, *, reps: int = 5) -> dict:
    import torch
    enc = build_encoder(EncoderConfig(name="siglip2", dtype="float16",
                                      device="cuda:0", batch_size=batch))
    fr = _frames(batch, RESOLUTIONS[res])
    enc.encode(fr[:1])                       # 预热
    lat = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc.encode(fr)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    s = float(np.median(lat))
    return {"batch": batch, "res": res, "call_s": round(s, 4),
            "gpu_s_per_frame": round(s / batch, 5),
            "fps": round(batch / s, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4B"],
                    choices=sorted(PRESETS))
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--res", nargs="+", default=["224", "real"],
                    choices=sorted(RESOLUTIONS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    print("=== Sentinel(SigLIP2 冻结)——批量吞吐 ===")
    for res in a.res:
        for b in (1, 8, 32):
            try:
                r = bench_encoder(res, b)
            except Exception as e:                    # noqa: BLE001
                print(f"  {res} batch={b}: {type(e).__name__} {str(e)[:60]}")
                continue
            rows.append({"kind": "encoder", **r})
            print(f"  {res:<5} batch={b:<3} {r['fps']:>7.1f} frame/s  "
                  f"{r['gpu_s_per_frame']*1000:>7.2f} ms/frame")

    print("\n=== 中间层——每帧 GPU 秒随窗口帧数/分辨率变化 ===")
    print(f"  {'模型':<8} {'分辨率':<6} {'窗口帧':>6} {'调用s':>8} "
          f"{'ms/帧':>9} {'显存GB':>8}")
    for mdl in a.models:
        for res in a.res:
            for n in a.windows:
                try:
                    r = bench_midtier(mdl, n, res)
                except Exception as e:                # noqa: BLE001
                    import torch as _t, gc as _g
                    _g.collect(); _t.cuda.empty_cache()
                    # 显存不足是**真实约束**不是测量故障:real 分辨率下
                    # 多帧窗口在 11GB 卡上放不下,这直接限制了可用的窗口大小。
                    print(f"  {mdl:<8} {res:<6} {n:>6}  OOM/错误: "
                          f"{type(e).__name__}")
                    continue
                rows.append({"kind": "midtier", **r})
                print(f"  {r['model']:<8} {r['res']:<6} {r['n_frames']:>6} "
                      f"{r['call_s']:>8.3f} "
                      f"{r['gpu_s_per_frame']*1000:>9.2f} "
                      f"{r['peak_gb']:>8.2f}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
