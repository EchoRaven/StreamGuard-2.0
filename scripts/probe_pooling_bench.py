#!/usr/bin/env python3
"""三种池化在**真实嵌入几何**上的对比:mean / global-attn / rolling。

为什么必须用真实嵌入:这个项目里合成结论已经被真实数据推翻过两次
(去重退化判据、帧驱逐策略)。高斯玩具里各向同性、维度独立,
真实 SigLIP2 嵌入高度各向异性且相邻帧近乎共线 —— 注意力的行为可能完全不同。

设定(对应 Gemini probes 的部署条件:**短序列上训,长序列上用**):
  - 帧来自真实视频,用真实 SigLIP2 编码
  - "needle" = 从一个视觉上截然不同的视频里取的连续 K 帧
  - 正例 = 长良性流 + 注入 needle;负例 = 纯良性流
  - 训练序列短,评测序列长

用法:
  python scripts/probe_pooling_bench.py --videos-dir <dir> --out results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sg2.models  # noqa: E402,F401  触发 @register 注册编码器
from sg2.probe import (RollingAttentionProbe, global_attn_score,  # noqa: E402
                       mean_pool_score, rolling_mean_score)


def extract_frames(path, n, size=224):
    """按 2fps 抽最多 n 帧,返回 (k,size,size,3) float。

    走仓库已有的 `sg2.synth.ffmpeg.ffmpeg_exe()`,不另找 ffmpeg ——
    本机 PATH 里没有 ffmpeg,靠 imageio-ffmpeg 自带的二进制。
    """
    import subprocess

    from sg2.synth.ffmpeg import ffmpeg_exe
    r = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path),
         "-vf", f"fps=2,scale={size}:{size}", "-frames:v", str(n),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    per = size * size * 3
    a = np.frombuffer(r.stdout, np.uint8)
    k = a.size // per
    if k == 0:
        raise RuntimeError(f"没解出帧: {r.stderr.decode()[:200]}")
    return a[:k * per].reshape(k, size, size, 3).astype(np.float32) / 255.0


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def build_streams(bank, needle_bank, n_seq, seq_len, needle_len, rng, *,
                  blend: float = 1.0):
    """从帧嵌入库里拼流。正例注入一段 needle。

    Args:
        blend: needle 的纯度。1.0 = 直接替换(来自另一个视频,线性可分很容易);
            越小越像背景,needle 越隐蔽。⚠️ 这个旋钮很重要:blend=1 时
            所有方法在短序列上都是 1.000,那个设定只测**长度鲁棒性**,
            不测**隐蔽性** —— 在它上面得到的"注意力无用"结论不能外推。
    """
    seqs, y = [], []
    for i in range(n_seq):
        idx = rng.integers(0, len(bank), seq_len)
        Z = bank[idx].copy()
        harmful = i % 2 == 0
        if harmful:
            s = int(rng.integers(0, max(1, seq_len - needle_len)))
            nidx = int(rng.integers(0, max(1, len(needle_bank) - needle_len)))
            nd = needle_bank[nidx:nidx + needle_len]
            k = min(len(nd), needle_len)
            seg = blend * nd[:k] + (1.0 - blend) * Z[s:s + k]
            Z[s:s + k] = seg / np.maximum(
                np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
        seqs.append(Z)
        y.append(1.0 if harmful else 0.0)
    return seqs, np.array(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--n-videos", type=int, default=30)
    ap.add_argument("--frames-per-video", type=int, default=24)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--needle-len", type=int, default=5)
    ap.add_argument("--train-len", type=int, default=30)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[30, 100, 400, 1500])
    ap.add_argument("--blend", type=float, default=1.0,
                    help="needle 纯度,1=直接替换(易),越小越隐蔽")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vids = sorted(pathlib.Path(args.videos_dir).glob("*.mp4"))[:args.n_videos]
    if len(vids) < 4:
        raise SystemExit(f"至少需要 4 个视频,找到 {len(vids)}")
    print(f"[1/4] 抽帧:{len(vids)} 个视频 x {args.frames_per_video} 帧")
    per_video = []
    for v in vids:
        try:
            f = extract_frames(v, args.frames_per_video)
            if len(f):
                per_video.append(f)
        except Exception as e:      # 坏文件跳过,但要说出来
            print(f"  ⚠️ 跳过 {v.name}: {type(e).__name__}")
    if len(per_video) < 4:
        raise SystemExit("可用视频不足 4 个")

    print(f"[2/4] 真实 SigLIP2 编码 {sum(len(f) for f in per_video)} 帧")
    from sg2.config import EncoderConfig
    from sg2.registry import build
    enc = build("encoder", "siglip2", EncoderConfig(name="siglip2"))
    embs = [enc.encode(f) for f in per_video]

    # needle 来自**最后两个**视频,良性库来自其余 —— 保证 needle 源不在良性分布里
    needle_bank = np.concatenate(embs[-2:])
    bank = np.concatenate(embs[:-2])
    print(f"      良性库 {bank.shape}  needle 库 {needle_bank.shape}")
    sims = bank[1:] @ bank[:-1].T
    print(f"      真实嵌入相邻相似度: 均值 {np.mean(np.diag(sims)):.3f}")

    rng = np.random.default_rng(0)
    d = bank.shape[1]
    print(f"[3/4] 在长度 {args.train_len} 上训练探针 "
          f"(d={d}, w={args.window}, needle 纯度 {args.blend})")
    tr, ytr = build_streams(bank, needle_bank, 160, args.train_len,
                            args.needle_len, rng, blend=args.blend)
    probe = RollingAttentionProbe(dim=d, window=args.window, epochs=120).fit(tr, ytr)
    q, w, b = probe._q, probe._w, probe._b
    tau = float(np.exp(probe._log_temp))
    print(f"      探针参数量 {probe.n_params:,}  学到的温度 τ={tau:.2f}")

    # ⚠️ 注意力有没有真的在工作。固定 1/√d + 归一化嵌入会让 softmax 退化成
    # 均匀平均,此时 global-attn 只是 mean-pool 的别名 —— 对照就是假的。
    diag = probe.attention_diagnostics(build_streams(
        bank, needle_bank, 1, max(args.eval_lens), args.needle_len,
        np.random.default_rng(5), blend=args.blend)[0][0])
    print(f"      注意力诊断: 有效样本 {diag['effective_n']:.0f}/{diag['n']} "
          f"({diag['effective_frac']:.1%})  logit 展布 {diag['logit_spread']:.4f}")
    if diag["degenerate"]:
        print("      ✗ **注意力已退化成均匀平均** —— 下面的 global-attn 一列"
              "与 mean-pool 必然相同,不构成对照。结论只说明'开窗取 max'有用,"
              "**不说明注意力有用**。")

    print("[4/4] 在更长的序列上评测（同一组 q/w，唯一差别是有没有窗口）")
    rows = []
    print(f"      {'评测长度':>8} {'mean-pool':>10} {'global-attn':>12} "
          f"{'窗+max(无注意力)':>17} {'rolling':>9}  {'注意力贡献':>10}")
    for n in args.eval_lens:
        te, yte = build_streams(bank, needle_bank, 240, n, args.needle_len,
                                rng, blend=args.blend)
        a = {
            "mean": auroc(*[[mean_pool_score(Z, w, b) for Z, t in zip(te, yte) if t == k]
                            for k in (1, 0)]),
            "global": auroc(*[[global_attn_score(Z, q, w, b, temperature=tau)
                               for Z, t in zip(te, yte) if t == k] for k in (1, 0)]),
            "roll_mean": auroc(*[[rolling_mean_score(Z, w, b, args.window)
                                  for Z, t in zip(te, yte) if t == k] for k in (1, 0)]),
            "rolling": auroc(*[[probe.score_sequence(Z)
                                for Z, t in zip(te, yte) if t == k] for k in (1, 0)]),
        }
        a["identical_to_mean"] = abs(a["global"] - a["mean"]) < 1e-9
        rows.append({"eval_len": n, **a})
        print(f"      {n:>8} {a['mean']:>10.3f} {a['global']:>12.3f} "
              f"{a['roll_mean']:>17.3f} {a['rolling']:>9.3f}  "
              f"{a['rolling']-a['roll_mean']:>+10.3f}")

    long_rows = [r for r in rows if r["eval_len"] > args.train_len]
    gain = [r["rolling"] - r["global"] for r in long_rows]
    print()
    if gain and max(gain) > 0.02:
        print(f"✓ 长度失配下 rolling 胜出,最大增益 {max(gain):+.3f}")
    else:
        print(f"✗ **真实嵌入上没有复现**合成实验的结论。"
              f"长序列上 rolling−global 最大只有 {max(gain) if gain else 0:+.3f}。")
        print("  合成实验里的增益来自高斯各向同性,真实嵌入几何下不成立。")
        print("  -> 不要在论文里声称窗口带来鲁棒性,除非换个设定重新验。")

    attn_gain = [r["rolling"] - r["roll_mean"] for r in rows]
    print()
    if max(attn_gain) <= 0.01:
        print(f"✗ **注意力没有贡献**:窗内加注意力相对窗内平均最大只有 "
              f"{max(attn_gain):+.3f}。增益全部来自'开窗 + 取 max'。")
        print("  -> 该去掉注意力:省 d 个参数,且少一个会静默失效的部件。")
    else:
        print(f"✓ 注意力有贡献,最大 {max(attn_gain):+.3f}")

    if all(r["identical_to_mean"] for r in rows):
        print("⚠️ global-attn 每一行都与 mean-pool 完全相同 —— 注意力没有参与,"
              "这一列不是有效对照。")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"config": vars(args), "rows": rows, "temperature": tau,
             "attention_diagnostics": diag,
             "embedding_dim": int(d), "n_videos": len(per_video)}, indent=2))
        print(f"\n结果写入 {args.out}")
    if hasattr(enc, "release"):
        enc.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
