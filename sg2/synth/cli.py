"""批量合成 CLI。

    python -m sg2.synth --demo --out <dir> --n 20

--demo 用 lavfi 生成素材,不需要真实数据 —— 用于端到端验证流水线。
真实素材走 --hosts/--needles/--benigns 指向的清单。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import ffmpeg as F
from .build import InsertClip, build_clip
from .plan import plan_cuts

DEMO_CATS = ["C1_violence", "C2_sexual", "C3_selfharm", "C4_extremism"]


def _make_demo_assets(root: Path, rng: random.Random, n_hosts: int = 4,
                      hard: bool = True):
    """用 lavfi 造素材。

    ⚠️ 默认 hard=True:needle 与宿主/benign 使用**同一图案族**,只在色调上
    有细微差别。用截然不同的图案(smptebars vs testsrc2)会让任务退化 ——
    颜色直方图就能拿到 AUC 1.0,烟测将毫无信息量,还会让人误以为方法有效。

    即便如此,这仍只是**管路测试**,不是方法验证。方法必须在真实安全数据上测。
    """
    a = root / "assets"
    hosts = [F.make_test_video(a / f"host{i}.mp4", duration_s=90.0,
                               pattern="testsrc2", freq=200 + 10 * i,
                               size="320x240") for i in range(n_hosts)]
    durs = [1.0, 2.0, 3.0, 4.0]
    # hard: needle/benign/host 同族图案,仅频率与音高微differ
    n_pat, b_pat = ("testsrc2", "testsrc2") if hard else ("smptebars", "testsrc")
    n_freq, b_freq = (240, 260) if hard else (900, 300)
    needles, benigns = [], []
    for i, d in enumerate(durs):
        needles.append(InsertClip(
            f"n{i}", str(F.make_test_video(a / f"n{i}.mp4", duration_s=d,
                                           pattern=n_pat, freq=n_freq + i,
                                           size="320x240")),
            d, category=DEMO_CATS[i % len(DEMO_CATS)],
            pixel_area_frac=[0.002, 0.02, 0.12, 0.3][i]))
        benigns.append(InsertClip(
            f"b{i}", str(F.make_test_video(a / f"b{i}.mp4", duration_s=d,
                                           pattern=b_pat, freq=b_freq + i,
                                           size="320x240")), d))
    return hosts, needles, benigns, durs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sg2.synth")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20, help="生成条数")
    ap.add_argument("--demo", action="store_true", help="用 lavfi 造素材")
    ap.add_argument("--easy-demo", action="store_true",
                    help="demo 素材用截然不同的图案(任务退化,仅供调试)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-rate", type=float, default=0.5)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    out = args.out
    (out / "videos").mkdir(parents=True, exist_ok=True)

    if not args.demo:
        ap.error("目前仅实现 --demo;真实素材清单接入见 docs/02_DATASET.md §2")

    hosts, needles, benigns, durs = _make_demo_assets(
        out, rng, hard=not args.easy_demo)

    # 池与 split 的分配(SPEC.md §3):校准池必须与 exemplar/compile 不相交
    def assign(i: int) -> tuple[str, str]:
        r = i % 10
        if r < 5:
            return "eval", "test"
        if r < 8:
            return "calibration", "calib"
        return ("exemplar", "train") if r == 8 else ("compile", "train")

    records = []
    for i in range(args.n):
        is_pos = rng.random() < args.pos_rate
        host = hosts[i % len(hosts)]
        hinfo = F.probe(host)
        plan = plan_cuts(f"host{i % len(hosts)}", hinfo.duration_s,
                         is_positive=is_pos, duration_pool=durs, rng=rng,
                         n_cuts_range=(2, 5))
        pool, split = assign(i)
        cid = f"sg2-{i:05d}"
        rec = build_clip(plan, host, needles, benigns,
                         out / "videos" / f"{cid}.mp4", clip_id=cid, rng=rng,
                         pool=pool, split=split, workdir=out / "work")
        records.append(rec)
        print(f"  [{i+1}/{args.n}] {cid} "
              f"{'unsafe' if not rec.is_safe else 'safe  '} "
              f"cuts={rec.splice.n_cuts} pool={pool}")

    mf = out / "manifest.jsonl"
    with open(mf, "w") as fh:
        fh.write(json.dumps({"__meta__": {"spec_version": "1.0",
                                          "generator": "sg2.synth --demo",
                                          "seed": args.seed}}) + "\n")
        for r in records:
            fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    print(f"\n写入 {mf}  ({len(records)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
