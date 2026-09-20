#!/usr/bin/env python3
"""把 SafeLens 的自报数字与我们的约束放在一起,看它在流式下是否可行。

数字定义在 `sg2.baselines`,本脚本只做展示。
用法: python scripts/safelens_baseline.py [--r-star 0.0253]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sg2.aci import abstention_floor  # noqa: E402
from sg2.baselines import SAFELENS as sl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r-star", type=float, default=0.0253,
                    help="我们实测的成本交叉点(真实分辨率),默认 2.53%%")
    ap.add_argument("--fps", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    r = sl.implied_escalation_rate
    print("=" * 68)
    print("SafeLens (2605.17610) 基线:从自报运行时反推升级率")
    print("=" * 68)
    print(f"  原文报告  S1={sl.s1_latency_s}s  S2={sl.s2_latency_s}s  "
          f"整体={sl.overall_latency_s}s   acc={sl.accuracy:.1%}")
    print(f"  反推      {sl.overall_latency_s} = {sl.s1_latency_s} + r x "
          f"{sl.s2_latency_s}  ->  r = {r:.1%}      [原文未给此数]")
    print(f"  其基线('全走 S2')下的加速  {sl.speedup_vs_all_slow:.2f}x")

    print("\n[1] 与 Kotte Prop 3 认证下界是否自洽")
    print(f"    µ = 1 - acc = {sl.base_risk:.3f}（保守粗估，见 docstring）")
    for a in (0.05, 0.10, 0.20):
        fl = abstention_floor(sl.base_risk, a)
        print(f"    α={a:.2f}  下界={fl:6.1%}   实际 r={r:.1%}   "
              f"{'✓ 自洽' if r >= fl else '✗ 报告数字互相矛盾'}")
    print("    -> 它的升级率不是调出来的，是被基础风险顶上去的。")

    print(f"\n[2] 与我们的成本交叉点 r* = {args.r_star:.2%} 相比")
    print(f"    r={r:.1%} vs r*={args.r_star:.2%}  ->  在我们的成本模型下"
          f"{'更便宜' if r <= args.r_star else '**比统一用中间层还贵**'}")
    print("    (基线不同：他们比的是'全走 S2'，我们比的是'统一用中间层')")

    print("\n[3] 原样搬进流式，截止期够不够")
    print(f"    {'fps':>5} {'截止期':>8} {'整体 1.76s':>12} {'S2 单独 5.02s':>14}")
    for fps in args.fps:
        ok_all = sl.deadline_feasible(fps)
        ok_s2 = sl.slow_tier_feasible(fps)
        print(f"    {fps:>5.1f} {1/fps:>7.2f}s {'✓ 可行' if ok_all else '✗ 排不开':>12}"
              f" {'✓' if ok_s2 else '✗ 超期':>14}")
    print("    -> S2 单独超过**所有**截止期。即使升级率降到 0 也绕不过。")

    print("\n[4] 升级率被三面夹，第三面是流式独有的")
    print("       (µ−α)/(M−α)  ≤  r  ≤  min( r*,  (D−T1)/T2 )")
    print("        认证下界           成本上界   实时上界")
    print("    SafeLens 的设定里 D=∞，第三项不存在 —— 它才能停在 34.3%。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
