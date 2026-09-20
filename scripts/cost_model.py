#!/usr/bin/env python3
"""成本交叉点(docs/06 实验 1)——三条预实验里排第一的那条。

它用**实测数字**回答一个能否定整个前提的问题:

    自适应分配要赢过均匀分配,升级率 r 最多能到多少?

若实测的 r 远超这个上界,Pareto 曲线会输在 $/stream-hour ——
**而且输给我们自己的 1.0**。这不需要任何精度实验就能算出来。

延迟数字来自本机实测(4x RTX 2080 Ti):
    2B fp16   1.23 s/tick      4.08 GB
    4B fp16   0.37 s/tick      8.50 GB
    8B nf4    0.59 s/tick      6.17 GB
API 单价是**外部输入**,随厂商变动,必须显式给而不能写死。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.eval import cost_crossover

# 本机实测(scripts/compare_backends.py)
MEASURED_LAT_S = {"2B": 1.23, "4B": 0.37, "8B-nf4": 0.59}
GPU_HOURLY_USD = 0.35          # 自有卡摊销的粗估,可覆盖


@dataclass
class CostModel:
    """每 stream-hour 的成本。

    ⚠️ 单位统一为**每流每小时美元**。混单位是这类估算最常见的错法。
    """
    stream_minutes: float = 60.0
    sentinel_fps: float = 0.5
    uniform_fps: float = 2.0
    sentinel_s_per_frame: float = 0.01      # 冻结 embedding,实测 18 帧/秒
    midtier_s_per_call: float = 0.37        # 4B 实测
    gpu_hourly: float = GPU_HOURLY_USD
    frontier_usd_per_call: float = 0.02     # 外部输入
    frontier_share: float = 0.3             # 升级中再走 frontier 的比例
    audit_usd_per_clip: float = 0.50        # 人工审计一个片段的单价
    # ⚠️ 审计的单位是**片段**不是 sentinel tick。按 tick 算会得出荒谬的
    # 数字:0.5fps 跑 1 小时有 1800 个 tick,抽 1% 就是 18 次人工、
    # $9.00/stream-hour —— 是均匀基线的十倍。1 小时视频请人看 18 次
    # 没有道理。这个单位错误会让前提看起来完全不成立。
    audit_clips_per_hour: float = 0.5

    @property
    def _hours(self) -> float:
        return self.stream_minutes / 60.0

    def uniform_cost(self, lat_s: float) -> float:
        """基线:固定 fps 全量跑中间层(1.0 的做法)。"""
        n = self.uniform_fps * self.stream_minutes * 60
        return n * lat_s / 3600.0 * self.gpu_hourly / self._hours

    def sentinel_cost(self) -> float:
        n = self.sentinel_fps * self.stream_minutes * 60
        return n * self.sentinel_s_per_frame / 3600.0 * self.gpu_hourly \
            / self._hours

    def per_escalation_cost(self) -> float:
        gpu = self.midtier_s_per_call / 3600.0 * self.gpu_hourly
        return gpu + self.frontier_share * self.frontier_usd_per_call

    def audit_cost(self) -> float:
        return self.audit_clips_per_hour * self.audit_usd_per_clip

    def adaptive_cost(self, r: float) -> float:
        n = self.sentinel_fps * self.stream_minutes * 60
        return (self.sentinel_cost() + r * n * self.per_escalation_cost()
                / self._hours + self.audit_cost())

    def crossover_r(self, lat_s: float) -> float:
        n = self.sentinel_fps * self.stream_minutes * 60
        budget = (self.uniform_cost(lat_s) - self.sentinel_cost()
                  - self.audit_cost())
        per = n * self.per_escalation_cost() / self._hours
        return budget / per if per > 0 else float("-inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier-usd", type=float, default=0.02,
                    help="frontier 单次调用美元(外部输入,随厂商变动)")
    ap.add_argument("--audit-usd", type=float, default=0.50,
                    help="人工审计**一个片段**的单价")
    ap.add_argument("--audit-clips", type=float, default=0.5,
                    help="每流每小时抽查的片段数(不是每 tick 的比例!)")
    ap.add_argument("--baseline", default="2B", choices=sorted(MEASURED_LAT_S))
    a = ap.parse_args()

    m = CostModel(frontier_usd_per_call=a.frontier_usd,
                  audit_usd_per_clip=a.audit_usd,
                  audit_clips_per_hour=a.audit_clips)
    lat = MEASURED_LAT_S[a.baseline]

    print(f"基线:均匀 {m.uniform_fps} fps 跑 {a.baseline}"
          f"(实测 {lat:.2f} s/tick)")
    print(f"自适应:sentinel {m.sentinel_fps} fps + 升级率 r 走中间层,"
          f"其中 {m.frontier_share:.0%} 再走 frontier"
          f"(${a.frontier_usd}/次)")
    print(f"审计:每流每小时 {a.audit_clips} 个片段 × ${a.audit_usd}\n")

    print(f"  均匀基线      ${m.uniform_cost(lat):8.4f} /stream-hour")
    print(f"  sentinel 固定  ${m.sentinel_cost():8.4f}")
    print(f"  审计固定       ${m.audit_cost():8.4f}"
          f"{'   ← 保证的价格' if m.audit_cost() > 0 else ''}")
    print(f"  每次升级       ${m.per_escalation_cost():8.4f}\n")

    r = m.crossover_r(lat)
    if r <= 0:
        print(f"✗ r* = {r:.3f} ≤ 0:**任何**升级率都赢不了。")
        print("  固定成本(sentinel+审计)已超过均匀基线,前提不成立。")
        return 1
    print(f"★ 交叉点 r* = {r:.4f}  ({r:.2%})")
    print(f"  即:升级率低于 {r:.2%} 时自适应更便宜,高于则更贵\n")

    print("  升级率 r 下的成本:")
    for rr in (0.01, 0.05, 0.10, 0.20, 0.40):
        c = m.adaptive_cost(rr)
        mark = "✓ 更便宜" if c < m.uniform_cost(lat) else "✗ 更贵"
        print(f"    r={rr:>5.0%}  ${c:8.4f}  {mark}")

    print("\n  对审计强度的敏感度(审计是保证的价格):")
    for ac in (0.0, 0.25, 0.5, 1.0, 2.0):
        mm = CostModel(frontier_usd_per_call=a.frontier_usd,
                       audit_usd_per_clip=a.audit_usd,
                       audit_clips_per_hour=ac)
        rr = mm.crossover_r(lat)
        mark = "" if rr > 0 else "  ← 前提不成立"
        print(f"    {ac:>4.2f} 片段/h  ${mm.audit_cost():5.2f}  "
              f"r*={rr:>8.2%}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
