#!/usr/bin/env python3
"""成本交叉点(docs/06 实验 1)——基于**实测吞吐**而非延迟。

⚠️ 本文件是对第一版的重写。第一版有三处结构性错误:

1. **把延迟当成本。** 成本由吞吐决定(一张卡服务多少流),吞吐受益于
   batching。实测 SigLIP2 从 batch=1 的 4.4 帧/秒到 batch=32 的 46 帧/秒,
   **相差 10 倍** —— 用 batch=1 的数字建模等于把 sentinel 成本高估十倍。
2. **假设每次调用 1 帧。** 真实基线把帧打包成窗口。224px 下 16 帧窗口
   是 180 ms/帧,单帧调用是 403 ms/帧,**差 2.2 倍**。
3. **用 224x224 测。** SafeWatch 实测中位 **720x480**,6.9 倍像素。
   真实分辨率下 11GB 卡**最多只能放 2 帧窗口**(4 帧 OOM),
   而 2 帧窗口是 763 ms/帧 —— 比 224px 的 16 帧窗口贵 4 倍。

所有数字标了来源:MEASURED_* 是本机实测,其余是外部输入必须显式给。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- 本机实测(scripts/throughput_bench.py, 4x RTX 2080 Ti) ----
MEASURED_SENTINEL_S_PER_FRAME = {         # SigLIP2 冻结,batch=32
    "224": 0.0203, "real": 0.0218,
}
MEASURED_MIDTIER_S_PER_FRAME = {          # Qwen3-VL-4B,(分辨率, 窗口帧)
    ("224", 1): 0.4027, ("224", 4): 0.4822,
    ("224", 8): 0.2672, ("224", 16): 0.1798,
    ("real", 1): 0.4777, ("real", 2): 0.7632,
}
MAX_WINDOW_11GB = {"224": 16, "real": 2}  # 再大就 OOM —— 硬约束

# SafeWatch 实测
REAL_MEDIAN_RES = (720, 480)
REAL_MEDIAN_DUR_S = 6.1


@dataclass
class CostModel:
    """每 stream-hour 的美元成本。

    所有时间单位为**GPU 秒**,再按 gpu_hourly 换算成钱。
    """
    res: str = "real"
    sentinel_fps: float = 0.5
    uniform_fps: float = 2.0
    uniform_window: int = 0               # 0 = 用该分辨率的最大可行窗口
    escalate_window: int = 0
    gpu_hourly: float = 0.35              # 外部输入:自有卡摊销
    frontier_usd_per_call: float = 0.02   # 外部输入:随厂商变动
    frontier_share: float = 0.3
    audit_usd_per_clip: float = 0.50      # 外部输入:人工单价
    audit_clips_per_hour: float = 0.5     # 单位是**片段**不是 tick

    def __post_init__(self):
        cap = MAX_WINDOW_11GB[self.res]
        self.uniform_window = self.uniform_window or cap
        self.escalate_window = self.escalate_window or cap
        for w, name in ((self.uniform_window, "uniform"),
                        (self.escalate_window, "escalate")):
            if w > cap:
                raise ValueError(
                    f"{name}_window={w} 超过 {self.res} 分辨率在 11GB 卡上的"
                    f"上限 {cap} —— 实测再大就 OOM")

    # ---- 单帧 GPU 秒 ----

    def _midtier_s(self, window: int) -> float:
        key = (self.res, window)
        if key in MEASURED_MIDTIER_S_PER_FRAME:
            return MEASURED_MIDTIER_S_PER_FRAME[key]
        # 未实测的窗口用最接近的实测值,**并标明是外推**
        cands = [(abs(w - window), v)
                 for (r, w), v in MEASURED_MIDTIER_S_PER_FRAME.items()
                 if r == self.res]
        return min(cands)[1]

    def _sentinel_s(self) -> float:
        return MEASURED_SENTINEL_S_PER_FRAME[self.res]

    # ---- 每 stream-hour 成本 ----

    def uniform_cost(self) -> float:
        """基线:固定 fps 全量跑中间层(1.0 的做法)。"""
        n = self.uniform_fps * 3600
        return n * self._midtier_s(self.uniform_window) / 3600 * self.gpu_hourly

    def sentinel_cost(self) -> float:
        n = self.sentinel_fps * 3600
        return n * self._sentinel_s() / 3600 * self.gpu_hourly

    def audit_cost(self) -> float:
        return self.audit_clips_per_hour * self.audit_usd_per_clip

    def per_escalation_cost(self) -> float:
        """一次升级 = 一个窗口的中间层 + 按比例的 frontier。"""
        gpu = (self.escalate_window * self._midtier_s(self.escalate_window)
               / 3600 * self.gpu_hourly)
        return gpu + self.frontier_share * self.frontier_usd_per_call

    def adaptive_cost(self, r: float) -> float:
        n = self.sentinel_fps * 3600
        return (self.sentinel_cost() + r * n * self.per_escalation_cost()
                + self.audit_cost())

    def crossover_r(self) -> float:
        n = self.sentinel_fps * 3600
        budget = self.uniform_cost() - self.sentinel_cost() - self.audit_cost()
        per = n * self.per_escalation_cost()
        return budget / per if per > 0 else float("-inf")

    # ---- 覆盖率 ----

    @staticmethod
    def coverage(needle_s: float, fps: float) -> float:
        """均匀采样下长度 needle_s 的区间被采到的概率上界。"""
        return min(1.0, needle_s * fps)


def report(m: CostModel, label: str) -> None:
    print(f"\n【{label}】res={m.res} sentinel={m.sentinel_fps}fps "
          f"uniform={m.uniform_fps}fps×{m.uniform_window}帧窗口")
    print(f"  单帧 GPU 秒:sentinel {m._sentinel_s()*1000:.1f}ms  "
          f"中间层 {m._midtier_s(m.uniform_window)*1000:.1f}ms")
    print(f"  均匀基线      ${m.uniform_cost():8.4f} /stream-hour")
    print(f"  sentinel 固定  ${m.sentinel_cost():8.4f}")
    print(f"  审计固定       ${m.audit_cost():8.4f}")
    print(f"  每次升级       ${m.per_escalation_cost():8.4f}")
    r = m.crossover_r()
    if r <= 0:
        print(f"  ✗ r* = {r:.3f} ≤ 0:任何升级率都赢不了")
        return
    print(f"  ★ r* = {r:.2%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier-usd", type=float, default=0.02)
    ap.add_argument("--gpu-hourly", type=float, default=0.35)
    ap.add_argument("--audit-clips", type=float, default=0.5)
    a = ap.parse_args()
    kw = dict(frontier_usd_per_call=a.frontier_usd, gpu_hourly=a.gpu_hourly,
              audit_clips_per_hour=a.audit_clips)

    print("=" * 66)
    print("成本交叉点 —— 基于实测吞吐")
    print(f"外部输入:GPU ${a.gpu_hourly}/h  frontier ${a.frontier_usd}/次  "
          f"审计 {a.audit_clips} 片段/h")
    print("=" * 66)

    for res in ("224", "real"):
        report(CostModel(res=res, **kw), f"{res} 分辨率")

    print("\n" + "=" * 66)
    print("对采样率的敏感度(real 分辨率)")
    print(f"  {'sentinel':>9} {'均匀':>9} {'sentinel成本':>12} {'r*':>9}  "
          f"1s needle 覆盖")
    for sf in (0.2, 0.5, 1.0, 2.0):
        m = CostModel(res="real", sentinel_fps=sf, **kw)
        cov = CostModel.coverage(1.0, sf)
        r = m.crossover_r()
        print(f"  {sf:>7.1f}fps {m.uniform_fps:>7.1f}fps "
              f"${m.sentinel_cost():>11.4f} {r:>8.2%}  {cov:>6.0%}")

    print("\n对审计强度的敏感度(real 分辨率, sentinel 0.5fps)")
    for ac in (0.0, 0.25, 0.5, 1.0, 2.0):
        m = CostModel(res="real", audit_clips_per_hour=ac,
                      frontier_usd_per_call=a.frontier_usd,
                      gpu_hourly=a.gpu_hourly)
        r = m.crossover_r()
        mark = "" if r > 0 else "  ← 前提不成立"
        print(f"  {ac:>4.2f} 片段/h  ${m.audit_cost():5.2f}  "
              f"r*={r:>8.2%}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
