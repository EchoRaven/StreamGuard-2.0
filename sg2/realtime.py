"""实时可行性:每一级必须在下一帧到达前出结果。

⚠️ 成本模型算的是**钱**($/stream-hour),但流式系统还有一条**独立**的
约束:**算得过来**。两者不可互相替代 ——
钱可以买更多卡,但单条流的每一级必须在**截止期内**完成,否则积压。

三级各自的截止期:
    sentinel  一帧间隔      = 1/sentinel_fps
    中间层     两次升级之间   = 1/(sentinel_fps * r)
    analyst   两次分析之间   = 1/(sentinel_fps * r * q)

⚠️ **batching 提高吞吐却伤害延迟** —— 要凑满一批就得等。
实测 SigLIP2 batch=1 是 4.4 帧/秒、batch=32 是 46 帧/秒(10 倍),
但 batch=32 意味着要攒够 32 帧。单流 0.5fps 攒 32 帧要 64 秒,
远超一帧间隔。**只能跨流攒批** —— 这正是被移出范围的多流调度,
它不是可选优化,而是吞吐的前提。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TierLoad:
    name: str
    deadline_s: float          # 两次输入之间的间隔
    service_s: float           # 处理一次要多久
    arrivals_per_s: float

    @property
    def utilization(self) -> float:
        """ρ = 到达率 × 服务时间。**必须 < 1**,否则队列无界增长。"""
        return self.arrivals_per_s * self.service_s

    @property
    def headroom(self) -> float:
        """截止期 / 服务时间。<1 = 算不过来。"""
        return self.deadline_s / self.service_s if self.service_s > 0 else math.inf

    @property
    def feasible(self) -> bool:
        return self.utilization < 1.0 and self.headroom >= 1.0

    @property
    def queue_delay_s(self) -> float:
        """M/M/1 近似的排队延迟。ρ→1 时爆炸。

        ⚠️ 这个延迟要**计入 E[τ−ν]** —— 系统算不过来造成的延迟,
        和模型看晚了造成的延迟一样是延迟。
        """
        u = self.utilization
        if u >= 1.0:
            return math.inf
        return self.service_s * u / (1.0 - u)

    def __str__(self) -> str:
        m = "✓" if self.feasible else "✗ 算不过来"
        h = "∞" if self.headroom == math.inf else f"{self.headroom:.1f}×"
        q = "∞" if self.queue_delay_s == math.inf else f"{self.queue_delay_s*1000:.0f}ms"
        return (f"{m} {self.name:<10} 截止期={self.deadline_s*1000:>8.1f}ms "
                f"服务={self.service_s*1000:>7.1f}ms 余量={h:>7} "
                f"利用率={self.utilization:>5.1%} 排队={q}")


@dataclass
class RealtimeBudget:
    """整条流水线的实时可行性。

    默认值是**本机实测**(scripts/throughput_bench.py):
      SigLIP2 real 分辨率 batch=32   21.8 ms/帧
      Qwen3-VL-4B real 2 帧窗口      763 ms/帧 → 一次调用 1526 ms
    """
    sentinel_fps: float = 0.5
    escalation_rate: float = 0.05
    frontier_share: float = 0.3

    sentinel_s_per_frame: float = 0.0218
    midtier_s_per_call: float = 1.526
    analyst_s_per_call: float = 8.0        # 多轮 agentic,外部输入

    n_streams: int = 1

    def tiers(self) -> list[TierLoad]:
        sf = self.sentinel_fps
        esc = sf * self.escalation_rate
        ana = esc * self.frontier_share
        return [
            TierLoad("sentinel", 1.0 / sf if sf else math.inf,
                     self.sentinel_s_per_frame, sf * self.n_streams),
            TierLoad("midtier", 1.0 / esc if esc else math.inf,
                     self.midtier_s_per_call, esc * self.n_streams),
            TierLoad("analyst", 1.0 / ana if ana else math.inf,
                     self.analyst_s_per_call, ana * self.n_streams),
        ]

    @property
    def feasible(self) -> bool:
        return all(t.feasible for t in self.tiers())

    def bottleneck(self) -> TierLoad:
        return max(self.tiers(), key=lambda t: t.utilization)

    def max_streams_per_gpu(self) -> int:
        """单卡能并发多少条流。**这才是部署的真问题。**

        取三级里最先饱和的那一级。
        """
        per_stream = sum(t.utilization for t in
                         RealtimeBudget(**{**self.__dict__, "n_streams": 1}
                                        ).tiers())
        return int(1.0 / per_stream) if per_stream > 0 else 0

    def total_added_delay_s(self) -> float:
        """排队带来的额外检测延迟。算不过来时是 ∞。"""
        return sum(t.queue_delay_s for t in self.tiers())


def min_batch_for_throughput(target_fps: float, per_frame_s_batched: float,
                             per_frame_s_single: float) -> int:
    """要达到目标吞吐至少需要多大的 batch。

    单流攒不满时只能**跨流**攒 —— 多流调度是吞吐的前提而非可选优化。
    """
    if per_frame_s_single <= 0 or target_fps <= 0:
        raise ValueError("参数必须为正")
    if 1.0 / per_frame_s_single >= target_fps:
        return 1
    need = target_fps * per_frame_s_batched
    return max(1, math.ceil(need))


def streams_to_fill_batch(batch: int, sentinel_fps: float,
                          window_s: float) -> int:
    """在 window_s 内凑满 batch 需要多少条并发流。"""
    per_stream = sentinel_fps * window_s
    return max(1, math.ceil(batch / per_stream)) if per_stream > 0 else 0
