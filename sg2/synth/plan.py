"""拼接规划:保证剪辑痕迹不携带标签信息。

这是整个合成流水线里唯一有实质内容的部分。判据不是"视频连贯",而是
**剪辑痕迹与标签的互信息为 0**(SPEC.md §2.5)。

三处必须与标签无关地采样,漏掉任何一处都会泄漏:
  1. 切点**数量**
  2. 切点**位置**
  3. 插入片段的**时长**  ← 最容易漏。needle 总是 1-3s 而 benign 插入总是
     10s 的话,时长本身就是标签。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

MIN_SEPARATION_S = 8.0     # 切点最小间隔,避免重叠
EDGE_MARGIN_S = 15.0       # 首尾留白,避免 needle 落在边界被裁掉


@dataclass(frozen=True)
class PlannedCut:
    t_s: float
    kind: str              # "benign" | "needle"
    duration_s: float
    insert_id: str | None = None


@dataclass(frozen=True)
class SplicePlan:
    host_id: str
    host_duration_s: float
    cuts: tuple[PlannedCut, ...]
    is_positive: bool

    @property
    def n_cuts(self) -> int:
        return len(self.cuts)

    @property
    def needle_cuts(self) -> tuple[PlannedCut, ...]:
        return tuple(c for c in self.cuts if c.kind == "needle")


def plan_cuts(host_id: str, host_duration_s: float, *, is_positive: bool,
              duration_pool: list[float], rng: random.Random,
              n_cuts_range: tuple[int, int] = (2, 6)) -> SplicePlan:
    """规划一条视频的拼接点。

    Args:
        duration_pool: 插入片段时长的经验分布。**正负例共用同一个池**,
            benign 插入的时长从 needle 的时长分布里采样。
        is_positive: 是否为 unsafe 视频。只影响**哪一个**切点是 needle,
            不影响切点的数量、位置或时长分布。

    正负例的差别只有一处:positive 把其中随机一个切点标记为 needle。
    """
    if not duration_pool:
        raise ValueError("duration_pool 不能为空 —— benign 插入需要从 needle "
                         "的时长分布采样,否则时长会泄漏标签")

    usable = host_duration_s - 2 * EDGE_MARGIN_S
    if usable <= MIN_SEPARATION_S:
        raise ValueError(f"{host_id}: 宿主时长 {host_duration_s:.1f}s 太短")

    # 1. 切点数量 —— 与标签无关
    max_by_room = max(1, int(usable // MIN_SEPARATION_S))
    lo, hi = n_cuts_range
    n = rng.randint(lo, min(hi, max_by_room))

    # 2. 切点位置 —— 与标签无关,且保证最小间隔
    positions = _sample_separated(n, EDGE_MARGIN_S,
                                  host_duration_s - EDGE_MARGIN_S,
                                  MIN_SEPARATION_S, rng)

    # 3. 时长 —— 全部从同一个池采样,与标签无关
    durations = [rng.choice(duration_pool) for _ in positions]

    # 4. 仅此一处依赖标签:随机挑一个切点作为 needle
    needle_idx = rng.randrange(n) if is_positive else -1

    cuts = tuple(
        PlannedCut(t_s=round(p, 3),
                   kind="needle" if i == needle_idx else "benign",
                   duration_s=round(d, 3),
                   insert_id=None)
        for i, (p, d) in enumerate(zip(positions, durations))
    )
    return SplicePlan(host_id=host_id, host_duration_s=host_duration_s,
                      cuts=cuts, is_positive=is_positive)


def _sample_separated(n: int, lo: float, hi: float, min_sep: float,
                      rng: random.Random) -> list[float]:
    """在 [lo, hi] 上采 n 个间隔 >= min_sep 的点。

    做法:先在压缩后的区间上均匀采样并排序,再把间隔加回去。这保证了
    在"间隔约束下"的均匀分布,而不是贪心重采样带来的有偏分布。
    """
    span = hi - lo - (n - 1) * min_sep
    if span < 0:
        raise ValueError(f"区间放不下 {n} 个间隔 {min_sep}s 的点")
    us = sorted(rng.uniform(0, span) for _ in range(n))
    return [lo + u + i * min_sep for i, u in enumerate(us)]


def duration_pool_from_events(needle_durations: list[float]) -> list[float]:
    """由 needle 片段的实际时长构造共用时长池。

    直接返回经验分布本身,不做平滑 —— benign 插入按同一分布采样即可
    使时长与标签独立。
    """
    if not needle_durations:
        raise ValueError("没有 needle 时长样本")
    return list(needle_durations)
