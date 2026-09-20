"""Ring buffer:保留最近 N 秒原生帧率的帧,供告警后回溯取证。

它把"晚报"转成**有界延迟**:sentinel 晚报时仍能回到 ν 附近重查。

⚠️ 但它**只能把晚报转成延迟,不能把"永不报"转成延迟**。sentinel 零信号
的内容滚过去就是硬 miss —— 这正是保底随机覆盖 ρ 存在的理由
(docs/05_RUNTIME.md §2)。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Frame:
    t_s: float
    data: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class RingBuffer:
    """按**时长**和**字节数**双重约束的环形缓冲。

    只按帧数限制是不够的:分辨率一变,同样的帧数占的内存差几十倍。
    """
    window_s: float = 300.0
    max_bytes: int = 2 * 1024 ** 3        # 2 GiB

    _frames: deque[Frame] = field(default_factory=deque, repr=False)
    _bytes: int = 0
    _n_dropped_time: int = 0
    _n_dropped_bytes: int = 0
    _n_appended: int = 0

    def append(self, t_s: float, data: np.ndarray) -> list[Frame]:
        """追加一帧,返回被逐出的帧。

        ⚠️ 时间戳必须**单调不减**。乱序会让 window 判据失效,而表现只是
        "缓冲区里帧数不对",很难追。
        """
        if self._frames and t_s < self._frames[-1].t_s:
            raise ValueError(
                f"时间戳回退: {t_s} < {self._frames[-1].t_s}。"
                "ring buffer 要求单调不减 —— 乱序会让 window 判据失效")
        f = Frame(t_s, data)
        self._frames.append(f)
        self._bytes += f.nbytes
        self._n_appended += 1

        out: list[Frame] = []
        while self._frames and t_s - self._frames[0].t_s > self.window_s:
            out.append(self._pop())
            self._n_dropped_time += 1
        while self._frames and self._bytes > self.max_bytes:
            out.append(self._pop())
            self._n_dropped_bytes += 1
        return out

    def _pop(self) -> Frame:
        f = self._frames.popleft()
        self._bytes -= f.nbytes
        return f

    # ---------- 回溯 ----------

    def rewind(self, t0_s: float, t1_s: float) -> list[Frame]:
        """取 [t0, t1] 区间内的帧。告警后回溯取证用。"""
        if t1_s < t0_s:
            raise ValueError(f"区间反了: [{t0_s}, {t1_s}]")
        return [f for f in self._frames if t0_s <= f.t_s <= t1_s]

    def rewind_from(self, t_s: float, back_s: float) -> list[Frame]:
        return self.rewind(t_s - back_s, t_s)

    def resample(self, t0_s: float, t1_s: float, fps: float) -> list[Frame]:
        """在区间内按目标帧率重采样。告警后的突发采样用。

        取每个目标时刻**最近**的一帧,而非跳帧 —— 原生帧率与目标帧率通常
        不成整数倍。
        """
        if fps <= 0:
            raise ValueError("fps 必须为正")
        win = self.rewind(t0_s, t1_s)
        if not win:
            return []
        out, step = [], 1.0 / fps
        t = t0_s
        while t <= t1_s + 1e-9:
            nearest = min(win, key=lambda f: abs(f.t_s - t))
            if not out or out[-1] is not nearest:
                out.append(nearest)
            t += step
        return out

    # ---------- 覆盖判断 ----------

    def covers(self, t_s: float) -> bool:
        """某时刻是否还在缓冲里。**回溯前必须先问这个** —— 滚出去的
        内容取不回来,那是硬 miss 不是延迟。"""
        return bool(self._frames) and self._frames[0].t_s <= t_s <= self._frames[-1].t_s

    @property
    def span_s(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].t_s - self._frames[0].t_s

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0

    @property
    def stats(self) -> dict:
        return {"frames": len(self._frames),
                "span_s": round(self.span_s, 2),
                "mb": round(self._bytes / 1024 ** 2, 1),
                "appended": self._n_appended,
                "dropped_time": self._n_dropped_time,
                "dropped_bytes": self._n_dropped_bytes}
