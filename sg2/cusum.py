"""CUSUM 升级控制器。

全局分配用**统计**不用学习 —— 这不是保守,而是 §6 两条定理的前提:
学出来的分配策略让延迟界和优雅退化同时失效(docs/07_STREAMING_LLM.md §0)。

    S_t = max(0, S_{t-1} + log[p1(s_t)/p0(s_t)])
    tau = inf{ t : S_t >= h }

⚠️ 采样时刻带抖动。确定性调度对**按采样表定时投放 needle** 的对手
没有任何保证 —— 这正是 §3 理论要补的那块。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import CusumConfig


@dataclass
class CusumState:
    S: float = 0.0
    t_last_s: float = 0.0
    n_steps: int = 0
    n_alarms: int = 0
    alarm_t_s: float | None = None


@dataclass
class CusumController:
    """在 sentinel 的连续分数上做变点检测。

    `p0`/`p1` 用高斯近似,均值由 `mu0`/`mu1` 给出,共享方差 `sigma`。
    真实部署应在 `pool="compile"` 上拟合这三个量。
    """
    cfg: CusumConfig = field(default_factory=CusumConfig)
    mu0: float = 0.0
    mu1: float = 1.0
    sigma: float = 1.0
    seed: int = 0

    _st: CusumState = field(default_factory=CusumState, repr=False)
    _rng: random.Random = field(default=None, repr=False)  # type: ignore

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma 必须为正")
        if self.mu1 <= self.mu0:
            raise ValueError(
                f"mu1({self.mu1}) <= mu0({self.mu0}):不安全分布的均值必须更高,"
                "否则 CUSUM 会朝反方向累积")
        self._rng = random.Random(self.seed)

    # ---------- 核心 ----------

    def llr(self, s: float) -> float:
        """对数似然比。高斯同方差下是 s 的线性函数。"""
        return ((self.mu1 - self.mu0) * s
                - 0.5 * (self.mu1 ** 2 - self.mu0 ** 2)) / (self.sigma ** 2)

    def update(self, s: float, t_s: float) -> bool:
        """喂一个分数。返回是否告警。"""
        self._st.S = max(0.0, self._st.S + self.llr(s) - self.cfg.drift)
        self._st.t_last_s = t_s
        self._st.n_steps += 1
        if self._st.S >= self.cfg.h:
            self._st.n_alarms += 1
            self._st.alarm_t_s = t_s
            return True
        return False

    def reset(self) -> None:
        """告警处理完毕后复位统计量,保留计数。"""
        self._st.S = 0.0
        self._st.alarm_t_s = None

    # ---------- 调度 ----------

    def next_sample_time(self, t_s: float, *, base_fps: float,
                         alarmed: bool = False) -> float:
        """下一次采样时刻,带抖动。

        抖动是**对抗性**需求而非工程细节:确定性调度下,知道采样表的对手
        可以把 needle 精确卡在两次采样之间。随机化才有 minimax 保证。
        """
        fps = self.cfg.burst_fps if alarmed else base_fps
        step = 1.0 / max(fps, 1e-6)
        jitter = self._rng.uniform(-self.cfg.jitter_s, self.cfg.jitter_s)
        return t_s + max(1e-3, step + jitter)

    # ---------- 分析 ----------

    def expected_delay_windows(self) -> float:
        """Lorden/Lai 渐近延迟估计: E[tau-nu] ≈ h / KL(p1||p0)。

        ⚠️ 这是**渐近**结果,前提是观测独立同分布且 p0/p1 已知。视频分数
        两条都不满足,所以它只能当量级参考,不能当保证。真实延迟看实测。
        """
        kl = (self.mu1 - self.mu0) ** 2 / (2 * self.sigma ** 2)
        if kl <= 0:
            return math.inf
        return self.cfg.h / kl

    @property
    def state(self) -> CusumState:
        return self._st

    @property
    def statistic(self) -> float:
        return self._st.S
