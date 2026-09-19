"""少样本决策边界调整 + 校准状态。

设计见 docs/03_ADAPTATION.md。这里只依赖 numpy,不需要 GPU/torch,
因为全部机制都作用在**冻结**嵌入上。

核心不变量:边界一动,阈值的 conformal 保证立即失效。本模块用类型强制它。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"      # n >= n_min,保证成立
    ESTIMATED = "estimated"        # 0 < n < n_min,只有点估计
    UNCALIBRATED = "uncalibrated"  # n=0,或边界已改未重校准


class UncalibratedClaim(RuntimeError):
    """试图声称一个没有挣到的保证。"""


def min_calibration_n(target_recall: float = 0.95, delta: float = 0.10,
                      n_miss: int = 0) -> int:
    """达到 recall >= target、置信 1-delta 所需的最小正例数。

    零失败时精确二项单边下界 p_L = delta**(1/n),解 p_L >= target。
    允许 n_miss 次失败时数值求解。
    """
    if n_miss == 0:
        return math.ceil(math.log(delta) / math.log(target_recall))
    for n in range(2, 100000):
        # P(X >= n - n_miss | p=target) <= delta 时 n 足够
        tail = sum(math.comb(n, k) * target_recall ** (n - k) *
                   (1 - target_recall) ** k for k in range(n_miss + 1))
        if tail <= delta:
            return n
    raise ValueError("未收敛")


def recall_lower_bound(n: int, n_miss: int, delta: float = 0.10) -> float:
    """Clopper-Pearson 单边下界:给定 n 个正例中漏了 n_miss 个,recall 下界。"""
    if n == 0:
        return 0.0
    if n_miss == 0:
        return delta ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):                      # 二分
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, k) * mid ** (n - k) * (1 - mid) ** k
                   for k in range(n_miss + 1))
        if tail > delta:
            hi = mid
        else:
            lo = mid
    return lo


@dataclass
class Decision:
    unsafe: bool
    score: float
    threshold: float
    calibration: CalibrationStatus
    recall_bound: float | None = None   # 仅 CALIBRATED 时非 None
    n_calib: int = 0

    def claim_recall_bound(self) -> float:
        """对外声称 recall 下界。没校准就抛,不给兜底默认值。"""
        if self.calibration is not CalibrationStatus.CALIBRATED:
            raise UncalibratedClaim(
                f"状态为 {self.calibration.value} (n={self.n_calib}),"
                f"无法声称 recall 下界。见 docs/03_ADAPTATION.md §3。")
        return self.recall_bound          # type: ignore[return-value]


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class FewShotBoundary:
    """冻结嵌入之上的少样本决策边界。

    机制 A(原型) + B(缓存检索) + C(线性头),见 docs/03_ADAPTATION.md §2。
    任何一次边界改动都会把校准状态打回 UNCALIBRATED。
    """
    dim: int
    alpha: float = 0.5        # 线性头 vs 缓存的混合
    beta: float = 5.0         # 缓存锐度
    target_recall: float = 0.95
    delta: float = 0.10

    _proto: np.ndarray | None = field(default=None, repr=False)
    _n_proto: int = 0
    _cache_x: list = field(default_factory=list, repr=False)
    _cache_y: list = field(default_factory=list, repr=False)
    _w: np.ndarray | None = field(default=None, repr=False)
    _b: float = 0.0
    _tau: float = 0.0
    _status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    _n_calib: int = 0
    _n_miss: int = 0
    _epoch: int = 0

    # ---------- 机制 A ----------
    def add_prototype(self, emb: np.ndarray) -> None:
        """增量更新类别原型。1 例即改变边界。"""
        e = _l2(np.atleast_2d(emb))
        k = e.shape[0]
        s = e.sum(axis=0)
        if self._proto is None:
            self._proto = s / k
        else:
            self._proto = (self._n_proto * self._proto + s) / (self._n_proto + k)
        self._proto = _l2(self._proto)
        self._n_proto += k
        self._invalidate()

    # ---------- 机制 B ----------
    def add_cache(self, emb: np.ndarray, label: int) -> None:
        """加入样例缓存。label ∈ {+1,-1}。"""
        if label not in (1, -1):
            raise ValueError("label 必须是 +1 或 -1")
        for e in _l2(np.atleast_2d(emb)):
            self._cache_x.append(e)
            self._cache_y.append(label)
        self._invalidate()

    # ---------- 机制 C ----------
    def fit_head(self, X: np.ndarray, y: np.ndarray) -> None:
        """L2 正则逻辑回归。无 sklearn 时退化为类均值差方向。"""
        Xn = _l2(np.atleast_2d(X))
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000)
            clf.fit(Xn, y)
            self._w, self._b = clf.coef_[0], float(clf.intercept_[0])
        except ImportError:
            pos, neg = Xn[y == 1], Xn[y != 1]
            self._w = _l2(pos.mean(0) - neg.mean(0))
            self._b = 0.0
        self._invalidate()

    def _invalidate(self) -> None:
        """边界改了 -> 旧阈值的保证失效。这是本模块的核心不变量。"""
        self._status = CalibrationStatus.UNCALIBRATED
        self._n_calib = 0
        self._n_miss = 0

    # ---------- 打分 ----------
    def score(self, emb: np.ndarray) -> float:
        e = _l2(np.atleast_1d(emb).astype(float))
        s_head = 0.0
        if self._w is not None:
            s_head = float(e @ self._w + self._b)
        elif self._proto is not None:
            s_head = float(e @ self._proto)
        s_cache = 0.0
        if self._cache_x:
            X = np.stack(self._cache_x)
            y = np.asarray(self._cache_y, dtype=float)
            sim = X @ e
            s_cache = float(np.exp(-self.beta * (1.0 - sim)) @ y)
            s_cache /= max(len(y), 1)
        if not self._cache_x:
            return s_head
        if self._w is None and self._proto is None:
            return s_cache
        return self.alpha * s_head + (1 - self.alpha) * s_cache

    # ---------- 机制 D ----------
    def calibrate(self, pos_emb: np.ndarray, neg_emb: np.ndarray | None = None
                  ) -> CalibrationStatus:
        """在校准正例上拟合 tau,使经验 recall 达标,并定状态。

        pos_emb 必须来自 pool="calibration",且未被用作 exemplar/compile。
        """
        pos = np.atleast_2d(pos_emb)
        n = pos.shape[0]
        if n == 0:
            self._status = CalibrationStatus.UNCALIBRATED
            return self._status
        scores = np.array([self.score(p) for p in pos])
        n_min = min_calibration_n(self.target_recall, self.delta)
        # 取使经验 recall >= target 的最大阈值(更严即更少假阳)
        k = max(1, math.floor(n * self.target_recall))
        self._tau = float(np.sort(scores)[n - k])
        self._n_miss = int((scores < self._tau).sum())
        self._n_calib = n
        self._status = (CalibrationStatus.CALIBRATED if n >= n_min
                        else CalibrationStatus.ESTIMATED)
        self._epoch += 1
        return self._status

    def decide(self, emb: np.ndarray) -> Decision:
        s = self.score(emb)
        bound = None
        if self._status is CalibrationStatus.CALIBRATED:
            bound = recall_lower_bound(self._n_calib, self._n_miss, self.delta)
        return Decision(unsafe=s >= self._tau, score=s, threshold=self._tau,
                        calibration=self._status, recall_bound=bound,
                        n_calib=self._n_calib)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def status(self) -> CalibrationStatus:
        return self._status
