"""Rolling attention probe:冻结隐状态上的轻量池化 + 线性判别。

出处:Kramár et al. 2026, *Building production-ready probes for Gemini*
(arXiv 2601.11516)。SafeLens(2605.17610)的 S1 就是它。

**在我们这里"token"是帧。** 序列是滑窗内的帧嵌入,不是文本 token。

三种池化,只差在怎么把 n 个隐状态压成一个:

    mean    z̄ = (1/n) Σ z_j                        长序列上信号被摊平
    attn    ω_j = softmax(qᵀz_j/√d),  z̄ = Σ ω_j z_j   q 失配时被"运气好的"负例抢走质量
    rolling v̄_t = Σ_{j∈W_t} α_j v_j / Σ_{j∈W_t} α_j,  窗宽 w 固定

⚠️ **Gemini 原文的输出是 max_t v̄_t。我们只在离线用它,在线绝不用** ——
见 `OnlineRollingProbe` 的 docstring,理由是单调统计量无法撤销。

⚠️ 标量值形式与嵌入池化形式在**线性头下等价**:
   v_j = wᵀz_j  =>  Σω_j v_j = wᵀ(Σω_j z_j)。
   本实现用嵌入池化形式,因为它也支持非线性头。

⚠️ **温度必须可学**,不能沿用 1/√d。教科书里的 qᵀz/√d 是为**未归一化**的
   transformer 隐状态标定的。我们喂的是 L2 归一化的 SigLIP2 嵌入(‖z‖=1),
   d=1152 时 |s_j| ≤ ‖q‖/33.9 —— 实测 logit 展布只有 0.002,
   softmax 完全退化成均匀平均(有效样本数 1500/1500,权重 max/min=1.003)。
   **注意力机制整个是死的,而系统照跑、指标还不错** —— 这类"机制写了
   但从未启动"的失败是本项目最贵的一类。故:
     - 用可学习的 log 温度替代固定 1/√d
     - `attention_diagnostics()` 直接报有效样本数,让失效可被看见

参数量:q ∈ R^d 加线性头,d=1152 时约 2.3K —— 相对编码器可忽略。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class RollingAttentionProbe:
    """离线训练 + 离线打分。一条序列一个标签。

    训练的只有 `q`(注意力 query)、`w`、`b`;骨干不参与,调用方传进来的
    嵌入就是冻结产物。这是它便宜的全部原因。
    """
    dim: int
    window: int = 10
    lr: float = 0.1
    epochs: int = 200
    l2: float = 1e-4
    # log 温度初值。exp(3) ≈ 20,配合 1/√d 把归一化嵌入的 logit 拉回可用量级。
    # 它是**可学的**,初值只影响收敛速度不影响能达到的解。
    init_log_temp: float = 3.0
    _q: np.ndarray | None = field(default=None, repr=False)
    _w: np.ndarray | None = field(default=None, repr=False)
    _b: float = 0.0
    _log_temp: float = 0.0

    # ---------- 前向 ----------

    def _pool(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """返回 (池化向量, 该窗的注意力权重, 窗起点)。

        窗宽 >= 序列长时退化为全局 attention pooling(不是报错:
        短序列是合法输入,真实视频开头就只有几帧)。

        实现说明:用滑窗视图一次算完所有窗。朴素双层循环是 O(n·w·d),
        长序列上会主导训练时间 —— 实测 200 条长 1500 的序列从 ~90s 降到 ~2s。
        """
        n = len(Z)
        w_ = min(self.window, n)
        s = self._logits(Z)
        # (n-w+1, w) 的滑窗视图,零拷贝
        sw = np.lib.stride_tricks.sliding_window_view(s, w_)
        a = np.exp(sw - sw.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)
        # v_j = w^T z_j 是标量,先投影再加权 == 先加权再投影(线性头下等价),
        # 但前者省掉 (n-w+1, d) 的中间张量。
        v = Z @ self._w
        vals = (a * np.lib.stride_tricks.sliding_window_view(v, w_)).sum(axis=1)
        t = int(np.argmax(vals))
        return a[t] @ Z[t:t + w_], a[t], t

    def _logits(self, Z: np.ndarray) -> np.ndarray:
        """s_j = exp(log_temp) · qᵀz_j / √d。温度可学,见类 docstring。"""
        return np.exp(self._log_temp) * (Z @ self._q) / np.sqrt(self.dim)

    def attention_diagnostics(self, Z: np.ndarray) -> dict:
        """注意力有没有真的在工作。

        `effective_n` = 1/Σa² 是有效样本数:等于 n 说明权重完全均匀,
        注意力退化成了平均池化 —— 此时 `degenerate` 为 True。
        """
        self._require_fit()
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        s = self._logits(Z)
        a = _softmax(s)
        eff = float(1.0 / np.sum(a ** 2))
        n = len(Z)
        return {"n": n, "effective_n": eff, "effective_frac": eff / n,
                "logit_spread": float(s.max() - s.min()),
                "weight_ratio": float(a.max() / max(a.min(), 1e-300)),
                "temperature": float(np.exp(self._log_temp)),
                # 有效样本数超过 95% 的 n = 注意力基本没做事
                "degenerate": eff / n > 0.95}

    def score_sequence(self, Z: np.ndarray) -> float:
        """离线打分 = max_t v̄_t。返回 logit(无界),不是概率。"""
        self._require_fit()
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        pooled, _, _ = self._pool(Z)
        return float(pooled @ self._w + self._b)

    # ---------- 训练 ----------

    def fit(self, sequences: list[np.ndarray], y: np.ndarray) -> "RollingAttentionProbe":
        """BCE + 类别加权。梯度只经过 argmax 那个窗(次梯度)。

        Args:
            sequences: 每条是 (n_i, d),n_i 可不同。
            y: (N,) 0/1,**每条序列一个标签** —— 这正是 SafeLens 的监督形式,
               也正是它拿不到时序定位的原因。
        """
        y = np.asarray(y, dtype=float).ravel()
        if len(sequences) != len(y):
            raise ValueError(f"序列数 {len(sequences)} != 标签数 {len(y)}")
        for i, Z in enumerate(sequences):
            if np.atleast_2d(Z).shape[1] != self.dim:
                raise ValueError(
                    f"第 {i} 条序列维度 {np.atleast_2d(Z).shape[1]} != 声明的 {self.dim}")

        rng = np.random.default_rng(0)
        self._q = rng.normal(0, 0.02, self.dim)
        self._w = np.zeros(self.dim)
        self._b = 0.0
        self._log_temp = float(self.init_log_temp)

        pos = max(y.sum(), 1.0)
        sw = np.where(y > 0, max(len(y) - pos, 1.0) / pos, 1.0)
        sw = sw / sw.mean()
        sqd = np.sqrt(self.dim)

        for _ in range(self.epochs):
            gq, gw, gb, gt = np.zeros(self.dim), np.zeros(self.dim), 0.0, 0.0
            for Z, yi, wi in zip(sequences, y, sw):
                Z = np.atleast_2d(np.asarray(Z, dtype=float))
                pooled, a, t = self._pool(Z)
                p = 1.0 / (1.0 + np.exp(-np.clip(pooled @ self._w + self._b, -30, 30)))
                g = (p - yi) * wi
                gw += g * pooled
                gb += g
                # 令 c_j = a_j·(wᵀz_j − wᵀpooled),则
                #   dL/dq        = g·τ·Σ_j c_j z_j / √d
                #   dL/dlog_temp = g·Σ_j c_j s_j
                Zw = Z[t:t + len(a)]
                coef = a * (Zw @ self._w - float(pooled @ self._w))
                tau = np.exp(self._log_temp)
                gq += g * tau * (coef @ Zw) / sqd
                gt += g * float(coef @ ((Zw @ self._q) * tau / sqd))
            n = len(sequences)
            self._w -= self.lr * (gw / n + self.l2 * self._w)
            self._b -= self.lr * (gb / n)
            self._q -= self.lr * (gq / n + self.l2 * self._q)
            # 温度不加 L2:把它拉向 0 等于把注意力拉回退化状态
            self._log_temp -= self.lr * (gt / n)
            self._log_temp = float(np.clip(self._log_temp, -10.0, 10.0))
        return self

    def _require_fit(self) -> None:
        if self._q is None or self._w is None:
            raise RuntimeError("未训练。先 fit(),或从 checkpoint 载入 q/w/b。")

    @property
    def n_params(self) -> int:
        """q + w + b + log_temp。"""
        return 2 * self.dim + 2

    def online(self) -> "OnlineRollingProbe":
        """派生一个在线版,共享已训练的参数。"""
        self._require_fit()
        return OnlineRollingProbe(self)


class OnlineRollingProbe:
    """逐帧更新的流式版本。

    ⚠️ **这是本模块唯一有争议的设计决定,写清楚理由。**

    Gemini/SafeLens 的输出是 `max_t v̄_t`。在**离线**、一条视频一个标签
    的设定下这没问题。在**流式**下它是错的:

      running max 是单调不减的。一旦某个窗越过阈值,这个统计量永远
      降不回来 —— 它无法支持 `clear`,无法闭合事件,一条 8 小时的流
      被一个瞬间尖峰点亮之后就再也熄不掉。

    这与 `sg2/anytime.py` 里"认证必须可撤销"是同一个教训:
    **在流式里,任何只增不减的量都不能直接当判决依据。**

    所以:
      `value`       —— 当前窗的 v̄_t,**会降**,这是喂给 CUSUM 的量
      `running_max` —— 单调量,仅用于与离线 SafeLens 数字对齐,**不做判决**

    窗未填满时 `warm` 为 False。此时 `value` 用已有帧算(短序列合法),
    但调用方应知道它的方差更大。
    """

    def __init__(self, probe: RollingAttentionProbe):
        self._p = probe
        self.reset()

    def reset(self) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=self._p.window)
        self._running_max = -np.inf
        self._n_seen = 0

    def update(self, emb: np.ndarray) -> float:
        """吃一帧嵌入,返回当前窗的 v̄_t(logit,会升也会降)。"""
        e = np.asarray(emb, dtype=float).ravel()
        if e.shape[0] != self._p.dim:
            raise ValueError(f"嵌入维度 {e.shape[0]} != 探针的 {self._p.dim}")
        self._buf.append(e)
        self._n_seen += 1
        Z = np.stack(self._buf)
        pooled = _softmax(self._p._logits(Z)) @ Z
        v = float(pooled @ self._p._w + self._p._b)
        self._running_max = max(self._running_max, v)
        return v

    @property
    def warm(self) -> bool:
        """窗是否已填满。"""
        return len(self._buf) >= self._p.window

    @property
    def running_max(self) -> float:
        """⚠️ 单调量,**不要用它做判决**。仅供与离线数字对齐。"""
        if self._n_seen == 0:
            raise RuntimeError("还没喂过帧")
        return self._running_max

    @property
    def stats(self) -> dict:
        return {"seen": self._n_seen, "warm": self.warm,
                "window": self._p.window,
                "running_max": None if self._n_seen == 0 else self._running_max}


# ---------- 对照用的两个基线池化 ----------

def mean_pool_score(Z: np.ndarray, w: np.ndarray, b: float = 0.0) -> float:
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    return float(Z.mean(axis=0) @ w + b)


def rolling_mean_score(Z: np.ndarray, w: np.ndarray, b: float = 0.0,
                       window: int = 10) -> float:
    """**开窗取 max,但不要注意力**:max_t mean_{j∈W_t} wᵀz_j。

    这是拆解 rolling attention probe 的关键对照。rolling 相对全局池化的
    增益可能全部来自"开窗 + 取 max",与注意力无关 —— 若本函数与
    `RollingAttentionProbe.score_sequence` 打平,注意力就是装饰,
    应当去掉(省 d 个参数,且少一个会静默失效的部件)。
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    v = Z @ w
    w_ = min(window, len(v))
    sw = np.lib.stride_tricks.sliding_window_view(v, w_)
    return float(sw.mean(axis=1).max() + b)


def global_attn_score(Z: np.ndarray, q: np.ndarray, w: np.ndarray,
                      b: float = 0.0, temperature: float = 1.0) -> float:
    """无窗版:softmax 覆盖整条序列。

    ⚠️ `temperature` 不是可选装饰:归一化嵌入上不给温度,softmax 会退化成
    均匀平均,于是这个函数变成 `mean_pool_score` 的别名 —— 拿它当对照
    等于没有对照。对比时务必传入与 rolling 探针相同的温度。
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    a = _softmax(temperature * (Z @ q) / np.sqrt(Z.shape[1]))
    return float((a @ Z) @ w + b)
