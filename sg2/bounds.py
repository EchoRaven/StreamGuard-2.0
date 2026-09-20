"""风险上界的层级:Hoeffding / Bernstein / e-CRC / 精确二项。

出处:Kotte 2026 (arXiv:2606.29054) §2.1 与 Proposition 2/7。
原文证明的层级是

    Hoeffding ⊆ Bernstein ⊆ e-CRC

但只在**严格目标**(α ≤ 0.20)成立;α ≥ 0.25 时 Hoeffding–Bernstein
会在闭式频界 α*(n,δ) 处反转。原文的结论是最小充分集为
{Hoeffding, e-CRC},Bernstein 只是闭式的便利。

⚠️ **原文没有列精确二项(Clopper-Pearson)**,而那正是我们在用的
(docs/03 §2.4 的 45 个正例)。本模块把它放进同一个比较里 ——
我们的损失是 0/1(漏报与否),精确二项对伯努利是**紧的**,
在 n≈45 的稀缺区间可能优于所有基于集中不等式的界。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BoundResult:
    name: str
    ucb: float                  # 风险的上置信界
    n: int
    certifies: bool             # ucb <= alpha ?

    def __str__(self) -> str:
        m = "✓" if self.certifies else "✗"
        return f"{m} {self.name:<12} UCB={self.ucb:.4f} (n={self.n})"


def hoeffding_ucb(risks: np.ndarray, delta: float = 0.1) -> float:
    """R̂ + sqrt(log(2/δ) / (2n))。不看方差,最保守。"""
    n = len(risks)
    if n < 1:
        return 1.0
    return float(np.mean(risks) + math.sqrt(math.log(2 / delta) / (2 * n)))


def bernstein_ucb(risks: np.ndarray, delta: float = 0.1) -> float:
    """经验 Bernstein(Maurer & Pontil 2009):

        R̂ + sqrt(2σ̂² log(2/δ) / n) + 7 log(2/δ) / (3(n−1))

    σ̂² 用 1/n 约定(与原文实现一致)。
    ⚠️ 那个 **加性项 7L/(3(n−1))** 正是 α 放宽时反转的根源。
    """
    n = len(risks)
    if n < 2:
        return 1.0
    L = math.log(2 / delta)
    var = float(np.var(risks))               # 1/n 约定
    return float(np.mean(risks) + math.sqrt(2 * var * L / n)
                 + 7 * L / (3 * (n - 1)))


def ecrc_certifies(risks: np.ndarray, alpha: float, delta: float = 0.1,
                   n_orderings: int = 22, seed: int = 0) -> bool:
    """e-CRC:testing-by-betting。

        W_0 = 1,  W_j = W_{j-1} · (1 + κ_j (α − r_j))

    κ 是 Kelly 式下注,裁剪到 [0, 0.5],κ_1 = 0(首观测前不下注)。
    e 值取**多个顺序下的最小财富**;达到 1/δ 即通过。

    H0: E[R] ≥ α 下财富是非负上鞅,由 Ville 不等式给出有效性。
    """
    n = len(risks)
    if n < 2:
        return False
    rng = np.random.default_rng(seed)
    orders = [np.arange(n), np.arange(n)[::-1]]
    orders += [rng.permutation(n) for _ in range(max(0, n_orderings - 2))]

    min_w = math.inf
    for order in orders:
        w, kappa = 1.0, 0.0
        seen: list[float] = []
        for j, i in enumerate(order):
            r = float(risks[i])
            w *= (1.0 + kappa * (alpha - r))
            if w <= 0:
                w = 0.0
                break
            seen.append(r)
            # Kelly 式:用已见样本估 (α − R̄),裁剪到 [0, 0.5]
            m = float(np.mean(seen))
            denom = max(alpha * (1 - alpha), 1e-6)
            kappa = float(np.clip((alpha - m) / denom, 0.0, 0.5))
        min_w = min(min_w, w)
    return min_w >= 1.0 / delta


def exact_binomial_ucb(risks: np.ndarray, delta: float = 0.1) -> float:
    """Clopper-Pearson 上界。**对伯努利损失是紧的。**

    风险非 0/1 时退回 Hoeffding —— 精确二项只对伯努利有效,
    硬套到连续损失上会给出无效的界。
    """
    n = len(risks)
    if n < 1:
        return 1.0
    if not np.all(np.isin(risks, (0.0, 1.0))):
        return hoeffding_ucb(risks, delta)
    k = int(risks.sum())
    if k == n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):                      # 二分解 P(X<=k | p) = delta
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, j) * mid ** j * (1 - mid) ** (n - j)
                   for j in range(k + 1))
        if tail > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


def compare_bounds(risks: np.ndarray, alpha: float,
                   delta: float = 0.1) -> list[BoundResult]:
    """在同一批校准风险上比较四个界。"""
    r = np.asarray(risks, dtype=float)
    out = [
        BoundResult("Hoeffding", hoeffding_ucb(r, delta), len(r), False),
        BoundResult("Bernstein", bernstein_ucb(r, delta), len(r), False),
        BoundResult("ExactBinom", exact_binomial_ucb(r, delta), len(r), False),
    ]
    for b in out:
        b.certifies = b.ucb <= alpha
    out.append(BoundResult("e-CRC", float("nan"), len(r),
                           ecrc_certifies(r, alpha, delta)))
    return out


def min_n_to_certify(bound: str, miss_rate: float, alpha: float,
                     delta: float = 0.1, n_max: int = 5000) -> int | None:
    """某个界要认证 alpha,至少需要多少校准正例。

    用**期望**的漏报模式(尽量均匀分布)而非随机抽样,
    这样结果可复现且是该界的典型需求。
    """
    fn = {"Hoeffding": hoeffding_ucb, "Bernstein": bernstein_ucb,
          "ExactBinom": exact_binomial_ucb}
    for n in range(2, n_max):
        k = int(round(miss_rate * n))
        r = np.zeros(n)
        if k:
            r[np.linspace(0, n - 1, k).astype(int)] = 1.0
        if bound == "e-CRC":
            if ecrc_certifies(r, alpha, delta):
                return n
        elif fn[bound](r, delta) <= alpha:
            return n
    return None
