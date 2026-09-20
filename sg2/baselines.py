"""已发表基线的**自报数字**,以及从中能推出而原文没推的量。

为什么这是一个模块而不是文档里的一张表:这些数会进论文的对比栏,
散落在 markdown 里就没人能验证、改一处漏一处。放这里可以被测试钉住。

⚠️ 纪律:本模块只放**原文报告的**数字,每个都带出处。
推论一律是 property 或函数,并在 docstring 里写清推论依赖的假设 ——
假设不成立时**抛异常,不返回近似值**。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafeLensReported:
    """SafeLens, arXiv 2605.17610 (2026-05)。

    架构:两级级联,**两级用同一个微调后的 Qwen3-VL-2B**(Algorithm 2)。
      S1 = Rolling Attention Probe,读 VLM **最终层**隐状态
           P_φ: R^(n×d) -> Δ^p;置信度 = max 概率,≥ τ 直接出判决。
           ⚠️ **S1 不含 caption** —— caption 只在 S2 分支生成(Alg.2 第 10 行)。
           这就是 0.04s 只是一次探针前向的原因。
      S2 = T 次 Florence-2 caption + prompt 拼成 X̃ = [X; c; q]
           + **第二次全视频前向**做结构化 CoT。开销随视频长度增长。
    训练:三步串行 —— TracIn 筛数据 -> 微调 VLM -> **冻结骨干单独训探针**。
         探针约 1.6 万参数(占 2B 的 0.0008%),一次微调同时产出两级。
         ⚠️ 原文**未公布任何训练超参**(优化器/LR/epoch/batch/LoRA)。
    数据:SafeWatch 2M 经 TracIn 影响函数筛到 48K(2.4%),
         CoT 轨迹由 Qwen3.5-27B 生成。消融里**数据筛选贡献 +3.8%**,
         大于任何架构选择 —— 记下来,因为它卖的是架构。
    """

    # --- 延迟(原文 runtime 表) ---
    s1_latency_s: float = 0.04
    s2_latency_s: float = 5.02
    overall_latency_s: float = 1.76

    # --- 精度(SafeWatch-GenAI test) ---
    accuracy: float = 0.767
    macro_f1: float = 0.753

    # --- 配置 ---
    tau_default: float = 0.9
    sample_fps_max: float = 1.0  # 原文只说 ≤1fps,未给确切帧数
    finetuned: bool = True
    training_examples: int = 48_000

    # 逐类准确率:Abuse / Violence 明显偏低,恰是最依赖时序上下文的两类
    per_category_acc: tuple[tuple[str, float], ...] = (
        ("C1_sexual", 0.923), ("C2_abuse", 0.623), ("C3_violence", 0.657),
        ("C4_misinformation", 0.764), ("C5_illegal", 0.705),
        ("C6_extremism", 0.898), ("safe", 0.795),
    )

    @property
    def implied_escalation_rate(self) -> float:
        """overall = s1 + r·s2  =>  r = (overall − s1)/s2 ≈ 34.3%。

        **原文没有报告升级率。** 这是从它自己的三个延迟反推的。

        依赖的假设:级联是**串行**的 —— S1 对每条样本都跑,S2 只对升级的跑。
        原文架构图正是如此(探针读隐层 → 不确定才生成)。
        并行方案会让 overall ≥ s2,而 1.76 < 5.02 已经排除了并行。

        Raises:
            ValueError: 反推结果落在 [0,1] 之外,说明串行假设不成立或
                报告的数字不自洽。**不返回近似值** —— 静默返回 r=1.3
                会一路流进论文表格。
        """
        r = (self.overall_latency_s - self.s1_latency_s) / self.s2_latency_s
        if not 0.0 <= r <= 1.0:
            raise ValueError(
                f"反推升级率 {r:.3f} 不在 [0,1]:串行级联假设不成立,"
                f"或 s1={self.s1_latency_s}/s2={self.s2_latency_s}/"
                f"overall={self.overall_latency_s} 三者不自洽")
        return r

    @property
    def speedup_vs_all_slow(self) -> float:
        """相对「全部走 S2」的加速。**这才是原文的基线** —— 2.85x。

        我们的基线不同(统一用中间层),所以同一个升级率在两边结论相反,
        这不是谁算错了。
        """
        return self.s2_latency_s / self.overall_latency_s

    @property
    def base_risk(self) -> float:
        """µ = 1 − accuracy,作为 Kotte 下界输入的**粗估**。

        ⚠️ 这是保守方向:准确率含全部多分类错误,而我们关心的漏报只是
        其中一部分,所以真实 µ ≤ 0.233、真实下界 ≤ 这里算出的。
        结论「34.3% 在下界之上」与这个方向同向,不受影响。
        """
        return 1.0 - self.accuracy

    def deadline_feasible(self, sentinel_fps: float) -> bool:
        """整条 SafeLens 能否在 1/fps 的截止期内跑完。"""
        return self.overall_latency_s <= 1.0 / sentinel_fps

    def slow_tier_feasible(self, sentinel_fps: float) -> bool:
        """S2 单独能否在截止期内跑完 —— 即使升级率降到 0 也绕不过它。"""
        return self.s2_latency_s <= 1.0 / sentinel_fps


SAFELENS = SafeLensReported()
