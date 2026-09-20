# 15 · 相关工作与定位（2026-09 核实）

⚠️ **结论先说:级联架构本身已经不新了。** 2026 年 5 月的 SafeLens 做了
几乎相同的事,而且用的是同一套数据和同一个基座。必须调整定位。

## 1. 最直接的竞争:SafeLens（arXiv 2605.17610）

| 维度 | SafeLens | 本提案 |
| --- | --- | --- |
| 架构 | fast-slow 两级级联 | 三级级联 |
| 升级判据 | 探针置信度 < τ（默认 0.9） | VOI / 不确定性 |
| 数据 | **SafeWatch 子集（48K）** | **同一数据集** |
| 基座 | **Qwen3-VL-2B** | **同一模型** |
| 快层 | **微调过** | 冻结 + 轻量判别层 |
| 粒度 | **整条视频分类**，≤1fps | **流式，逐 tick** |
| 保证 | **无** | conformal + ACI |
| 成绩 | 76.7% acc / 75.3% Macro F1，打赢 SafeWatch-8B、GPT-5.4、Gemini-3.1-pro | — |

**重合的部分**:级联、置信度路由、SafeWatch 数据、Qwen3-VL 基座。
这些不能再当卖点。

## 2. 其余相关工作

| 工作 | 与本提案的关系 |
| --- | --- |
| [Is Escalation Worth It?（2605.06350）](https://arxiv.org/pdf/2605.06350) | **VOI 升级判据的决策论刻画**，还证了级联何时有用/无用。我的能力路由是它的一个实例 |
| Bachar et al. 2026 | 从 logprob/熵/verbalized confidence 学升级元模型，用于人机协同审核 |
| [Quickest Detection of Hallucination Onset（2606.12476）](https://arxiv.org/html/2606.12476v1) | **QCD + 学习型 CUSUM + 延迟界**用于 LLM 安全。§3 的理论框架已被用在另一模态上 |
| [FreoStream（2606.13737）](https://arxiv.org/abs/2606.13737) | 流式 guardrail，future-aware reasoning |
| [Kelp（2510.09694）](https://arxiv.org/pdf/2510.09694) | 流式安全护栏，latent dynamics 驱动 |
| [Safety-Flag（2609.19072）](https://arxiv.org/html/2609.19072) | guard model 的可靠性与校准基准 |
| [When Can CRC Certify LLM Outputs?（2606.29054）](https://arxiv.org/pdf/2606.29054) | ⚠️ **不可能性结果**。PDF 未能解析，**必读**——可能直接削弱 conformal 那根柱子 |

## 3. 还没被做的

SafeLens 明确**没有**做的四件事，正好是提案里最有分量的部分:

### 3.1 流式与时序定位（最大空档）

SafeLens 是**整条视频分类**，≤1fps，无增量判决、无 ν、无检测延迟。
`E[(τ−ν)⁺]` 这套形式化它完全没碰。

⚠️ 但 QCD 已被用在 hallucination onset 上（2606.12476），
所以"把 QCD 用于 AI 安全"本身不新，新的只能是**视频流 + 受控观测预算**
这个组合，以及对抗自适应对手的随机化调度。

### 3.2 校准与保证 —— 而且这是 SafeLens 自己点名的局限

> SafeLens 的局限（其附录 A）：**"cascading threshold requires
> validation-set tuning across different deployment contexts"**

**这正是 conformal 校准要解决的问题。** 他们手调阈值，我们给
distribution-free 的下界 + ACI 跟踪漂移。这是最干净的差异化切入点。

⚠️ 前提是 2606.29054 的不可能性结果不会把这条堵死。**先读那篇。**

### 3.3 政策归纳与 out-of-policy

没有查到有人做"从标签反推政策定义"或"内容有害但无条款覆盖"的第四动作。
这块看起来是空的。

### 3.4 系统约束

实时截止期、跨流 batching、成本交叉点 —— 没查到有人报这些数。
但单独成篇分量不够，更适合作为支撑材料。

## 4. 定位建议

**不能再说的**：
- "第一个级联式视频 guardrail"
- "廉价层 + 升级" 本身

**可以说的**：
1. **流式**：SafeLens 是整条视频分类，我们做逐 tick 与检测延迟
2. **冻结基座**：SafeLens 微调 2B，我们不动基座 → 换代免费变强可测
3. **有保证**：SafeLens 手调阈值并自陈为局限，我们给校准下界
4. **政策可变**：归纳、out-of-policy、零日 —— 目前无人涉足

**最诚实的一句话定位**：

> 在 SafeLens 已证明"级联对视频审核有效"之后，本工作问的是：
> 当基座冻结、政策会变、且必须逐帧实时判定时，**什么还能被保证**。

## 5. 必做的事

- [ ] **读 2606.29054**（conformal 不可能性）——可能影响理论支柱
- [ ] 读 SafeLens 全文，复现其数字作为基线
- [ ] 读 2605.06350，把能力路由明确定位为其实例并引用
- [ ] 读 2606.12476，厘清 QCD 部分的增量到底是什么
- [ ] SafeLens 用的是 SafeWatch 48K 子集；确认我们的评测切分与其可比
