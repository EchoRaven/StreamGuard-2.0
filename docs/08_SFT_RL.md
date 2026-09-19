# 08 · SFT 与 RL

两阶段：SFT 教会**格式与感知**，RL 教会**何时提交**。

## 1. 为什么需要 RL，SFT 不够

SFT 的目标是逐窗口的分类准确率。但真实目标是 §3 那个式子：

```latex
\min_{\pi}\; \mathbb{E}[(\tau-\nu)^{+}] \quad\text{s.t.}\quad
\mathbb{E}_{\infty}[\tau]\geq\gamma,\;\; \text{cost}(\pi)\leq B
```

**检测延迟、误报率、计算量三者的权衡是序贯决策问题，不是分类问题。**
逐窗口交叉熵无法表达"早报一点但多花 token 是否值得"。这正是 RL 的位置。

⚠️ 但 RL 只学**窗口内的提交时机**，不学全局采样调度。全局仍是 CUSUM +
随机化 —— 否则 §6 的延迟界与优雅退化两条定理同时失效。与第 6 轮 agentic
工具的边界是同一条原则：**全局用统计，局部用学习**。

## 2. SFT

### 2.1 数据构造

从 manifest 展开成**因果窗口**。窗口只含 `[t-w, t]`，绝不含未来帧。

| 窗口类型 | 取法 | 目标 action | 占比 |
|---|---|---|---|
| 正例-完整 | 完全覆盖某 event | `flag` | 25% |
| 正例-部分 | 只覆盖 event 前半 | **`hold`** | 25% |
| 负例-同视频 | 同一视频的非 event 时段 | `clear` | 35% |
| 负例-跨视频 | 纯 benign 视频 | `clear` | 15% |

**"正例-部分"是最重要的一类。** 没有它，模型只见过"证据齐全"和"完全没有"
两种情形，学不会 `hold`，在真实流上会被迫在证据不全时硬猜。

**同视频负例比跨视频负例难得多**，是有价值的难例：同样的场景、光照、
编码参数，唯一差别是有没有 needle。

### 2.2 目标格式

```json
{"action": "flag", "category": "C2_sexual",
 "evidence_frames": [63102, 63140], "policy_citation": "SW-C2.3"}
```

`clear` 与 `hold` 不需要 citation；`flag` 必须有，且 category 必须与
ground truth 一致。

### 2.3 配方

见 `04_TRAINING.md §4.2` 的 ms-swift 命令。相对那里的基础配方，SFT 阶段：

- `--freeze_vit true` —— 视觉塔是本层的能力来源，小数据上微调它会退化
- `--lora_rank 16` —— 判别任务不需要大 rank
- `--max_length 8192` —— 容纳政策头 + 16s 视觉窗
- 2 epoch，更多会过拟合到合成数据的拼接特征

## 3. RL

### 3.1 算法：GRPO

选 GRPO 而非 PPO：奖励是**可验证的**（有 ground truth ν 和类别），
不需要价值网络，显存省一半。DAPO 的稳定化技巧在规模上去后再考虑。

框架：**ms-swift 的 GRPO**（与 SFT 同栈，不换框架）；备选 EasyR1（verl 的
GRPO 精简版）。

### 3.2 奖励函数

一条流采样一整条轨迹，按轨迹给奖励：

```latex
R = \gamma_{\text{hit}}\cdot\mathbb{1}[\text{命中}]
  - \alpha\cdot\frac{(\tau-\nu)^{+}}{W}
  - \beta\cdot n_{\text{FA}}
  - \mu\cdot\frac{\text{tokens}}{T_{0}}
```

**硬门（先于一切）**：

| 门 | 条件 | 后果 |
|---|---|---|
| 格式门 | 输出不是合法 JSON / action 非法 | R = 0 |
| 引用门 | `flag` 无 `policy_citation` | R = 0 |
| 类别门 | citation 指向的类别与 ground truth 不符 | 该 flag 计为误报 |

引用门是防"无证据乱报"的关键。没有它，模型会发现**先 flag 再说**总是
划算的 —— 延迟惩罚是连续的，而误报惩罚是离散的。

### 3.3 反奖励钻空子的设计

RL 跑砸最常见的原因是奖励能被钻空子。已识别的五条路径与对策：

| 钻法 | 对策 | 测试 |
|---|---|---|
| 一开始就全 flag | `β·n_FA` 惩罚 | `test_flag_everything_scores_worse` |
| flag 但不给引用 | 引用门 R=0 | `test_flag_without_citation_is_zero` |
| 引用随便填一个条款 | 类别门 → 计为误报 | `test_wrong_citation_counts_as_fa` |
| 永远 hold（怕误报） | 漏报惩罚必须压过误报 | `test_never_flag_scores_worst` |
| 灌长推理骗 token 项 | `μ·tokens` 惩罚 | `test_token_penalty_applies` |

**第四条最危险**：若 β 设得过大，最优策略就是永不 flag，而这在训练曲线上
看起来非常稳定（奖励方差小、loss 平滑），很容易被误判为收敛良好。
因此 `sg2/train/reward.py` 里有一条不变量：

> 完全不 flag 的轨迹，奖励必须严格低于正确 flag 的轨迹。

这条在参数被改动时会被测试直接拦住。

### 3.4 超参

| 参数 | 值 | 理由 |
|---|---|---|
| `γ_hit` | 1.0 | 归一化基准 |
| `α`（延迟） | 0.5 | 延迟以窗口数归一 |
| `β`（误报） | 0.3 | 必须 < γ_hit，否则退化为永不 flag |
| `μ`（token） | 0.05 | 只做轻微整形，不该主导 |
| group size | 8 | GRPO 组内相对优势 |

β < γ_hit 不是调参偏好，是**结构性约束**，由 §3.3 第四条决定。

## 4. 评测这两个阶段

| 问题 | 指标 |
|---|---|
| SFT 学会格式了吗 | JSON 合法率、citation 存在率 |
| SFT 学会 hold 了吗 | 部分覆盖窗口上的 `hold` 比例 |
| RL 改善延迟了吗 | E[(τ−ν)⁺] vs SFT-only |
| RL 是否以误报换延迟 | 固定误报率下的延迟（不是两者各自的均值） |
| 是否钻了空子 | 五条钻法逐一检查，见 §3.3 |

⚠️ **延迟和误报必须联合报告。** 分别报均值可以让一个变差换另一个变好看起来
像全面改善。正确做法是画延迟-误报曲线，和 §7 的 Pareto 图同源。
