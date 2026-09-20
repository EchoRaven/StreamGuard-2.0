# StreamGuard-2

**Where you look matters more than what you look with.**

A training-free, agentic streaming video guardrail. This repository holds the
dataset specification, the few-shot decision-boundary adaptation layer, and the
calibration machinery. The research argument lives in a separate proposal document.

> **Status**: pre-data. Spec and adaptation layer are implemented and tested;
> the synthesis pipeline, sentinel and mid-tier training are not yet built.

### What is here

| Component | Purpose |
|---|---|
| [`SPEC.md`](SPEC.md) | Dataset format v1.0 — four orthogonal difficulty axes, frame-level ground truth |
| [`sg2/adapt.py`](sg2/adapt.py) | Few-shot boundary adaptation (prototype / cache / linear head) + conformal calibration |
| [`sg2/schema.py`](sg2/schema.py) | Records and a loader that enforces train/calibration pool separation |
| [`sg2/validate.py`](sg2/validate.py) | Dataset-level invariants, including a permutation test for splice leakage |
| [`sg2/synth/`](sg2/synth/) | Synthesis pipeline — splice planning that provably carries no label information |
| [`sg2/audit.py`](sg2/audit.py) | Adversarial audit: attack the dataset with the features the sentinel will read |
| [`sg2/train/fusion.py`](sg2/train/fusion.py) | Sentinel fusion head — multi-channel scores into one continuous CUSUM statistic |
| [`docs/`](docs/) | Full implementation plan: model choice, data construction, training, runtime, experiments |

### Three ideas worth the click

1. **Difficulty is four independent axes, not an easy/hard split.** Temporal
   sparsity, perceptual subtlety, modality locus, context dependence. Keeping
   the factors is what lets you diagnose *which* kind of hardness breaks *which*
   component.
2. **Moving a decision boundary invalidates its calibration — immediately.**
   Adding a single exemplar resets the status to `UNCALIBRATED`. Otherwise the
   conformal layer is decoration.
3. **A guarantee you have not earned is not returned as a default.**
   `claim_recall_bound()` raises rather than degrading silently.

---

## 中文

流式视频 guardrail 的数据集规范、少样本适配层与校准机制。
研究论证见独立提案文档；本仓库讲怎么做。

从 [`docs/00_OVERVIEW.md`](docs/00_OVERVIEW.md) 看起，那里有完整决策表。

| 文档 | 内容 |
|---|---|
| [00_OVERVIEW](docs/00_OVERVIEW.md) | 决策表、里程碑、当前阻塞 |
| [01_MODELS](docs/01_MODELS.md) | 三层 base model 选型 + 显存适配 |
| [02_DATASET](docs/02_DATASET.md) | 来源、合成流水线、切分、规模、法律 |
| [03_ADAPTATION](docs/03_ADAPTATION.md) | **少样本决策边界**：四机制 + 校准协议 |
| [04_TRAINING](docs/04_TRAINING.md) | 框架、配方、超参、Turing 的坑 |
| [05_RUNTIME](docs/05_RUNTIME.md) | 推理栈、Agent 工具集、输出契约 |
| [06_EXPERIMENTS](docs/06_EXPERIMENTS.md) | 实验协议，每个带否定条件 |
| [07_STREAMING_LLM](docs/07_STREAMING_LLM.md) | **中间层流式设计**：KV 分段、三元动作 |
| [08_SFT_RL](docs/08_SFT_RL.md) | **SFT 与 RL**：数据构造、GRPO、反奖励钻空子 |
| [09_POLICY](docs/09_POLICY.md) | **政策语料与 out-of-policy**：SafeWatch 六类、补缺流程、位置偏置 |
| [10_POLICY_INDUCTION](docs/10_POLICY_INDUCTION.md) | **从标签反推政策定义**：离线编译器、迭代循环、四条对照 |
| [11_POLICY_TO_TRAINING](docs/11_POLICY_TO_TRAINING.md) | **生成的政策怎么接进训练**：反事实敏感度、六种政策增强 |
| [SPEC](SPEC.md) | 数据集格式规范 v1.0（已冻结） |

### 状态

| 组件 | 状态 |
|---|---|
| 格式规范 + JSON Schema + 池守卫 loader | ✅ |
| 数据集校验器（6 项，含置换检验） | ✅ |
| 少样本适配 + conformal 校准 | ✅ 四机制 |
| 合成流水线（规划/渲染/CLI） | ✅ 40 条端到端验证 |
| 压缩域特征 + 对抗泄漏审计 | ✅ |
| Sentinel 融合头 | ✅ numpy，无需 GPU |
| 流式 KV 分段（sink/政策/事件/视觉） | ✅ 脱离模型可测 |
| SFT 数据构造 + 配比重采样 | ✅ |
| RL 奖励 + 五条反钻空子 | ✅ |
| 配置系统（嵌套 dataclass + YAML + 硬件校验） | ✅ |
| 组件注册表（按名字换实现） | ✅ |
| **Prompt 模板系统**（四段可配 + 指纹化缓存键） | ✅ |
| **政策语料**（SafeWatch 六类 + 版本化 + 生命周期） | ✅ |
| **out-of-policy**（第四动作 + 补缺 + 人工闸门） | ✅ |
| **政策归纳**（从 k 条标签反推定义 + 迭代循环） | ✅ |
| **judge 能力基准**（真值已知 + 常数基线 + 退化检查） | ✅ |
| **ACI 在线校准**（含饱和诊断 + DtACI） | ✅ |
| **Ring buffer**（时长+字节双约束 + 重采样） | ✅ |
| **评测指标**（帕累托 / 优雅退化 / 逐分层 ROC / 成本交叉点） | ✅ |
| **Analyst 工具集**（7 个工具 + 预算计量） | ✅ |
| **SafeWatch-Bench 接入**（1400 条 → SG2 格式，0 error） | ✅ |
| **政策敏感度指标**（反事实翻转，实测基线 1.00） | ✅ |
| **政策增强**（六种，防 SFT 学成背政策） | ✅ |
| **政策动物园**（多 schema 混合 + 数量/多样性解耦消融） | ✅ |
| **反事实一致性奖励**（RL 能做而 SFT 做不到的） | ✅ |
| 动作协议解析（严格，不修补） | ✅ |
| 端到端 Pipeline 编排 | ✅ |
| **SigLIP 2 编码器** | ✅ 真实权重跑通：1152 维，18 帧/秒，2.2 GB |
| **Qwen3-VL-2B 流式后端** | ✅ 真实模型跑通：4.0 GB，1.2 s/tick |
| 多卡分片 / 4-bit 量化（8B 可在 4×11GB 上跑） | ✅ |
| Agent / 中间层 LoRA **训练** | ⬜ 待 ≥40 GB 卡 |
| 组件注册表（按名字换实现） | ✅ |
| 编码器 / Sentinel / CUSUM 控制器 | ✅ 含 mock，无 GPU 可跑 |
| Agent / 中间层实际训练 | ⬜ 待 GPU |

### 快速开始

```bash
make install     # pip install -e ".[dev]"
make test        # 327 passed
make validate    # clean 0 error;dirty 必须报 2 个拼接泄漏
make synth       # 用 lavfi 合成 demo 数据集,不需要真实素材
make audit       # 压缩域对抗泄漏审计
make doctor      # 环境自检：驱动/torch/显存/配置/磁盘
```

`make synth` 用 ffmpeg 的 lavfi 自己生成素材，所以**无需任何真实数据**
就能端到端验证整条合成流水线。

### 选型（理由见 01_MODELS）

| 层 | 决定 |
|---|---|
| Sentinel | SigLIP 2 so400m，冻结，逐帧 + 压缩域/ASR/OCR/弹幕多通道融合 |
| 中间层 | Qwen3-VL-8B-Instruct（Apache 2.0），备选 InternVL3.5-8B |
| Analyst | Frontier API，接口层统一，换 backbone 即实验 |
| 训练框架 | ms-swift |

### 少样本改边界

四个机制，数据需求差两个数量级。A/B/C 改**边界**，D/E 给**保证**，顺序不能错。

| 机制 | 需要样本 | 改什么 | 有保证 |
|---|---|---|---|
| A 原型平移 | 1–5 | 边界位置 | ❌ |
| B 缓存检索 | 5–30 | 边界形状 | ❌ |
| C 线性头重拟合 | 30–200 | 边界方向 | ❌ |
| D 阈值重校准 | **≥45 正例** | 只动 τ | ✅ conformal |
| E 在线 ACI | 持续反馈 | τ 随时间 | ✅ 长程覆盖 |

45 这个数来自精确二项（Clopper-Pearson）单边下界：`n ≥ ln δ / ln target`。
阈值按**置信下界**选而非经验召回——经验召回 95% 不等于「recall ≥ 95% 有 90% 置信」。

### 四条设计原则

1. **不掩盖失败。** 校准不足报 `estimated(n=12)`；`claim_recall_bound()` 未校准时抛异常。
2. **规矩写进代码。** 三池分离在 loader 强制，统一重编码在构造函数强制，
   拼接泄漏在校验器用置换检验查。写在文档里的规矩会被违反。
3. **边界一动，保证立即失效。**
4. **难度是四个独立的轴。** 保留因子才有诊断力。

### 已知阻塞

1. **磁盘** — 完整数据集约需 2.2 TB。
2. **数据** — SafeWatch-Bench 已接通（1400 条，0 error）。
   ⚠️ 但它**没有时间戳**（全库实测），检测延迟在其上无真值——
   要测延迟必须把它的片段插进长宿主视频。见 [02_DATASET §2.2](docs/02_DATASET.md)。
3. **驱动**（比显存更早卡住）— 460.91（2021-07），最高 CUDA 11.2。
   cu121/cu124 直接 `RuntimeError: driver too old`；靠 CUDA minor 兼容只能用 **cu118**。
   升级驱动需 root，且这是共享机器。`make doctor` 会按约束顺序逐层检查。
4. **GPU** — 4× RTX 2080 Ti 11GB（Turing sm_75，无 bf16/FA2）；8B LoRA 需 ≥40GB 卡。

## License

Apache-2.0
