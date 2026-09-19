# SG2 总览与决策表

本目录是 StreamGuard-2 的实现文档。提案（研究论证）在 Claude Docs
"StreamGuard 2.0 — Proposal v0.2"；本目录只讲**怎么做**。

## 文档索引

| 文档 | 内容 |
|---|---|
| `00_OVERVIEW.md` | 本文。决策表、里程碑、关键风险 |
| `01_MODELS.md` | 三层 base model 选型与硬件适配 |
| `02_DATASET.md` | 数据集构建：来源、合成流水线、切分、规模 |
| `03_ADAPTATION.md` | **少样本决策边界调整**（四个机制 + 校准协议） |
| `04_TRAINING.md` | 训练框架、配方、超参、命令 |
| `05_RUNTIME.md` | 推理栈、agent、如何跑 |
| `06_EXPERIMENTS.md` | 预实验与核心实验的可执行协议 |
| `../SPEC.md` | 数据集格式规范 v1.0（已冻结） |

## 一句话架构

三层级联，**所有 backbone 冻结**，分配层是统计不是学习：

```
压缩域+ASR+OCR+弹幕+SigLIP2 → 融合分数 → CUSUM → [8B 中间层] → [Frontier analyst]
                                            ↑
                                    保底随机覆盖 ρ（旁路，永远开）
```

## 决策表（全部已定，理由见各文档）

| 项 | 决定 | 理由 |
|---|---|---|
| Sentinel 视觉编码器 | **SigLIP 2 so400m，冻结，逐帧** | 冻结线性探针最强；逐帧正是 needle 检测要的粒度 |
| Sentinel 融合头 | ~2M 参数 MLP，可训 | §4.5 允许；分钟级重训 |
| 时序通道（C 类） | InternVideo2 clip 编码器，可选 | SigLIP2 逐帧平均对时序组合无效 |
| 中间层 | **Qwen3-VL-8B-Instruct**（Apache 2.0） | 2026 开源全能默认；8B 对应 1.0 的角色 |
| 中间层降级选项 | Qwen3-VL-4B / 2B | 2B fp16 约 4.5GB，**本机 2080Ti 可跑** |
| Analyst | Frontier API（GPT-5.x / claude-opus-5 / claude-fable-5-1） | 稀疏调用，判断与政策推理 |
| 训练框架 | **ms-swift**（LLaMA-Factory 为备选） | 300+ MLLM，原生支持 Qwen3-VL / InternVL3.5 |
| Sentinel 训练 | 原生 PyTorch + sklearn | 只有 2M 参数，不需要框架 |
| 推理服务 | vLLM（中间层）；sentinel 直接 torch | vLLM 支持 sm_75 但**须 fp16**，Turing 无 bf16 |
| 数据集切分 | 按**源视频**切，不按片段 | 同源片段进不同 split 会泄漏 |
| 校准粒度 | **逐类别**（跨难度 bin 汇总） | 逐 cell 校准需 45×cells 正例，不可行 |
| 少样本边界 | 原型 + Tip-Adapter 缓存（免训练） | 1 例起效，毫秒级 |
| 边界改动后 | **强制重校准，发布为新 epoch** | 边界一动，旧阈值的 conformal 保证立即失效 |

## 关键设计原则

1. **不会掩盖失败。** 校准不足时运行时必须报 `uncalibrated(n=12, lower_bound=0.82)`，
   绝不假装有保证。适配 API 返回校准状态，调用方拿不到自己没挣到的保证。
2. **规矩写进代码，不写进文档。** 三池分离在 loader 里强制（`PoolViolation`），
   统一重编码在构造函数里强制，拼接泄漏在校验器里查。
3. **能被证伪的量优先。** 每个实验都先写好否定条件（见 `06_EXPERIMENTS.md`）。

## 里程碑

| # | 内容 | 依赖 | 状态 |
|---|---|---|---|
| M0 | 格式规范 + 校验器 | — | ✅ 完成 |
| M1 | 预实验 1–3（成本交叉点 / A-B-C 分层 / 阈值曲线） | needle 帧 + API | ⬜ 缺数据 |
| M2 | 合成流水线 | ffmpeg ✅ | ✅ 完成,40 条 demo 端到端验证 |
| M3 | Sentinel（冻结 + 融合头 + 少样本适配） | M2 | 🔄 压缩域通道 ✅ / 融合头 ✅ / 视觉通道待真实权重 |
| M4 | Agent 与运行栈 | M3 | ⬜ |
| M5 | 中间层 LoRA | 外部 GPU | ⬜ |

## 当前阻塞（按解除顺序）

1. **磁盘** — `/data` 剩 67G、`/data2` 剩 49G、`/` 剩 99G，三卷共享。
   合成长视频按每条 ~1G 算，215G 只够几百条，且要和他人抢。**比 GPU 更早的瓶颈。**
2. **数据** — 本机无任何视频安全数据。需要 SafeWatch 类别表 + 数据获取路径。
3. **GPU** — 4× RTX 2080 Ti 11GB，Turing sm_75，**无 bf16、无 FlashAttention-2**。
   可训 sentinel，不能训 8B。M5 需要 ≥40GB 卡。
