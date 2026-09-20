# 01 · Base Model 选型

## 0. 选型原则

每层的选型由它**在级联中的职责**决定，不由"哪个模型更强"决定。这是提案
第 8 轮的结论：闭源模型赢在通用理解与判断，不赢在"当一个廉价高召回触发器"。

| 层 | 职责 | 要的能力 | 不要的能力 |
|---|---|---|---|
| Sentinel | 分配预算 | **敏感**、连续分数、极低单价 | 聪明、会讲理由 |
| 中间层 | 细粒度判别 | 判别力、可微调 | 政策推理 |
| Analyst | 判断与引用 | 通用理解、政策推理 | 便宜 |

## 1. Sentinel 层

### 1.1 视觉通道：SigLIP 2 so400m（冻结）

**决定**：`google/siglip2-so400m-patch14-384`，fp16，冻结，**逐帧**打分。

理由：
- 冻结表征做线性探针/少样本，SigLIP 2 大幅优于 SigLIP 与 CLIP（[SigLIP 2 论文](https://arxiv.org/html/2502.14786v1)）。
- **逐帧正是我们要的粒度。** 已知 SigLIP 2 抽 8 帧取平均做视频*检索*很弱
  （NDCG@10 仅 0.325，[Mixpeek 2026 基准](https://mixpeek.com/blog/video-embedding-benchmark-2026)），
  但那测的是视频级语义检索；needle 检测本来就是逐帧问题，不做时序平均。
- 输出连续 cosine 相似度 → **CUSUM 要的就是平滑连续统计量**。小 VLM 吐
  safe/unsafe 是 1-bit 信号，做 CUSUM 输入很糟。

⚠️ 例外：C 类（时序组合型伤害）逐帧必然失败。该类走 **InternVideo2** clip
编码器作为独立通道，仅在 `modality_locus.bin == "composition"` 的流量上启用。

### 1.2 非视觉通道（都比视觉便宜几个数量级）

| 通道 | 实现 | 成本 | 覆盖的伤害类型 |
|---|---|---|---|
| 压缩域 | ffmpeg 读 H.264 运动矢量/残差/I 帧/码率 | **不解码即可得** | 场景切换、剧烈动作 |
| ASR | faster-whisper (CTranslate2, int8) | ~0.05× 实时/CPU | 威胁、骚扰、自残、极端主义 |
| OCR | PaddleOCR / RapidOCR | 低 | 屏幕文字类 |
| 弹幕 | 直接解析 | ~0 | 直播早期信号（**常早于任何分类器**） |

### 1.2b ⚠️ 冻结 SigLIP **给不出判决**

这一点先前表述含糊，需要说死：

> **冻结的 SigLIP 输出的是 1152 维向量，不是「安全/不安全」。**
> 它自己判不了任何东西。

要得到判决，必须在它之上再加一层，而三种加法的**标注需求完全不同**：

| 方式 | 需要什么 | 是否 training-free |
| --- | --- | --- |
| 零样本文本对比 | 只要提示词（用 SigLIP 的**文本塔**） | ✅ 真的不需要标注 |
| 原型 / kNN | k 个带标注正例 | ⚠️ 需要标注，但不需梯度 |
| 线性探针 / 融合头 | 全量标注 + 拟合 | ❌ 要训练（虽然只有 2M 参数） |

所以「training-free」这个说法必须精确到**哪一层冻结**：
**frontier backbone 冻结是真的，但廉价判别层需要标注。**
这不影响提案的主张（§1 说的是 backbone 可插拔），但论文里不能写成
「整个系统无需标注」。

`SigLIP2Encoder.encode_text()` / `zero_shot()` 提供了第一种方式。
`scripts/sentinel_viability.py` 在真实 SafeWatch 数据上量三者的差距 ——
**零样本与 proto-k 的差，就是标注买到了多少**。

### 1.3 融合头（唯一可训的部分）

输入：各通道分数 + 置信度，约 32 维。
结构：`Linear(32→64) → GELU → Linear(64→1)`，约 2M 参数以内。
输出：单一连续风险分数 `s ∈ ℝ`，供 CUSUM 与阈值使用。

重训成本：分钟级。**重训 sentinel ≠ 重训 guardrail。**

## 2. 中间层

**决定**：`Qwen/Qwen3-VL-8B-Instruct`，LoRA 微调。

理由：
- Qwen3-VL 是 2026 开源全能默认，dense 版 **Apache 2.0**（商用与再分发干净）。
- 有 2B/4B/8B/30B-A3B/32B/235B-A22B 全谱系 + Instruct/Thinking 两个变体 +
  官方 FP8 量化，降级路径连续。
- 8B 恰好对应 StreamGuard 1.0 在级联里的角色。

**备选**：`OpenGVLab/InternVL3_5-8B`。许可同样干净，MVBench/MLVU 上常优于
同尺寸 Qwen；且其 pixel shuffle（1024→256 token/tile，Flash 变体 64）对
长上下文流式更友好。**M5 时两个都跑一遍，按 §7.3 实验 2 的 B 类表现选。**

不选 LLaVA-OneVision：同尺寸下判别表现落后于 InternVL3 与 Qwen2.5-VL。

## 3. Analyst 层

**决定**：Frontier API，不自建。可换，且换 backbone 本身是实验（提案 §7.3-4）。

候选：`gpt-5.x` / `claude-opus-5` / `claude-fable-5-1` / `gemini-*`。
接口层统一成 `sg2/analyst/base.py` 的 `AnalystBackend` 协议，换模型只改配置。

⚠️ 每个 backbone 的**拒绝率作为发现报告，不隐藏**；且需注明拒绝行为随厂商
策略变化，结果不可跨时复现。走内容审核用途的正式访问通道，不做工程绕开。

## 4. 本机显存适配（4× RTX 2080 Ti，11GB，Turing sm_75）

Turing 的两个硬限制，影响所有选型：
- **无 bf16**（需 sm_80+）→ 必须 fp16，而多数现代 VLM 是 bf16 原生，需注意溢出
- **FlashAttention-2 需 sm_80+，用不了** → 长上下文吞吐大打折扣

| 模型 | fp16 权重 | 11GB 单卡 | 说明 |
|---|---|---|---|
| SigLIP 2 so400m | ~1.6 GB | ✅ 宽裕 | sentinel 主力 |
| InternVideo2-1B | ~2 GB | ✅ | 时序通道 |
| Qwen3-VL-2B | ~4.5 GB | ✅ | **预实验的"廉价模型"角色** |
| Qwen3-VL-4B | ~9 GB | ⚠️ 勉强 | 需限制帧数与上下文 |
| Qwen3-VL-8B | ~17 GB | ❌ | 4-bit nf4 约 6GB 可推理，但 Turing int4 kernel 差 |
| Qwen3-VL-8B LoRA 训练 | ~24–40 GB | ❌ | **必须外部 GPU** |

### 4.1 11GB 单卡放不下 8B —— 但有两条路

磁盘（`/data` 约 78 GB 可用）不是瓶颈，**显存才是**。8B fp16 权重约 17 GB，
单张 11 GB 卡放不下。两条可行路径：

| 方式 | 显存 | 代价 | 配置 |
| --- | --- | --- | --- |
| **4-bit 量化（nf4）** | ~6 GB，单卡 | Turing 的 int4 kernel 一般，吞吐会掉 | `quantization="nf4"` |
| **多卡分片** | 17 GB 摊到 4×11 GB | 层间通信走 PCIe（无 NVLink） | `device="auto"` + `max_memory_per_gpu` |

⚠️ `max_memory_per_gpu` 要给每张卡**留出激活的余量**。按权重填满会在前向时 OOM。
本机 11 GB 卡建议设 `"9GiB"`。

⚠️ `device="auto"` 时不能再用 `cfg.device` 搬输入张量——"auto" 不是设备名。
代码里取 `model.device`（第一层所在的卡）。

### 4.2 实测：四方对照（2B / 4B / 8B-nf4 / 8B-分片）

8 帧 × 三种政策语料，本机 4× RTX 2080 Ti。

#### 资源与吞吐

| 配置 | 显存峰值 | 延迟/tick | 加载 |
| --- | --- | --- | --- |
| 2B fp16 单卡 | 4.08 GB | 1.23 s | 7.9 s |
| 4B fp16 单卡 | 8.50 GB | 0.37 s | 6.1 s |
| **8B nf4 单卡** | **6.17 GB** | 0.59 s | 12.0 s |
| 8B fp16 分片×4 | 16.86 GB | 0.72 s | 11.5 s |

> **nf4 推翻了我先前的假设。** 我原以为「Turing 的 int4 kernel 支持一般，
> 吞吐会掉」。实测 8B-nf4 比 8B 分片**更快且省 10 GB**——分片要跨 PCIe
> 传激活，而本机无 NVLink，通信开销盖过了 int4 kernel 的劣势。
> nf4 的 8B 甚至比 fp16 的 4B 还省显存。

#### 能力：4B 是断点

对比必须有**区分力**才有意义。前两种语料（完整六类 / 仅无关条款）
所有模型都输出 `hold`，因为 demo 视频是 `testsrc2` 彩条图，
里面没有任何像违规的内容——这只测出了"会不会照格式输出"。

第三种语料 `MATCHING` 写一条**确实匹配画面**的条款（禁止测试信号图），
此时正确行为是 flag 并引用它：

| 模型 | flag 率 | 引用可解析 | 引到目标条款 |
| --- | --- | --- | --- |
| 2B | **0/8** | — | — |
| 4B | 8/8 | 8/8 | **8/8** |
| 8B-nf4 | 8/8 | 8/8 | **8/8** |
| 8B-分片 | 8/8 | 8/8 | **8/8** |

**2B 读不懂政策，4B 起就完全正确，8B 没有额外增益。**
断点在 2B→4B 之间，不在 4B→8B。这直接影响中间层选型：
本机跑 4B 就够做协议层的开发与验证，不必每次都上 8B。

⚠️ 这个结论**只覆盖协议遵守**（能否按政策 flag 并正确引用），
不覆盖真实安全内容上的判准。后者要等真实数据。

#### 一个否定结果

`uncovered` 使用率：**2B / 4B / 8B-nf4 / 8B-分片 全部 0/8**。
即使只给一条完全无关的条款，四个配置都输出 `hold` 而非 `uncovered`。

**与模型规模无关。** 这个动作必须靠 SFT 教——
`docs/08_SFT_RL.md` §2.1 的窗口配比需要加第五类（out-of-policy 窗口），
否则训出来的模型也不会用它。

⚠️ **显存不会自动释放。** 顺序加载多个 backbone 时，`del model` +
`empty_cache()` **不够**：`device_map` 分片的模型会在 accelerate 的 hook 里
留下对各卡子模块的引用。实测先加载分片版再加载 nf4 版，总占用 22.29 GB
而非预期的 5.96 GB。`Qwen3VLStreaming.release()` 会移除 hook 并回收——
实测释放后归零。

### 4.3 8B 在 4×11 GB 上的两种跑法

8 帧 × 两种政策语料，Qwen3-VL，本机 4× RTX 2080 Ti：

| 配置 | 显存峰值 | 延迟/tick | 加载 | 格式合法率 |
| --- | --- | --- | --- | --- |
| 2B fp16 单卡 | 4.08 GB | 1.24 s | 8.2 s | 100% |
| **8B nf4 单卡** | **6.17 GB** | **0.58 s** | 11.6 s | 100% |
| 8B fp16 分片×4 | 16.86 GB | 0.71 s | 11.8 s | 100% |

> **nf4 是本机最优解，这推翻了我先前的假设。** 我原以为「Turing 的 int4
> kernel 支持一般，吞吐会掉」，实测 nf4 比 fp16 分片**更快**——分片要跨
> PCIe 传激活，而本机没有 NVLink，通信开销盖过了 int4 kernel 的劣势。
> 而且 nf4 只比 2B 多 2 GB 显存，单卡即可。

⚠️ **显存不会自动释放。** 顺序加载多个 backbone 时，`del model` +
`empty_cache()` **不够**：`device_map` 分片的模型会在 accelerate 的 hook 里
留下对各卡子模块的引用。实测先加载分片版再加载 nf4 版，总占用 22.29 GB
而非预期的 5.96 GB，逐次堆积后会在 NVML 探测处报一个**与量化无关**的错
（旧驱动上探测 NVLink 拓扑本身也会崩）。
`Qwen3VLStreaming.release()` 会移除 hook 并回收——实测释放后归零。

**结论**：本机可完成 M1–M4 的全部工作，并可用 nf4 或分片**实际运行 8B 推理**。
只有 M5（8B LoRA **训练**）仍需 ≥40 GB 卡——训练要存优化器状态和激活，
比推理多一个数量级。

## 5. 权重获取

```bash
# 需先设 HF_HOME 到有空间的卷
export HF_HOME=/data2/hf_cache          # 或任何有余量的位置
huggingface-cli download google/siglip2-so400m-patch14-384
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
```

⚠️ 下载前先解决磁盘。当前三卷共剩约 215G 且共享。

## Sources

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
- [InternVL3.5 论文](https://arxiv.org/pdf/2508.18265)
- [SigLIP 2 论文](https://arxiv.org/html/2502.14786v1)
- [视频 embedding 基准 2026](https://mixpeek.com/blog/video-embedding-benchmark-2026)
- [开源 VLM 指南 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
