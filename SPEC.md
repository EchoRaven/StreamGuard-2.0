# StreamGuard-2 数据集格式规范 v1.0

本规范定义 SG2 数据集的存储布局与清单（manifest）格式。设计目标是让
**「决定部署召回的那些量」可测**——现有视频安全 benchmark 都不控制这些量。

## 0. 设计约束（为什么是这个格式）

| 约束 | 格式上的体现 |
|---|---|
| 检测延迟 E[τ−ν] 必须可测 | `events[].t_start_s` 是**必填**的帧级真值 ν |
| 难度不是一个轴 | `axes` 下四个**互相独立**的因子，easy/hard 只是派生切片 |
| 难度标签不能随模型漂移 | `axes` 只存**内在可测属性**；A/B/C 是 `measurements/` 下带模型版本与日期的独立测量 |
| 校准集不得被污染 | `pool` 字段在**格式层**强制三池分离，loader 拒绝越界使用 |
| 拼接不得泄漏标签 | `splice` 记录全部剪辑点，供泄漏审计；正负例拼接同分布是**可校验不变量** |
| 廉价通道会带来假阳性 | `metadata_adversarial.condition` 独立成轴 |

## 1. 目录布局

```
<root>/
├── manifest.jsonl              # 每行一条 ClipRecord
├── videos/<id>.mp4             # 媒体（或 sources.jsonl 指向外部 URL+时间戳）
├── asr/<id>.json               # 转写（可选）
├── chat/<id>.jsonl             # 直播弹幕（可选）
├── spec/taxonomy/<name>.json   # 分类体系（继承 SafeWatch）
└── measurements/               # 对数据集做的、带版本的测量，不属于数据集本身
    └── abc.<cheap>__<frontier>.<date>.jsonl
```

## 2. ClipRecord

每条记录描述一个视频。完整 JSON Schema 见 `spec/manifest.schema.json`。

### 2.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✓ | 全局唯一 |
| `source` | object | ✓ | `kind` ∈ {`native`, `synthetic`}，许可与出处 |
| `media` | object | ✓ | 时长、fps、编解码、**统一重编码参数**、sha256 |
| `label` | object | ✓ | `safe` 布尔 + `categories` + `events[]` |
| `axes` | object | ✓ | 四个难度因子，**仅内在属性** |
| `metadata_adversarial` | object | ✓ | 标题/描述/ASR/弹幕 与 `condition` |
| `splice` | object \| null | ✓ | 合成来源的完整剪辑溯源；`native` 为 null |
| `split` | enum | ✓ | `train` / `calib` / `test` / `zeroday_holdout` |
| `pool` | enum | ✓ | `exemplar` / `compile` / `calibration` / `eval` |

### 2.2 `label.events[]` —— ν 的真值

```json
{
  "event_id": "e0",
  "category": "<taxonomy id>",
  "t_start_s": 2103.4,          // ν：不安全内容开始的时刻
  "t_end_s": 2104.9,
  "frame_start": 63102,
  "frame_end": 63147,
  "severity": "low|medium|high",
  "evidence_modality": ["pixel"],     // pixel|speech|ocr|motion|composition
  "sufficient_alone": ["pixel"],      // 哪些通道单独就足以判定
  "bbox_track": null,
  "annotator_ids": ["a1","a2"],
  "agreement": 0.83
}
```

`safe: true` 的记录 `events` 为空数组。**一条视频可以有多个 event。**

### 2.3 `axes` —— 四个难度因子

每个因子存**连续的内在度量** + 一个派生 `bin`。`bin` 只是为了方便切片，
分箱规则在 `sg2/bins.py`，可重算；连续量才是真值。

```json
{
  "temporal_sparsity": {
    "needle_total_s": 1.5,
    "video_duration_s": 3600.0,
    "ratio": 0.000417,
    "bin": "ultra_sparse"            // dense|sparse|ultra_sparse
  },
  "perceptual_subtlety": {
    "min_evidence_pixel_area_frac": 0.004,
    "evidence_contrast": 0.21,
    "evidence_span_frames": 45,
    "requires_ocr": false,
    "bin": "subtle"                  // salient|moderate|subtle
  },
  "modality_locus": {
    "decisive": ["pixel"],
    "pixel_only_sufficient": true,
    "bin": "visual"                  // visual|speech|text|composition|multi
  },
  "context_dependence": {
    "frame_alone_sufficient": true,
    "window_required_s": 0.0,
    "bin": "self_contained"          // self_contained|short_context|long_context
  }
}
```

**`axes` 里不存 A/B/C。** 见 §4。

### 2.4 `metadata_adversarial`

```json
{
  "title": "...",
  "description": "...",
  "asr_path": "asr/<id>.json",
  "chat_path": null,
  "condition": "aligned"       // aligned | fp_bait | camouflage
}
```

- `aligned`：元数据与视频标签一致（常规情形）
- `fp_bait`：视频 **benign**，但标题/描述/弹幕含不安全词 → 压测廉价文本通道的假阳性
- `camouflage`：视频 **unsafe**，但元数据与语音全部无害 → 廉价通道全线失效，强制走视觉

### 2.5 `splice` —— 拼接溯源

```json
{
  "host_id": "host-0042",
  "n_cuts": 4,
  "cuts": [
    {"t_s": 120.0,  "kind": "benign", "insert_id": "b-771"},
    {"t_s": 2103.4, "kind": "needle", "insert_id": "n-991"}
  ],
  "audio_crossfade_ms": 40,
  "loudness_lufs": -23.0,
  "reencoded_uniformly": true
}
```

**判据不是「视频连贯」，而是「剪辑痕迹不携带标签信息」。** 因此：

1. 正负例都拼接，`n_cuts` 与位置分布同分布（`validate.py` 校验）
2. 整条合成视频统一重编码（压缩域 sentinel 会读编码器指纹，不统一等于自埋捷径）
3. 音频交叉淡入 + EBU R128 统一响度
4. 泄漏审计：只用剪辑点位置训分类器预测标签，超过随机即判泄漏

## 3. `pool` —— 格式层强制三池分离

| pool | 用途 | 禁止 |
|---|---|---|
| `exemplar` | analyst 的 in-context 示例、case library | 不得进校准 |
| `compile` | prompt / workflow 离线编译的验证信号 | 不得进校准 |
| `calibration` | conformal 阈值拟合 | **不得被模型以任何形式见过** |
| `eval` | 最终评测 | 不得进前三者 |

同一批样本既当 exemplar 又当校准集会使校准失效（模型见过 → 分数乐观偏置 →
阈值偏松）。写在文档里会被违反，所以**写进 loader**：`sg2.schema.load()`
接受 `pool=` 参数并拒绝越界读取。

## 4. A/B/C 是测量，不是标签

若把「廉价模型看不出」存成数据集标签，模型一变强标签就漂移，benchmark 过期。
因此 A/B/C 存在 `measurements/` 下，每次测量一个文件：

```
measurements/abc.siglip-so400m__gpt-5.2026-09-19.jsonl
```

每行：
```json
{"id":"s2s-0001","event_id":"e0","cheap_ok":false,"frontier_ok":true,
 "stratum":"B","cheap_model":"siglip-so400m","frontier_model":"gpt-5",
 "measured_at":"2026-09-19","protocol":"exact_frame_v1"}
```

分层定义（两个模型都喂**精确 needle 帧**，不抽样）：

| cheap | frontier | stratum | 含义 |
|---|---|---|---|
| ✓ | ✓ | A | 短但显著。覆盖率决定一切 |
| ✗ | ✓ | B | 短且细微。级联的真正问题区 |
| ✗ | ✗ | C | 不在这一帧里，需时序/语音通道 |

副产品：同一数据集上 A/B/C 比例随模型代际的迁移曲线，是 model-swap 纵向研究的天然舞台。

## 5. 分类体系

继承 SafeWatch，不自造。`spec/taxonomy/safewatch.json` 为可插拔文件。
类别标签是**必填**的，因为以下四件事没有它直接失效：

1. 逐类别 conformal 校准（每类别约需 45 个正例，见提案 §4.1）
2. 零日政策实验需要故意留出类别（`split: zeroday_holdout`）
3. 政策条款引用需要类别→条款映射
4. 「稀有=严重」的非对称预算是类别层面的事实

但论文的头条轴是 `axes` 的四个难度因子，分类只是分层变量，不是贡献。

## 6. 版本

`manifest.jsonl` 首行可选 `{"__meta__": {"spec_version": "1.0", ...}}`。
本规范变更走语义化版本；`axes` 增删因子属 major。
