# 02 · 数据集构建

格式规范见 `../SPEC.md`（已冻结 v1.0）。本文讲**怎么造出来**。

## 1. 三个子集，职责不同

| 子集 | 来源 | 规模目标 | 作用 |
|---|---|---|---|
| **SG2-Synth** | 合成（拼接） | ~6000 条 | 主力。受控 ν、受控难度因子 |
| **SG2-Native** | 原生长视频 | ~400 条 | 外部效度验证。零拼接痕迹 |
| **SG2-Context** | 原生，语境型 | ~300 条 | 上下文依赖轴**必须**原生 |

规则（第 9 轮结论）：**needle 自足的轴用合成；伤害依赖语境的轴必须原生。**
语境型伤害一拼接就把要测的东西毁了。

## 2. 素材来源

### 2.1 宿主视频（长 benign）
- Internet Archive 的 CC / 公共领域长录像
- YouTube CC-BY 长直播回放（需存 URL+时间戳而非内容，见 §6）
- 已有素材：`/data/common/haibotong/tiktok_material/`（短，仅够做插入片段）

要求：时长 ≥ 20 min，有音轨，画面多样（避免全是静态讲话头）。

### 2.2 Needle 片段（unsafe）
- **SafeWatch-Bench**（分类体系也继承自它，见 §5）
- StreamGuard 1.0 的 Safe2Shot：4K+ unsafe 视频、**帧级标注** ← 关键，直接给 ν
- AdvVideo-Bench：对抗子集

### 2.2 ⚠️ 已核实：SafeWatch-Bench 没有时间戳

实测（2026-09-19，全库 1400 条标注）：

```
字段: video_path / labels / subcategories / video_content / violate_reason
带时序字段的条目: 0
```

**标注是视频级的，拿不到 ν。** 这不是抽样结论，是全库 1400 条逐条统计。

后果：

| 影响 | 说明 |
| --- | --- |
| 检测延迟 E[(τ−ν)⁺] | **在这个数据集上无真值**，不能直接测 |
| 能测什么 | 「整条视频安全与否」的判定、分类 F1、引用准确率 |
| 要测延迟怎么办 | 把它的片段当作 **needle 插进长宿主视频**（合成流水线），此时 ν 由我们自己的拼接规划给出 |

格式层面已支持这种数据：`ClipRecord.granularity ∈ {temporal, video_level}`。
`video_level` 的记录 `has_nu == False`，**会被排除出延迟统计**——
`safe=false 必须有 events` 这条不变量只在 `temporal` 下强制。

### 2.3 实测的库存

| 项 | 数值 |
| --- | --- |
| 标注条目 | 1400（real 810 / genai 590） |
| 视频文件 | 1400 个 mp4，抽样实测 0.9–3.5 MB、5–160 s |
| unsafe / benign | 1095 / 305 |
| 类别分布 | C1:267 C2:141 C3:334 C4:177 C5:192 C6:229 |
| 子任务 | 38 种 |
| 访问 | **gated: manual**，需在 HF 页面申请并等作者批准 |

> 论文说 2M 视频，但 HF 上这个仓库只有 1400 个——它是**评测子集**而非
> 训练全集。好消息是体量可控（估计 20–50 GB），本机装得下。

**免费的分层划分。** SafeWatch 自己就区分了明显与隐晦：C1 下有
`evident` / `subtle` / `implication`。这直接对应我们感知细微度轴的
A/B/C 分层——初始划分可以借用它，再由 2×2 自动分层校准，
不必从零构造。`sg2.policy.SAFEWATCH_SUBTASKS` 记了全部 38 种。

### 2.4 拉取

```bash
export HF_TOKEN=hf_...          # 需先申请 gate 批准
python scripts/fetch_safewatch.py --meta-only          # 只拉标注,转成 SG2 格式
python scripts/fetch_safewatch.py --n-videos 50        # 再下 50 个视频
```

产出 `manifest_safewatch.jsonl`，1400 条，`sg2.validate` **0 error**。

⚠️ 未下视频时 `duration_s` / `fps` 填 **`null` 而不是 0**——
填 0 是在假装知道一个不知道的值，下游按时长归一化时会除零。

### 2.3 干扰片段（benign，用于对照组）
从同一批宿主视频里切出，与 needle 同长度分布。**这是防泄漏的关键素材**。

## 3. 合成流水线

```mermaid
flowchart LR
  H[宿主长视频] --> N[归一化<br/>分辨率/fps/响度]
  N --> P[规划拼接点<br/>随机位置+数量]
  P --> S{正例?}
  S -->|是| I1[插入 needle 片段]
  S -->|否| I2[插入 benign 片段<br/>同数量同分布]
  I1 --> R[统一重编码<br/>libx264 crf23 gop250]
  I2 --> R
  R --> M[写 manifest<br/>含 splice 溯源]
  M --> V[validate.py<br/>泄漏审计]
```

### 3.1 判据：不是"视频连贯"，是"剪辑痕迹不携带标签信息"

后者便宜得多，也才是正确门槛。真实直播本来就到处是切镜头。具体要求：

1. **正负例都拼接，剪辑数与位置同分布**（`check_splice_balanced` +
   `check_cut_position_leakage` 用置换检验查）
2. **整条统一重编码** —— 压缩域 sentinel 读的就是编码器指纹，不统一等于自埋捷径。
   `Splice(reencoded_uniformly=False)` 构造即报错。
3. 音频交叉淡入 40ms + EBU R128 响度归一（`loudnorm=I=-23`）
4. 边界色彩/曝光匹配
5. 拼接位置、时长、数量全部随机化

**三处必须与标签无关地采样**（`sg2/synth/plan.py`）——漏掉任何一处都泄漏：
切点**数量**、切点**位置**、插入片段**时长**。第三处最容易漏：needle 总是
1–3s 而 benign 插入总是 10s 的话，时长本身就是标签。因此 benign 插入的时长
**从 needle 的经验时长分布里采样**。

### 3.1.1 泄漏审计分两层，缺一不可

| 层 | 模块 | 查什么 |
|---|---|---|
| 规划层 | `sg2/validate.py` | 切点数/位置/时长是否与标签独立（置换检验） |
| 渲染层 | `sg2/audit.py` | 用 sentinel 实际会读的**压缩域特征**训分类器猜标签 |

> ⚠️ **实测发现**：x264 在场景切换处强制插 I 帧，所以**所有切点在压缩域一定
> 可见**。e2e 验证中 needle 边界 107.14/110.14s 处确实出现关键帧 107.3/110.33，
> benign 切点处同样出现。
>
> 这不是泄漏 —— **切点可见 ≠ 标签可见**。只要正负例切点同分布，关键帧模式
> 就不含标签信息。40 条 demo 实测：随机森林 AUC 0.500、逻辑回归 0.536，
> 准确率（0.525/0.550）均**低于**多数类基线 0.575，置换检验 **p=0.577**（200 次置换）。
>
> 但反过来说：**规划层一旦出错，压缩域会把错误直接喂给 sentinel**，
> 它会学到剪辑点而不是内容。所以两层都必须过。
>
> ⚠️ 小样本检验力低。实测 n=40 时 AUC 标准差 ±0.162，`audit.py` 会在
> n<200 时标记 `underpowered` 并在输出里明写「未检出 ≠ 不存在」。
>
> 运行成本：首次 108s（含 31s 特征抽取），命中缓存后 75s。

### 3.2 ffmpeg 参数（已验证可用：7.0.2 static，含 libx264）

```bash
FFMPEG=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")

# 归一化（宿主与插入片段都要，先统一再拼）
$FFMPEG -i in.mp4 -vf "scale=1280:720:force_original_aspect_ratio=decrease,\
pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30" \
  -af "loudnorm=I=-23:LRA=7:TP=-2" \
  -c:v libx264 -crf 20 -preset medium -g 250 -pix_fmt yuv420p \
  -c:a aac -b:a 128k norm.mp4

# 拼接后统一重编码（不要用 -c copy，会保留各片段的编码器指纹）
$FFMPEG -f concat -safe 0 -i list.txt \
  -c:v libx264 -crf 23 -preset medium -g 250 -pix_fmt yuv420p \
  -c:a aac -b:a 128k out.mp4
```

⚠️ **严禁 `-c copy`**。拼接必须重编码，否则各片段的量化参数、GOP 结构差异
会成为与标签相关的捷径。

### 3.3 压缩域特征抽取（不解码成 RGB）

```bash
# 运动矢量
$FFMPEG -flags2 +export_mvs -i in.mp4 -vf codecview=mv=pf+bf+bb -f null -
# 帧类型与码率(逐帧)
ffprobe -select_streams v -show_frames \
  -show_entries frame=pict_type,pkt_size,best_effort_timestamp_time \
  -of csv in.mp4
```

## 4. 切分与规模

### 4.1 切分规则：按**源视频**切，不按片段

同一宿主或同一 needle 源出现在两个 split 会泄漏。`split_by_source()` 先对
`source.provenance` 做分组，再分配。

### 4.2 池与 split 的关系

`pool` 和 `split` 是**正交**的两个字段（SPEC.md §3）：

| pool | split | 用途 |
|---|---|---|
| `exemplar` | train | analyst in-context 示例、case library |
| `compile` | train | prompt/超参（α, β）调优的验证信号 |
| `calibration` | calib | conformal 阈值拟合。**模型不得以任何形式见过** |
| `eval` | test | 最终评测 |
| `eval` | zeroday_holdout | 零日政策实验，整个类别留出 |

### 4.3 规模推算

校准粒度**逐类别**（跨难度 bin 汇总）。逐 cell 校准需要 45×cells 正例，
不可行；逐 cell 只报观测值，不给保证。

设 SafeWatch 有 C≈8 个类别：

| 池 | 每类别正例 | 合计正例 | 负例(3×) | 说明 |
|---|---|---|---|---|
| calibration | **120** | 960 | 2880 | 120 > 45，留出余量允许若干漏报 |
| exemplar | 40 | 320 | 320 | 少样本机制只需几十 |
| compile | 40 | 320 | 960 | 调 α/β |
| eval | 200 | 1600 | 4800 | 逐 cell 拆开后每格仍有样本 |

**正例合计约 3200，负例约 9000，总计约 12000 条。**
其中 2 个类别整体划入 `zeroday_holdout`，不进 exemplar/calibration。

### 4.4 难度轴的因子设计

全因子在 2 个主轴上，其余 2 轴做部分因子（全因子会爆炸）：

- **主轴**：时序稀疏度 (3) × 感知细微度 (3) = 9 格，每类别每格 ≥ 20 条
- **副轴**：模态位点 (5)、上下文依赖 (3) 在 eval 池的子集上分层，不求满格
- **元数据轴**：`aligned` : `fp_bait` : `camouflage` ≈ 6 : 2 : 2

## 5. 分类体系

继承 SafeWatch，**不自造**。`spec/taxonomy/safewatch.json` 目前是占位文件，
需从正式发布抄录类别 id。

类别标签必填，因为四件事没它直接失效：逐类别校准、零日留出、政策条款引用、
"稀有=严重"的非对称预算。但论文头条轴是难度因子，分类只是分层变量。

## 6. 法律与伦理

部分类别根本不能再分发。三种标准做法，按类别选：

| 方式 | 适用 | 代价 |
|---|---|---|
| 发 URL + 时间戳 | 公开平台来源 | 链接会失效，复现性下降 |
| 只发特征/embedding | 高危类别 | 别人无法换编码器复现 |
| 门控访问（申请制） | 中等敏感 | 使用门槛 |

**这会同时是审稿问题和 IRB 问题，而且反过来限制能收哪些类别。**
需要在收数据之前定，不是之后。

## 7. 磁盘预算

| 项 | 估算 |
|---|---|
| 宿主归一化后 720p30，20min | ~250 MB/条 |
| 合成输出 12000 条 × 平均 15min | **~2.2 TB** |
| 抽帧缓存（needle 精确帧） | ~20 GB |
| 压缩域特征 | ~5 GB |

当前 `/data` 剩 67G、`/data2` 剩 49G、`/` 剩 99G，**共约 215G，远远不够**。

缓解顺序：
1. 先只做 **SG2-Synth-mini**（每类别 30 条，共 ~400 条，约 80 GB）跑通全流程
2. 降到 480p / crf 28 可省约 60%
3. 中长期必须申请独立存储
