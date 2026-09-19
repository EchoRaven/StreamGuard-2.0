# 04 · 训练框架与配方

**只有三样东西会被训练**：sentinel 融合头、蒸馏 sentinel、中间层 LoRA。
frontier backbone 永远冻结 —— 这是提案的前提，不是可选项。

| 被训对象 | 参数量 | 框架 | 硬件 | 耗时 |
|---|---|---|---|---|
| 融合头 | ~2M | 原生 PyTorch | 本机 1 卡 | 分钟 |
| 线性头/校准 | ~1K | sklearn（numpy 亦可） | CPU | 秒 |
| 蒸馏 sentinel | ~100M | 原生 PyTorch | 本机 1–4 卡 | 小时 |
| 中间层 LoRA | 8B × r16 | **ms-swift** | 外部 ≥40GB | 天 |

## 1. 环境

当前 `dt` env 只有 numpy + imageio-ffmpeg。建独立 env，别污染 `dt`：

```bash
conda create -n sg2 python=3.11 -y && conda activate sg2

# ⚠️ Turing sm_75:必须 cu121 或更早的轮子;新版 torch 可能已放弃 sm_75
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch;print(torch.cuda.get_device_capability(0))"   # 期望 (7,5)

pip install transformers accelerate sentencepiece pillow
pip install scikit-learn scipy pandas
pip install faster-whisper rapidocr-onnxruntime      # ASR / OCR 通道
pip install imageio-ffmpeg av
export HF_HOME=/data2/hf_cache                        # 别放 /data,已 99%
```

### 1.0 ⚠️ 先看驱动,它比显存更早卡住你

本机实测:

```
NVIDIA-SMI 460.91.03   Driver Version: 460.91.03   CUDA Version: 11.2
```

驱动 460.91 是 **2021 年 7 月**的版本,最高支持 CUDA 11.2。后果:

| 轮子 | 需要驱动 | 本机可用 |
|---|---|---|
| cu121 / cu124（torch 默认） | ≥ 525 | ❌ `RuntimeError: driver too old (found 11020)` |
| **cu118** | ≥ 450.80.02 | ✅ 靠 CUDA **minor 版本兼容** |
| cu117 及更早 | ≥ 450.80.02 | ✅ |

CUDA 的 minor 版本兼容规则:11.x 的 runtime 可以跑在任何支持 11.0+ 的驱动上
(即 ≥450.80.02)。跨 major(11→12)则不兼容,所以 cu12x 全部出局。

```bash
# 本机唯一可行的安装方式
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

⚠️ 这条**先于**下面所有 Turing 相关的考虑。显存不够只是跑不了大模型,
驱动不对是 `torch.cuda.is_available()` 直接返回 False,什么都跑不了。
升级驱动需要 root,且这是共享机器。

### 1.1 Turing 上的三个坑（会静默降速或报错）

1. **无 bf16。** 所有 `torch_dtype=torch.bfloat16` 改成 `float16`。
   Qwen3-VL 是 bf16 原生训练的，fp16 推理需警惕溢出 → 出现 NaN 时把
   attention 与 layernorm 保持 fp32（`attn_implementation="eager"`）。
2. **FlashAttention-2 需 sm_80+，装不上也用不了。** 显式指定
   `attn_implementation="sdpa"`，别让库去猜然后回落到最慢的实现。
3. **无 NVLink**，4 卡间走 PCIe。DDP 可用，模型并行不划算。

## 2. Sentinel 融合头

### 2.1 数据

离线预抽好各通道分数，训练时只读特征向量 —— 融合头的训练完全不碰视频。

```
features/<clip_id>.npz
  siglip   : (T, 1152)  逐帧 SigLIP2 嵌入
  motion   : (T, 4)     压缩域:mv 能量/残差/帧类型/码率
  asr      : (T, 1)     滚动窗口文本风险分
  ocr      : (T, 1)
  chat     : (T, 2)     速率/情绪
  y        : (T,)       逐帧标签,由 events[] 的 [t_start,t_end] 展开
```

### 2.2 配方

```yaml
model:      Linear(32→64) → GELU → Dropout(0.1) → Linear(64→1)
loss:       BCEWithLogits(pos_weight = N_neg/N_pos)   # 极度不平衡
optimizer:  AdamW(lr=3e-4, weight_decay=0.01)
scheduler:  cosine, warmup 5%
batch:      4096 帧
epochs:     20（早停:验证集 AUPRC 连续 3 轮不升）
选型指标:   AUPRC（不是 AUC —— 正例率 <1% 时 AUC 会虚高）
```

⚠️ **按源视频切验证集**，不能按帧切。同一视频的相邻帧几乎相同，按帧切会
把验证集变成训练集的副本，指标虚高十几个点。

### 2.3 命令

```bash
python -m sg2.train.fusion --features features/ --manifest manifest.jsonl \
    --pool compile --out ckpt/fusion.pt
```

## 3. 蒸馏 Sentinel（针对 B 类）

只在预实验 2 测出 **B 类占比高**时才做。训练集就是自动分出来的 B 层
（`measurements/abc.*.jsonl` 里 `stratum=="B"` 的那些）—— 难例挖掘直接闭环。

```yaml
student:    SigLIP2-base 的前 6 层 + 新分类头（~100M）
teacher:    frontier analyst 在精确帧上的 logit/verdict
loss:       0.7 * BCE(hard) + 0.3 * KL(student ‖ teacher, T=2)
输入:       全分辨率 crop，不是缩略图     # 见 07 三分诊的第 1 条病因
硬件:       本机 4×2080Ti DDP,fp16
```

**先跑三分诊**（`06_EXPERIMENTS.md` §2）：若病因是分辨率，蒸馏是浪费，
直接上 crop proposal 就够。

## 4. 中间层 LoRA（需外部 GPU）

### 4.1 框架：ms-swift

选它而非 LLaMA-Factory：支持 300+ MLLM，**原生覆盖 Qwen3-VL 与 InternVL3.5**
两个候选，换 backbone 不用换框架。LLaMA-Factory 作为备选，两者配方可互译。

```bash
pip install ms-swift[all]
```

### 4.2 配方

```bash
swift sft \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset sg2_midtier_train.jsonl \
  --train_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length 8192 \
  --gradient_checkpointing true \
  --attn_impl flash_attn \
  --save_steps 200 --eval_steps 200 \
  --output_dir ckpt/midtier-qwen3vl8b
```

关键选择及理由：

| 参数 | 值 | 理由 |
|---|---|---|
| `--freeze_vit true` | 冻结视觉塔 | 视觉编码器是这层的**能力来源**，微调它会在小数据上退化 |
| `--lora_rank 16` | 16 | 判别任务不需要大 rank；32 起收益递减 |
| `--torch_dtype bfloat16` | bf16 | **仅外部 Ampere+ 卡**。本机跑要改 float16 |
| `--attn_impl flash_attn` | FA2 | **仅外部卡**。本机改 `sdpa` |
| `--freeze_vit` + LoRA | — | 显存从 ~80GB 降到 ~40GB，单张 A100 可训 |

### 4.3 训练数据格式

中间层做的是**逐窗口判别**，不是生成。构造成短输出，省 token：

```json
{"messages": [
  {"role": "user", "content": "<video>判断该片段是否违反政策 C2。只回 yes 或 no。"},
  {"role": "assistant", "content": "yes"}],
 "videos": ["clips/sg2-00001_w042.mp4"]}
```

窗口由 `events[]` 展开：正窗口 = 与某 event 相交；负窗口 = 同视频内不相交
的随机窗口（**同视频负例**比跨视频负例难得多，是有价值的难例）。

## 5. 不训练但要"拟合"的东西

| 对象 | 方法 | 数据 | 池 |
|---|---|---|---|
| α, β（缓存混合） | 网格搜索 | 少量 | `compile` |
| τ_c（阈值） | 精确二项 | ≥45 正例/类 | `calibration` |
| ACI 的 γ | DtACI 多专家 | 在线反馈 | — |
| prompt / 清单 | 离线编译 | 验证信号 | `compile` |

⚠️ 这四样**都不能碰 `calibration` 池**。碰了校准就失效，而且失效是静默的。

## 6. 复现

所有训练脚本必须写死 seed 并把以下内容存进 checkpoint 旁的 `run.json`：
git commit、manifest 的 sha256、spec_version、各通道模型名与版本、
`measurements/` 里用到的分层文件名。**A/B/C 是带日期的测量，不是固定标签**，
不记下来就无法复现。

## 7. 工程教训（实测踩到的，非通用建议）

### 7.1 字符串替换必须断言，否则会留下半改状态

批量改代码时用 `s.replace(old, new)` 而不检查是否命中，会在锚点漂移后**静默
失败**。本项目实际发生过：连续三处替换只有一处生效，结果是函数体用了一个
签名里不存在的参数，导入时才 `NameError`。

```python
assert old in s, "锚点未命中"          # 每次 replace 前都加
s = s.replace(old, new, 1)
```

改完必须验证的是**行为**而非"脚本打印了成功"：

```python
import inspect
sig = inspect.signature(fn)
assert "cache_dir" in sig.parameters
```

### 7.2 小数据上 `n_jobs=-1` 反而更慢

实测 n=40、60 棵浅树时：`n_jobs=-1` 996 ms/次，`n_jobs=1` 359 ms/次，
**并行慢 2.8 倍**。派发开销高于计算开销。

`sg2/audit.py` 据此按样本量自动选择（`<500` 用串行）。训练 sentinel 融合头
时同理——特征已预抽取，单步计算极小，别无脑开并行。

### 7.3 先测量再优化

审计最初超时，我的第一反应是置换检验太慢。实测拆开：特征抽取 31s、
置换检验 72s。两者都要治——置换用小森林 + 串行，抽取加缓存。
若只按直觉改置换，会漏掉占三分之一时间的抽取，而抽取是**每次重跑都付**的。
