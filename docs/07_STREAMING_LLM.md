# 07 · Streaming Video LLM（中间层）

**这一层是什么**：级联的第二级，对已升级窗口做细粒度判别。对应
StreamGuard 1.0 的角色。基座 Qwen3-VL-8B-Instruct（Apache 2.0），ViT 冻结，
LoRA 微调，先 SFT 后 RL。

**这一层不是什么**：它**不**决定全局去哪看。全局分配是 CUSUM + 随机化调度
（§3 的理论建立在此）。把分配交给学出来的策略，延迟界和优雅退化两条都失效。
学习只在**窗口内**发生 —— 与第 6 轮 agentic 工具的边界划分是同一条原则。

## 1. 为什么需要"streaming"，而不是滑窗重跑

朴素做法是每个窗口独立成一次 prompt。问题在于**政策头会被反复重编码**，
而它是最长的那段文本。1.0 附录的 Σctx² 分析说的就是这件事。

真正的流式要求三条：

| 要求 | 含义 | 朴素滑窗为什么不行 |
|---|---|---|
| 因果 | 不能看未来帧 | 滑窗可以做到，但训练数据容易漏掉这条 |
| 有界内存 | KV 不随时长增长 | VideoLLM-Online 式的无界 KV **几分钟内就爆显存** |
| 增量 | 新帧追加而非全量重算 | 滑窗每次重编码政策头 |

## 2. KV 分段结构

参考 StreamingVLM 的组织方式（attention sink + 短视觉窗 + 长文本窗），
加上本项目特有的政策头段：

```
┌──────────┬────────────────┬──────────────────┬─────────────────┐
│ S0 sink  │ S1 政策头       │ S2 事件上下文     │ S3 视觉滑窗      │
│ 4 tokens │ 政策+清单       │ 当前事件的证据    │ 最近 N 秒的帧    │
│ 永不驱逐  │ 永不驱逐(按版本) │ 事件闭合时截断    │ FIFO 驱逐        │
└──────────┴────────────────┴──────────────────┴─────────────────┘
```

**S0 attention sink** —— 前 4 个 token 必须保留。StreamingLLM 的发现：
注意力会大量汇聚到序列最前端的几个 token，驱逐它们会让注意力分布崩坏，
即使这些 token 语义上无意义。**这是最容易漏掉的一条。**

**S1 政策头** —— 每个政策版本只编码一次，之后所有流共享这段 KV 前缀。
政策改版 = 换一个前缀缓存 key。

**S2 事件上下文** —— 当前开放事件累积的证据与中间判断。长文本窗，约 512 token。
**事件闭合 = 截断 S2**。这正是 1.0 的 context reset 在缓存层面的对应物。

**S3 视觉滑窗** —— 最近 N 秒的视觉 token，FIFO 驱逐。N 默认 16s。

### 2.1 为什么这个划分是对的

三段的**生命周期不同**，混在一起就只能整体丢弃：

| 段 | 生命周期 | 丢弃它的代价 |
|---|---|---|
| S1 | 政策版本 | 重编码最长的文本，成本最高 |
| S2 | 一个事件 | 丢失本事件已积累的证据 |
| S3 | N 秒 | 无 —— 本来就该滚动 |

## 3. 流式推理协议

每个 tick（由 CUSUM 决定是否发生）：

```
1. 新帧编码 -> 追加到 S3
2. S3 超长 -> FIFO 驱逐最旧的视觉 token
3. 前向,输出一个结构化动作
4. action == "flag"  -> 写入 S2,事件保持开放
   action == "clear" -> 截断 S2,事件闭合
   action == "hold"  -> 什么都不做,等更多证据
```

### 3.1 三元动作，不是二分类

这是与 1.0 的关键差别。模型输出的不是 safe/unsafe，而是：

| action | 含义 | 什么时候该用 |
|---|---|---|
| `hold` | 证据不足，继续看 | needle 只露了一部分 |
| `flag` | 判定违规 | 证据充分且能引用条款 |
| `clear` | 判定安全，闭合事件 | 已看完足够上下文 |

`hold` 是整个流式设定的核心。没有它，模型被迫在证据不全时二选一，
学到的是"宁可早报"或"宁可晚报"，而**何时提交本身就是要学的东西**。
这也正是 §3 的 quickest detection 在模型层面的体现。

### 3.2 输出契约

```json
{"action": "flag", "category": "C2_sexual",
 "evidence_frames": [63102, 63140],
 "policy_citation": "SW-C2.3",
 "confidence": 0.83}
```

`flag` 必须带 `policy_citation`，否则视为格式错误（SFT 阶段），
或奖励归零（RL 阶段，见 `08_SFT_RL.md` §3.2）。

## 4. 实现分层

```
sg2/stream/
├── context.py    # KV 分段管理器 —— 纯逻辑,可脱离模型测试
├── protocol.py   # 动作/输出契约的解析与校验
└── runner.py     # 绑定具体后端(vLLM / transformers)
```

`context.py` 刻意不依赖任何模型：驱逐策略、段边界、事件闭合这些是
**会出错且出错很隐蔽**的逻辑，必须能在没有 GPU 的情况下详尽测试。

## 5. 与压缩域 sentinel 的分工

| | Sentinel | 本层 |
|---|---|---|
| 覆盖 | 100% 流量 | 仅升级窗口 |
| 输入 | 压缩域+ASR+OCR+弹幕+缩略图 | 全分辨率帧 |
| 输出 | 连续风险分（喂 CUSUM） | 三元动作 + 引用 |
| 是否流式 | 是（逐帧） | 是（KV 分段） |
| 训练 | 融合头，分钟级 | LoRA，天级 |

## 6. 已知风险

| 风险 | 缓解 |
|---|---|
| Turing 无 FA2，长上下文吞吐差 | 本层只在外部 A100+ 上跑；本机用 2B 验证逻辑 |
| KV 分段实现错误难察觉 | `context.py` 脱离模型，做穷举测试 |
| `hold` 学成永远 hold | RL 奖励里漏报惩罚必须压过误报惩罚（§08 §3.3） |
| 政策头缓存与政策版本不同步 | 缓存 key 带政策语料的 sha256 |

## Sources

- [StreamingVLM: Real-Time Infinite Video Understanding](https://www.emergentmind.com/papers/2510.09608)
- [LiveVLM: Streaming-Oriented KV Cache and Retrieval](https://arxiv.org/html/2505.15269)
- [V-Rex: Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval](https://arxiv.org/html/2512.12284)
- [ViCoStream: Streaming VideoLLMs Beyond 100 FPS](https://arxiv.org/html/2606.19849v1)
- [Harnessing Streaming Video in the Wild](https://arxiv.org/html/2606.08615v1)
