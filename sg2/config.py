"""配置系统。

一切可换的东西都走配置:组件按**名字**选实现(见 sg2/registry.py),
参数用嵌套 dataclass 表达,加载时立即校验。

两条原则:

1. **快速失败。** 配置错误要在加载时报,不是跑到第三小时才崩。所有
   `__post_init__` 都做结构性校验。
2. **结构性约束写进类型。** 例如 `beta_fa < gamma_hit` 不是调参偏好,
   而是"否则最优策略退化为永不 flag"(docs/08_SFT_RL.md §3.3),
   所以它是构造时的错误而非文档里的一句提醒。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 工具


def _resolve_types(cls) -> dict:
    """解析 dataclass 字段的真实类型。

    ⚠️ `from __future__ import annotations` 下 `field.type` 是**字符串**,
    直接 `is_dataclass(f.type)` 永远为假 —— 嵌套还原会静默退化成 dict,
    然后在第一次属性访问时才报 AttributeError。必须用 get_type_hints。
    """
    import typing
    return typing.get_type_hints(cls)


def _coerce(cls, data: Any):
    """把嵌套 dict 还原成嵌套 dataclass。"""
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: 未知配置项 {sorted(unknown)}。"
            f"可用: {sorted(known)}")
    hints = _resolve_types(cls)
    kw = {}
    for name in known:
        if name not in data:
            continue
        t = hints.get(name)
        kw[name] = _coerce(t, data[name]) if is_dataclass(t) else data[name]
    return cls(**kw)


# ---------------------------------------------------------------- 各段


@dataclass
class EncoderConfig:
    """视觉编码器。冻结,不训练。"""
    name: str = "siglip2"                 # registry 里的名字
    model_id: str = "google/siglip2-so400m-patch14-384"
    image_size: int = 384
    dtype: str = "float16"
    device: str = "cuda:0"
    batch_size: int = 32


@dataclass
class ChannelConfig:
    """sentinel 的各廉价通道。关掉某路只需置 false。"""
    codec: bool = True                    # 压缩域:不解码即可得
    vision: bool = True
    asr: bool = False                     # 需 faster-whisper
    ocr: bool = False                     # 需 rapidocr
    chat: bool = False                    # 仅直播有
    # rolling attention probe(SafeLens S1 的机制,见 sg2/probe.py)。
    # ⚠️ 默认关:它需要一个**已训练**的探针对象,没有就该报错而不是静默跳过。
    rolling_probe: bool = False


@dataclass
class SentinelConfig:
    name: str = "multichannel"
    sample_fps: float = 0.5
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fusion_ckpt: str | None = None
    dedup_embedding_delta: float = 0.02   # 感知去重阈值
    max_decode_fps_multiplier: float = 3.0  # 成本放大攻击的硬上限
    probe_window: int = 10                # rolling 窗宽(单位:tick,不是秒)

    def __post_init__(self):
        if self.sample_fps <= 0:
            raise ValueError("sample_fps 必须为正")
        if not any(asdict(self.channels).values()):
            raise ValueError("至少要启用一个通道")
        if self.probe_window < 1:
            raise ValueError("probe_window 必须 >= 1")
        if self.channels.rolling_probe and not self.channels.vision:
            # 探针吃的是视觉编码器的嵌入,vision 关掉它就没有输入
            raise ValueError(
                "channels.rolling_probe 需要 channels.vision 开启 —— "
                "探针读的是视觉编码器的嵌入")


@dataclass
class CusumConfig:
    """升级控制器。全局分配用统计,不用学习 —— 否则延迟界失效。"""
    h: float = 5.0                        # 告警阈值, h ≈ log(平均误报间隔)
    drift: float = 0.1
    jitter_s: float = 0.4                 # 随机化调度,对抗按采样表定时的对手
    burst_fps: float = 4.0                # 告警后的突发采样率
    rewind_s: float = 30.0                # 告警后回溯 ring buffer 的长度

    def __post_init__(self):
        if self.h <= 0:
            raise ValueError("h 必须为正")
        if self.jitter_s < 0:
            raise ValueError("jitter_s 不能为负")


@dataclass
class CoverageConfig:
    """保底随机覆盖。一个机制买三样:优雅退化 + ACI 无偏信号 + minimax。"""
    rho: float = 0.02                     # 无条件送 analyst 的流量比例
    audit_rate: float = 0.01              # 未升级流量的人工抽检率

    def __post_init__(self):
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho 必须在 [0,1]")
        if self.rho == 0.0:
            raise ValueError(
                "rho=0 会让优雅退化定理失效(docs/06 §6.1):"
                "sentinel 全瞎的类别上召回将为 0。至少给一个很小的正值。")


@dataclass
class StreamContextConfig:
    """KV 分段预算。见 docs/07_STREAMING_LLM.md §2。"""
    sink_tokens: int = 4                  # attention sink,永不驱逐
    max_policy_tokens: int = 2048
    max_event_tokens: int = 512
    max_vision_tokens: int = 4096
    vision_window_s: float = 16.0

    def __post_init__(self):
        if self.sink_tokens < 1:
            raise ValueError(
                "sink_tokens 至少为 1。驱逐 attention sink 会让注意力"
                "分布崩坏 —— 即使这些 token 语义上无意义。")


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    freeze_vit: bool = True               # 视觉塔是能力来源,小数据微调会退化


@dataclass
class PolicyConfig:
    """政策语料。

    `shuffle_scope` 控制位置偏置的缓解粒度。SafeWatch(ICLR 2025)实测基线
    MLLM 的注意力与政策位置强相关(|ρ|=0.90),它用 PEPE 改 RoPE 解决;
    我们包冻结模型改不了 RoPE,只能在输入侧换顺序。

    代价明确:换顺序 = 换 prompt 文本 = KV 前缀缓存失效。因此:
      never   —— 保序,缓存最优,但承受位置偏置
      event   —— 每个事件换一次,事件内缓存仍有效(**推荐**)
      tick    —— 每次都换,偏置最小但缓存全失效
    """
    corpus_path: str | None = None        # None -> safewatch 基线
    shuffle_scope: str = "event"          # never | event | tick
    allow_uncovered: bool = True          # 是否启用第四动作
    gap_min_cases: int = 5                # 少于此数不提议补条款

    def __post_init__(self):
        if self.shuffle_scope not in ("never", "event", "tick"):
            raise ValueError(f"shuffle_scope 非法: {self.shuffle_scope}")

    def build_corpus(self):
        from .policy import PolicyCorpus, safewatch_corpus
        return (PolicyCorpus.load(self.corpus_path) if self.corpus_path
                else safewatch_corpus())


@dataclass
class PromptConfig:
    """Prompt 模板。四段 KV 的文本内容与拼接全部可配。

    `preset` 选内置预设,`path` 指向外部 YAML(优先级更高),
    `overrides` 做逐字段覆盖。三者叠加,便于消融实验只改一段。
    """
    preset: str = "default"
    path: str | None = None
    overrides: dict = field(default_factory=dict)
    checklist: str = ""
    perception_only: bool = False     # 感知-判断解耦,见 docs/05 §5.4

    def build_templates(self):
        from .prompts import PromptTemplates, load_preset
        t = (PromptTemplates.load(self.path) if self.path
             else load_preset(self.preset))
        if self.overrides:
            d = t.to_dict(); d.update(self.overrides)
            t = PromptTemplates.from_dict(d)
        return t


@dataclass
class MidtierConfig:
    """中间层流式 VLM。

    单卡放不下时两条路(本机 4x11GB 实测都可行):
      quantization="nf4"  —— 8B 压到约 6GB,单卡即可,但 Turing 的 int4
                             kernel 支持一般,吞吐会掉
      device="auto"       —— accelerate 按层分片到多卡,保持 fp16 精度
    """
    name: str = "qwen3vl"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    dtype: str = "bfloat16"
    attn_impl: str = "flash_attn"
    max_length: int = 8192
    device: str = "cuda:0"          # "auto" -> 按 max_memory 分片到多卡
    quantization: str | None = None  # None | "nf4" | "int8"
    max_memory_per_gpu: str | None = None  # 如 "10GiB";device="auto" 时生效
    lora: LoraConfig = field(default_factory=LoraConfig)
    context: StreamContextConfig = field(default_factory=StreamContextConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class AnalystConfig:
    name: str = "openai"
    model_id: str = "gpt-5"
    max_turns: int = 6
    max_tool_calls: int = 10
    budget_tokens: int = 40_000
    timeout_s: int = 45
    perception_only: bool = True          # 感知-判断解耦,见 docs/05 §5.4


@dataclass
class CalibrationConfig:
    target_recall: float = 0.95
    delta: float = 0.10
    # ACI 步长。实测 0.02 在 3%<->8% 漂移下会让 alpha_t 摆满量程;
    # 0.002 才稳。经验法则:约取 1/(一个漂移周期内的反馈条数)。
    aci_gamma: float = 0.002
    min_release_events: int = 3           # 两次 case library 发布的最小间隔 = k/gamma

    @property
    def n_min(self) -> int:
        """达标所需的最小校准正例数(精确二项,零失败)。"""
        import math
        return math.ceil(math.log(self.delta) / math.log(self.target_recall))


@dataclass
class RewardConfigC:
    """RL 奖励。与 sg2.train.reward.RewardConfig 同构,此处仅承载配置。"""
    gamma_hit: float = 1.0
    alpha_delay: float = 0.5
    beta_fa: float = 0.3
    mu_token: float = 0.05
    token_ref: float = 2000.0
    max_breakeven_precision: float = 0.3

    def __post_init__(self):
        if self.beta_fa >= self.gamma_hit:
            raise ValueError(
                f"beta_fa({self.beta_fa}) >= gamma_hit({self.gamma_hit}):"
                "最优策略会退化为永不 flag。见 docs/08_SFT_RL.md §3.3")


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    ring_buffer_s: float = 300.0
    seed: int = 0
    cache_dir: str | None = None


@dataclass
class SG2Config:
    """顶层配置。"""
    sentinel: SentinelConfig = field(default_factory=SentinelConfig)
    cusum: CusumConfig = field(default_factory=CusumConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    midtier: MidtierConfig = field(default_factory=MidtierConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    reward: RewardConfigC = field(default_factory=RewardConfigC)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        try:
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True,
                                           sort_keys=False))
        except ImportError:
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                       indent=2))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "SG2Config":
        return _coerce(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SG2Config":
        text = Path(path).read_text()
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            d = json.loads(text)
        return cls.from_dict(d or {})

    # ---------- 覆盖 ----------

    def override(self, dotted: str, value: Any) -> "SG2Config":
        """按点号路径覆盖,如 `midtier.lora.rank=32`。

        命令行传参与消融实验都走这里 —— 改配置不改代码。
        """
        obj: Any = self
        parts = dotted.split(".")
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise ValueError(f"配置里没有 `{p}`(路径 {dotted})")
            obj = getattr(obj, p)
        last = parts[-1]
        if not hasattr(obj, last):
            raise ValueError(f"配置里没有 `{last}`(路径 {dotted})")
        cur = getattr(obj, last)
        if cur is not None and not isinstance(value, type(cur)):
            try:
                value = type(cur)(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{dotted}: 无法把 {value!r} 转成 {type(cur).__name__}") from e
        setattr(obj, last, value)
        # 重新触发该段的校验
        if is_dataclass(obj) and hasattr(obj, "__post_init__"):
            obj.__post_init__()
        return self

    # ---------- 硬件校验 ----------

    def validate_for_device(self, capability: tuple[int, int] | None = None
                            ) -> list[str]:
        """按实际 GPU 能力检查配置,返回问题列表(空 = 无问题)。

        Turing(sm_75)没有 bf16 也没有 FlashAttention-2。这两条不检查的话
        会在跑起来之后才以 NaN 或 ImportError 的形式暴露。
        """
        if capability is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    return ["CUDA 不可用;仅能跑 mock 组件"]
                capability = torch.cuda.get_device_capability(0)
            except ImportError:
                return ["未安装 torch;仅能跑 mock 组件"]

        problems = []
        sm = capability[0] * 10 + capability[1]
        if sm < 80:
            if self.midtier.dtype == "bfloat16":
                problems.append(
                    f"sm_{sm} 不支持 bfloat16 -> midtier.dtype 应设为 float16")
            if self.sentinel.encoder.dtype == "bfloat16":
                problems.append(
                    f"sm_{sm} 不支持 bfloat16 -> sentinel.encoder.dtype 应设为 float16")
            if "flash" in self.midtier.attn_impl:
                problems.append(
                    f"sm_{sm} 不支持 FlashAttention-2 -> midtier.attn_impl 应设为 sdpa")
        return problems
