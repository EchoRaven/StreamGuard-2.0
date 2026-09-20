"""StreamingVLM 实现:KV 分段 + 三元动作。

两个后端:

- `mock`   —— 无需权重,用可配置的脚本化行为驱动整条流水线。
- `qwen3vl` —— 真实后端。torch/transformers 惰性导入。

两者共用 `StreamContext` 做 KV 预算管理,共用 `protocol.py` 做输出解析,
所以换后端不改调用方 —— 这正是 registry 存在的理由。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import MidtierConfig
from ..registry import build, register
from ..stream.context import StreamContext
from ..prompts import PromptBuilder
from ..stream.protocol import parse_step
from .base import StreamStep


def _ctx_from(cfg: MidtierConfig) -> StreamContext:
    c = cfg.context
    return StreamContext(sink_tokens=c.sink_tokens,
                         max_policy_tokens=c.max_policy_tokens,
                         max_event_tokens=c.max_event_tokens,
                         max_vision_tokens=c.max_vision_tokens,
                         vision_window_s=c.vision_window_s)


@register("midtier", "mock")
class MockStreamingVLM:
    """脚本化的流式 VLM。

    用途不是模拟模型的智能,而是让**编排逻辑**(KV 驱逐、事件开闭、
    CUSUM 联动、奖励计算)能在没有 GPU 的情况下被完整测试。

    `script` 给定每次 step 返回的原始文本;耗尽后重复最后一条。
    """

    def __init__(self, cfg: MidtierConfig | None = None, *,
                 script: list[str] | None = None,
                 tokens_per_frame: int = 256,
                 tokens_per_step: int = 40):
        self.cfg = cfg or MidtierConfig()
        self.ctx = _ctx_from(self.cfg)
        self.script = list(script or ['{"action":"hold"}'])
        self.tokens_per_frame = tokens_per_frame
        self.tokens_per_step = tokens_per_step
        self._i = 0
        self._policy_text = ""
        self.prompts = PromptBuilder(self.cfg.prompt.build_templates())

    def set_policy(self, policy_text: str, key: str | None = None) -> bool:
        """key=None 时由模板指纹 + 政策内容自动派生。

        ⚠️ 手动传 key 且只反映政策内容的话,改模板会命中旧缓存 ——
        模型照跑,只是在用旧前缀。自动派生把这条堵死。
        """
        self._policy_text = policy_text
        key = key or self.prompts.cache_key(policy_text,
                                            self.cfg.prompt.checklist)
        header = self.prompts.policy_header(policy_text,
                                            self.cfg.prompt.checklist)
        return self.ctx.set_policy(max(1, len(header) // 4), key)

    def ingest(self, frames: np.ndarray, t_s: float) -> None:
        frames = np.atleast_3d(frames)
        if frames.ndim == 3:
            frames = frames[None]
        for f in frames:
            self.ctx.append_frame(self.tokens_per_frame, t_s)

    def step(self) -> StreamStep:
        raw = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        s = parse_step(raw, tokens=self.tokens_per_step)
        if s.action == "flag":
            self.ctx.append_event(self.tokens_per_step, tag="flag")
        return s

    def close_event(self) -> int:
        return self.ctx.close_event()

    def reset(self) -> None:
        self.ctx = _ctx_from(self.cfg)
        self._i = 0


@register("midtier", "qwen3vl")
class Qwen3VLStreaming:
    """Qwen3-VL 流式后端。

    ⚠️ 当前实现每个 tick 重建一次视觉输入(滑窗内的帧),依赖 HF 的
    `past_key_values` 做文本前缀复用。真正的分段 KV 复用需要直接操作
    cache 对象,列为后续优化 —— 先把**正确性**跑通,再谈吞吐。
    `ctx` 的预算仍然生效,决定滑窗里保留哪些帧。
    """

    def __init__(self, cfg: MidtierConfig | None = None):
        self.cfg = cfg or MidtierConfig()
        self.ctx = _ctx_from(self.cfg)
        self._model = None
        self._proc = None
        self._policy_text = ""
        self._frames: list[tuple[float, np.ndarray]] = []
        self.prompts = PromptBuilder(self.cfg.prompt.build_templates())
        self._evidence: list[str] = []

    # ---------- 惰性加载 ----------

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor
        dtype = getattr(torch, self.cfg.dtype)
        if dtype is torch.bfloat16 and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 8:
                raise RuntimeError(
                    f"sm_{cap[0]}{cap[1]} 不支持 bfloat16。"
                    "把 midtier.dtype 改成 float16(见 configs/turing_local.yaml)")
        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls
        self._proc = AutoProcessor.from_pretrained(self.cfg.model_id)

        kw: dict = {"torch_dtype": dtype,
                    "attn_implementation": self.cfg.attn_impl}

        if self.cfg.quantization:
            from transformers import BitsAndBytesConfig
            if self.cfg.quantization == "nf4":
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True)
            elif self.cfg.quantization == "int8":
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"未知量化方式 {self.cfg.quantization}")

        if self.cfg.device == "auto":
            # accelerate 按层分片。max_memory 要给每张卡留出激活的余量,
            # 填满权重会在前向时 OOM。
            kw["device_map"] = "auto"
            if self.cfg.max_memory_per_gpu:
                n = torch.cuda.device_count()
                kw["max_memory"] = {i: self.cfg.max_memory_per_gpu
                                    for i in range(n)}
            self._model = Cls.from_pretrained(self.cfg.model_id, **kw).eval()
        elif self.cfg.quantization:
            kw["device_map"] = {"": self.cfg.device}
            self._model = Cls.from_pretrained(self.cfg.model_id, **kw).eval()
        else:
            self._model = Cls.from_pretrained(
                self.cfg.model_id, **kw).to(self.cfg.device).eval()

    # ---------- 协议 ----------

    def set_policy(self, policy_text: str, key: str | None = None) -> bool:
        self._policy_text = policy_text
        self._lazy()
        key = key or self.prompts.cache_key(policy_text,
                                            self.cfg.prompt.checklist)
        header = self.prompts.policy_header(policy_text,
                                            self.cfg.prompt.checklist)
        n_tok = len(self._proc.tokenizer(header)["input_ids"])
        return self.ctx.set_policy(n_tok, key)

    def ingest(self, frames: np.ndarray, t_s: float) -> None:
        frames = np.atleast_3d(frames)
        if frames.ndim == 3:
            frames = frames[None]
        for f in frames:
            evicted = self.ctx.append_frame(
                self._tokens_per_frame(), t_s)
            self._frames.append((t_s, f))
            for _ in evicted:
                if self._frames:
                    self._frames.pop(0)

    def _tokens_per_frame(self) -> int:
        """粗估每帧视觉 token 数。真实值随分辨率变化。"""
        return 256

    def step(self) -> StreamStep:
        import torch
        from PIL import Image
        self._lazy()
        if not self._frames:
            return StreamStep(action="hold", raw='{"action":"hold"}')

        imgs = [Image.fromarray((f * 255).astype(np.uint8))
                for _, f in self._frames]
        prompt = self.prompts.build(
            policy_text=self._policy_text, n_frames=len(imgs),
            checklist=self.cfg.prompt.checklist,
            timestamps=[t for t, _ in self._frames],
            evidence=self._evidence,
            perception_only=self.cfg.prompt.perception_only,
            include_sink=True)
        msgs = [{"role": "user",
                 "content": [{"type": "image"} for _ in imgs]
                            + [{"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=imgs, return_tensors="pt")
        # 分片时输入要送到**第一层所在的卡**,不能用 cfg.device("auto" 不是设备名)
        dev = getattr(self._model, "device", None) or self.cfg.device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=160,
                                       do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = self._proc.tokenizer.decode(gen, skip_special_tokens=True)
        s = parse_step(raw, tokens=int(gen.shape[0]))
        if s.action == "flag":
            self.ctx.append_event(s.tokens, tag="flag")
            self._evidence.append(f"t={self._frames[-1][0]:.1f}s {s.category}")
        return s

    def close_event(self) -> int:
        self._evidence.clear()
        return self.ctx.close_event()

    def reset(self) -> None:
        self.ctx = _ctx_from(self.cfg)
        self._frames.clear()
        self._evidence.clear()


def build_midtier(cfg: MidtierConfig, **kw):
    return build("midtier", cfg.name, cfg, **kw)
