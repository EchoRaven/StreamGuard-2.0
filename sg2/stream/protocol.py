"""动作输出的解析与校验。

模型吐的是文本,契约要求是结构化动作。解析必须**严格**:静默接受畸形输出
会让 RL 的格式门形同虚设,也会让评测里混进无法判定的样本。
"""
from __future__ import annotations

import json
import re

from ..contracts import StreamStep

# uncovered = 看起来有害但现行政策未覆盖。见 sg2/outofpolicy.py ——
# 没有它,"有害但无条款可引"的内容没有合法输出,模型只能编造引用或放行。
VALID_ACTIONS = ("hold", "flag", "clear", "uncovered")

# 模型常把 JSON 包在 ```json ... ``` 里
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE = re.compile(r"\{.*\}", re.S)

# 模型常把条款 id 连同渲染时的方括号一起抄回来 —— 实测 Qwen3-VL 输出
# "[C1_sexual]" 而非 "C1_sexual",导致 valid_citation() 全部判非法。
# 归一化而非放宽校验:方括号是渲染格式,不是 id 的一部分。
_CITE_JUNK = "[]()<>「」 \t\n\"'`"
_NULLISH = {"", "none", "null", "n/a", "na", "无", "不适用"}
# 渲染时写的键名,模型会连键名一起抄
_CITE_PREFIXES = ("id=", "id:", "policy_id=", "条款id=", "条款=")


def normalize_citation(raw) -> str | None:
    """把模型抄回来的渲染格式剥掉,还原成纯 id。

    实测 Qwen3-VL 的四种抄法(全是渲染格式泄漏,不是模型不听话):
        "[C1_sexual]"      渲染用方括号包 id
        "1"                行首序号
        "C1"               截断成前缀(交给 resolve_citation 处理)
        "id=T1_testpattern" 渲染写成 "id=xxx",连键名一起抄
    """
    if raw is None:
        return None
    c = str(raw).strip(_CITE_JUNK).strip()
    for pfx in _CITE_PREFIXES:
        if c.lower().startswith(pfx):
            c = c[len(pfx):].strip(_CITE_JUNK).strip()
            break
    return None if c.lower() in _NULLISH else c




class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """从模型输出里抽出 JSON 对象。

    只做**定位**不做修补:不补缺失的括号、不猜字段。修补会让格式门失效。
    """
    if not text or not text.strip():
        raise ParseError("空输出")
    for pat in (_FENCE, _BARE):
        m = pat.search(text)
        if m:
            try:
                d = json.loads(m.group(1) if pat is _FENCE else m.group(0))
            except json.JSONDecodeError as e:
                raise ParseError(f"JSON 解析失败: {e}") from e
            if not isinstance(d, dict):
                raise ParseError(f"顶层不是对象,而是 {type(d).__name__}")
            return d
    raise ParseError("输出里找不到 JSON 对象")


def parse_step(text: str, *, tokens: int = 0) -> StreamStep:
    """解析成 StreamStep。非法时返回 action='invalid' 而非抛异常。

    返回而非抛出,是因为调用方需要**统计**非法率(它是 SFT 的质量指标,
    见 docs/08_SFT_RL.md §4),抛异常会让这个统计写起来很别扭。
    """
    try:
        d = extract_json(text)
    except ParseError:
        return StreamStep(action="invalid", raw=text, tokens=tokens)

    action = d.get("action")
    if action not in VALID_ACTIONS:
        return StreamStep(action="invalid", raw=text, tokens=tokens)

    frames = d.get("evidence_frames") or []
    if not isinstance(frames, (list, tuple)):
        frames = []

    step = StreamStep(
        action=action, raw=text,
        category=d.get("category"),
        policy_citation=normalize_citation(d.get("policy_citation")),
        evidence_frames=tuple(int(f) for f in frames
                              if isinstance(f, (int, float))),
        confidence=d.get("confidence"),
        # ⚠️ uncovered 专用的两个字段。漏解析的话缺口追踪会把所有案例
        # 归到 "<未分类>",于是永远聚不出一个可行动的缺口 —— 而 StreamStep
        # 里明明有这两个字段,静默丢失。
        description=d.get("description"),
        suggested_category=d.get("suggested_category"),
        tokens=tokens)
    # flag 必须带引用 —— 契约见 docs/07 §3.2
    if step.action == "flag" and not step.policy_citation:
        return StreamStep(action="invalid", raw=text, tokens=tokens)
    return step


def build_prompt(policy_text: str, *, n_frames: int,
                 perception_only: bool = False, builder=None,
                 **kw) -> str:
    """构造一次 tick 的用户提示。

    实际装配交给 `sg2.prompts.PromptBuilder` —— 模板是**数据**,
    离线 prompt 编译(docs/08 §4.3)才可能做。
    """
    if builder is None:
        from ..prompts import PromptBuilder
        builder = PromptBuilder()
    return builder.build(policy_text=policy_text, n_frames=n_frames,
                         perception_only=perception_only, **kw)
