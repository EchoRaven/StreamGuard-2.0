"""动作输出的解析与校验。

模型吐的是文本,契约要求是结构化动作。解析必须**严格**:静默接受畸形输出
会让 RL 的格式门形同虚设,也会让评测里混进无法判定的样本。
"""
from __future__ import annotations

import json
import re

from ..models.base import StreamStep

VALID_ACTIONS = ("hold", "flag", "clear")

# 模型常把 JSON 包在 ```json ... ``` 里
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE = re.compile(r"\{.*\}", re.S)


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
        policy_citation=d.get("policy_citation"),
        evidence_frames=tuple(int(f) for f in frames
                              if isinstance(f, (int, float))),
        confidence=d.get("confidence"),
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
