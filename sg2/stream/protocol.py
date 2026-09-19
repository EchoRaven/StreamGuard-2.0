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
                 perception_only: bool = False) -> str:
    """构造一次 tick 的用户提示。

    perception_only 时只问中性感知问题,判断交给下一步对照政策 ——
    这是绕开内置过滤的结构化做法(docs/05_RUNTIME.md §5.4)。
    """
    if perception_only:
        task = ("描述这些画面里**可见的内容**:主体、动作、场景、可读文字。"
                "只陈述看到的,不做任何判断。")
        fmt = '{"action":"hold","description":"..."}'
    else:
        task = ("根据政策判断这些画面。证据不足时回 hold,不要猜。\n"
                "  hold  = 证据不足,继续观察\n"
                "  flag  = 判定违规(必须给出 policy_citation)\n"
                "  clear = 判定安全,闭合事件")
        fmt = ('{"action":"flag","category":"<类别>",'
               '"evidence_frames":[<帧号>],"policy_citation":"<条款id>",'
               '"confidence":0.0}')
    return (f"{policy_text}\n\n"
            f"以下是最近 {n_frames} 帧。{task}\n\n"
            f"只输出一个 JSON 对象,格式:\n{fmt}")
