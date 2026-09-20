"""Prompt 模板系统。

四段 KV(sink / 政策头 / 事件上下文 / 视觉滑窗)各自的文本内容与拼接方式
全部可配置。这不是美观问题:

1. **离线 prompt 编译**(docs/08_SFT_RL.md §4.3)要求 prompt 是**数据**
   而非代码 —— `pool="compile"` 存在的意义就是在它上面搜模板。
2. **模板变动必须使 KV 前缀缓存失效。** 改了政策头模板却沿用同一个
   cache key,会拿到过期的前缀 —— 模型照跑,只是在用旧政策。
   因此 `cache_key()` 把模板内容一起哈希进去。

占位符用 `$name` / `${name}`(string.Template),**不用 str.format** ——
输出格式说明里全是 `{` `}`,用 .format 会当成占位符炸掉。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from string import Template

# ---------------------------------------------------------------- 默认模板

DEFAULT_SINK = "你是流式视频内容审核系统。"

DEFAULT_POLICY_HEADER = """${policy_text}

判定清单:
${checklist}"""

DEFAULT_EVENT_RECAP = """已积累的证据(本事件):
${evidence}"""

DEFAULT_FRAME_MARKER = "[帧 ${index} @ ${t_s}s]"

DEFAULT_TASK_JUDGE = """以下是最近 ${n_frames} 帧。根据政策判断。证据不足时回 hold,不要猜。
  hold  = 证据不足,继续观察
  flag  = 判定违规(必须给出 policy_citation)
  clear = 判定安全,闭合事件"""

DEFAULT_TASK_PERCEPTION = """以下是最近 ${n_frames} 帧。描述其中**可见的内容**:
主体、动作、场景、可读文字。只陈述看到的,不做任何判断。"""

DEFAULT_FORMAT_JUDGE = """只输出一个 JSON 对象:
{"action":"flag","category":"<类别>","evidence_frames":[<帧号>],"policy_citation":"<条款id>","confidence":0.0}"""

DEFAULT_FORMAT_PERCEPTION = """只输出一个 JSON 对象:
{"action":"hold","description":"<你看到的内容>"}"""

DEFAULT_EXEMPLAR = """示例 ${index}:
输入: ${input}
输出: ${output}"""


class TemplateError(ValueError):
    pass


@dataclass
class PromptTemplates:
    """四段 KV 的文本模板 + 任务与格式说明。

    每个字段都是 `string.Template` 语法。改任何一个都会改变 `cache_key()`,
    从而使 KV 前缀缓存失效 —— 这是**有意的**。
    """
    sink: str = DEFAULT_SINK
    policy_header: str = DEFAULT_POLICY_HEADER
    event_recap: str = DEFAULT_EVENT_RECAP
    frame_marker: str = DEFAULT_FRAME_MARKER
    task_judge: str = DEFAULT_TASK_JUDGE
    task_perception: str = DEFAULT_TASK_PERCEPTION
    format_judge: str = DEFAULT_FORMAT_JUDGE
    format_perception: str = DEFAULT_FORMAT_PERCEPTION
    exemplar: str = DEFAULT_EXEMPLAR

    joiner: str = "\n\n"                 # 段间分隔
    exemplar_joiner: str = "\n\n"
    name: str = "default"

    # 各模板**必须**出现的占位符 —— 缺了就是配置错,应在加载时报
    REQUIRED = {
        "policy_header": ("policy_text",),
        "task_judge": ("n_frames",),
        "task_perception": ("n_frames",),
        "event_recap": ("evidence",),
        "frame_marker": ("index",),
        "exemplar": ("input", "output"),
    }

    def __post_init__(self):
        for fname, required in self.REQUIRED.items():
            body = getattr(self, fname)
            missing = [p for p in required if f"${p}" not in body
                       and f"${{{p}}}" not in body]
            if missing:
                raise TemplateError(
                    f"模板 `{fname}` 缺少必需占位符 {missing};"
                    f"当前内容: {body[:60]!r}")

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "PromptTemplates":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise TemplateError(f"未知模板字段 {sorted(unknown)};可用 {sorted(known)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplates":
        text = Path(path).read_text()
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            d = json.loads(text)
        return cls.from_dict(d or {})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        try:
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True,
                                           sort_keys=False, width=100))
        except ImportError:
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                       indent=2))
        return path

    # ---------- 指纹 ----------

    def fingerprint(self) -> str:
        """模板内容的哈希。模板一变,KV 前缀缓存必须失效。"""
        blob = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _render(tpl: str, **kw) -> str:
    try:
        return Template(tpl).substitute(**kw)
    except KeyError as e:
        raise TemplateError(f"模板缺少变量 {e}") from e
    except ValueError as e:
        raise TemplateError(
            f"模板语法错误({e})。占位符用 $name / ${{name}};"
            "字面量 $ 要写成 $$") from e


@dataclass
class PromptBuilder:
    """按模板装配四段内容。

    装配逻辑本身也很薄 —— 真正的可配置性来自模板,不来自这里的 if。
    """
    templates: PromptTemplates = field(default_factory=PromptTemplates)

    # ---------- 分段 ----------

    def sink(self) -> str:
        return self.templates.sink

    def policy_header(self, policy_text: str, checklist: str = "") -> str:
        return _render(self.templates.policy_header,
                       policy_text=policy_text, checklist=checklist)

    def event_recap(self, evidence: list[str]) -> str:
        if not evidence:
            return ""
        return _render(self.templates.event_recap,
                       evidence="\n".join(f"- {e}" for e in evidence))

    def frame_markers(self, timestamps: list[float]) -> list[str]:
        return [_render(self.templates.frame_marker, index=i,
                        t_s=f"{t:.2f}")
                for i, t in enumerate(timestamps)]

    def exemplars(self, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        return self.templates.exemplar_joiner.join(
            _render(self.templates.exemplar, index=i + 1, input=inp,
                    output=out)
            for i, (inp, out) in enumerate(items))

    # ---------- 组装 ----------

    def build(self, *, policy_text: str, n_frames: int,
              checklist: str = "", evidence: list[str] | None = None,
              exemplars: list[tuple[str, str]] | None = None,
              timestamps: list[float] | None = None,
              perception_only: bool = False,
              include_sink: bool = True) -> str:
        """按 S0→S1→(示例)→S2→S3→任务→格式 的顺序装配。"""
        t = self.templates
        task = _render(t.task_perception if perception_only else t.task_judge,
                       n_frames=n_frames)
        fmt = t.format_perception if perception_only else t.format_judge

        parts: list[str] = []
        if include_sink and t.sink:
            parts.append(t.sink)
        parts.append(self.policy_header(policy_text, checklist))
        if exemplars:
            parts.append(self.exemplars(exemplars))
        recap = self.event_recap(evidence or [])
        if recap:
            parts.append(recap)
        if timestamps:
            parts.append(" ".join(self.frame_markers(timestamps)))
        parts.append(task)
        parts.append(fmt)
        return t.joiner.join(p for p in parts if p)

    # ---------- 缓存键 ----------

    def cache_key(self, policy_text: str, checklist: str = "") -> str:
        """KV 前缀缓存的 key。

        ⚠️ 必须同时包含**政策内容**与**模板指纹**。只用政策内容做 key 的话,
        改了模板会命中旧缓存 —— 模型照跑,只是在用旧的前缀。
        """
        h = hashlib.sha256()
        h.update(self.templates.fingerprint().encode())
        h.update(b"\x00")
        h.update(policy_text.encode())
        h.update(b"\x00")
        h.update(checklist.encode())
        return h.hexdigest()[:24]


# ---------------------------------------------------------------- 预设

PRESETS: dict[str, dict] = {
    "default": {},
    "minimal": {
        "name": "minimal",
        "sink": "",
        "policy_header": "${policy_text}",
        "task_judge": "看这 ${n_frames} 帧。回 hold/flag/clear。",
        "format_judge": '{"action":"...","policy_citation":"..."}',
    },
    # ⚠️ 用此预设时须同时设 PromptConfig(perception_only=True),
    # 否则模板问的是感知、格式说明却要求判决 —— 实测模型会照样输出 flag。
    "perception_first": {
        "name": "perception_first",
        "sink": "你是视觉描述助手。",
        "task_perception": ("描述这 ${n_frames} 帧里可见的:主体、动作、"
                            "场景、文字。不做判断。"),
    },
}


def load_preset(name: str) -> PromptTemplates:
    if name not in PRESETS:
        raise TemplateError(f"未知预设 {name!r};可用 {sorted(PRESETS)}")
    return PromptTemplates.from_dict(PRESETS[name])
