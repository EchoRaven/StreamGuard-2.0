"""政策语料:结构化、可版本化、可增改。

政策**不是一段文本 blob**,而是一组有 id、有版本、有生命周期的条款。
理由:

1. **引用门**要求 flag 必须指向具体条款(docs/07 §3.2),条款得有稳定 id。
2. **零日政策**(docs/06 §3.6)= 往语料里加条款 + 45 个正例重校准,
   不是一次重训。
3. **out-of-policy**:内容明显有害但**没有任何条款覆盖**时,模型既不该
   flag(引用门会把奖励清零),也不该 clear(它确实有害)。需要第四个动作。
4. **政策变更 = 校准纪元边界**(docs/03 §4)。边界一动,旧阈值失效。

分类体系继承 SafeWatch(ICLR 2025),不自造 —— 见 SAFEWATCH_CATEGORIES。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path


class ClauseStatus(str, Enum):
    DRAFT = "draft"            # 已提议,未生效。**不参与判定**
    ACTIVE = "active"
    DEPRECATED = "deprecated"  # 仍参与判定,但已标记待撤
    RETIRED = "retired"        # 不参与判定,保留以便复现历史判决


# SafeWatch-Bench 的六个不安全类别(ICLR 2025, arXiv:2412.06878)
# 六个类别名**已从 SafeWatch-Bench 的标注实测确认**(1400 条,从
# violate_reason 的 "category Cn (...)" 抽取),不是从论文转述。
SAFEWATCH_CATEGORIES: dict[str, str] = {
    "C1_sexual": "Sexual Content",
    "C2_abuse": "Harassment & Bullying",
    "C3_violence": "Threats, Violence & Harm",
    "C4_misinformation": "False & Deceptive Information",
    "C5_illegal": "Illegal/Regulated Activities",
    "C6_extremism": "Hateful Content & Extremism",
}

# 实测的子任务(38 种)。SafeWatch **自己就区分了明显与隐晦** ——
# evident / subtle / implication 直接对应我们 A/B/C 分层里的感知细微度轴,
# 分层不必从零构造,可用它做初始划分再由 2x2 自动分层校准。
SAFEWATCH_SUBTASKS: dict[str, tuple[str, ...]] = {
    "C1_sexual": ("evident", "subtle", "implication", "hentai"),
    "C2_abuse": ("abuse", "animal_abuse", "campus_bully", "child abuse",
                 "sexual bullying"),
    "C3_violence": ("assault", "fighting", "shooting", "vandalism",
                    "sexual violence", "suicide", "explosion", "terrorism"),
    "C4_misinformation": ("Acting", "AIGC Content", "Alteration",
                          "Misinformation", "Out-of-date"),
    "C5_illegal": ("drugs", "robbery and burglary", "Shoplifting and Stealing",
                   "arsen and vandalism", "military action"),
    "C6_extremism": ("Incitement to Violence", "Incitement to Mental Depression",
                     "Extremely Disturbing Content", "War and Military Actions"),
}


@dataclass
class PolicyClause:
    """一条政策。id 是引用门要引的东西,一旦发布不得复用。"""
    id: str
    category: str
    title: str
    text: str
    status: ClauseStatus = ClauseStatus.ACTIVE
    version: int = 1
    added_on: str = field(default_factory=lambda: date.today().isoformat())
    supersedes: str | None = None          # 修订自哪条
    provenance: str = "manual"             # manual | generated | imported
    examples: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id or " " in self.id:
            raise ValueError(f"条款 id 不能为空或含空格: {self.id!r}")
        if isinstance(self.status, str):
            self.status = ClauseStatus(self.status)
        if self.provenance == "generated" and self.status is ClauseStatus.ACTIVE:
            raise ValueError(
                f"{self.id}: 生成的条款不得直接置为 active。"
                "自动生成并自动启用审核政策是危险的 —— 必须经人工复核,"
                "见 PolicyCorpus.approve()")

    @property
    def is_enforced(self) -> bool:
        """是否参与判定。draft 与 retired 不参与。"""
        return self.status in (ClauseStatus.ACTIVE, ClauseStatus.DEPRECATED)


@dataclass
class PolicyCorpus:
    """一组条款 + 版本。

    渲染进 prompt 的只有 `is_enforced` 的条款。
    """
    clauses: list[PolicyClause] = field(default_factory=list)
    name: str = "default"
    version: str = "1.0.0"

    # ---------- 查询 ----------

    def enforced(self) -> list[PolicyClause]:
        return [c for c in self.clauses if c.is_enforced]

    def by_id(self, cid: str) -> PolicyClause | None:
        return next((c for c in self.clauses if c.id == cid), None)

    def categories(self) -> set[str]:
        return {c.category for c in self.enforced()}

    def covers(self, category: str) -> bool:
        """该类别是否有生效条款。out-of-policy 判定的依据。"""
        return any(c.category == category for c in self.enforced())

    def resolve_citation(self, cid: str | None) -> tuple[PolicyClause | None, str]:
        """把模型给的引用对到真实条款上,并说明对不上的原因。

        实测 Qwen3-VL 的三种错法,都不是归一化能救的:
          "[C1_sexual]" -> 抄了渲染用的方括号(已在 protocol 归一化)
          "C1"          -> 截断成前缀
          "1"           -> 抄了行首序号

        前缀唯一时接受并记为 `prefix`;歧义或不存在则拒绝。宽容要有边界:
        接受歧义前缀等于让模型随便指一条。
        """
        if not cid:
            return None, "empty"
        exact = self.by_id(cid)
        if exact is not None:
            return (exact, "exact") if exact.is_enforced else (None, "not_enforced")
        if cid.isdigit():
            return None, "ordinal"          # 抄了序号,不是 id
        hits = [c for c in self.enforced() if c.id.startswith(cid)]
        if len(hits) == 1:
            return hits[0], "prefix"
        if len(hits) > 1:
            return None, "ambiguous_prefix"
        return None, "unknown"

    def valid_citation(self, cid: str | None) -> bool:
        return self.resolve_citation(cid)[0] is not None

    # ---------- 变更 ----------

    def add(self, clause: PolicyClause) -> "PolicyCorpus":
        if self.by_id(clause.id):
            raise ValueError(f"条款 id 已存在: {clause.id};修订请用 amend()")
        self.clauses.append(clause)
        return self

    def amend(self, cid: str, *, text: str, note: str = "") -> PolicyClause:
        """修订一条。旧条款转 retired,新条款带 supersedes 指回。

        不就地改文本 —— 历史判决引用的是旧 id,就地改会让它们无法复现。
        """
        old = self.by_id(cid)
        if old is None:
            raise ValueError(f"找不到条款 {cid}")
        old.status = ClauseStatus.RETIRED
        new = PolicyClause(
            id=f"{cid}.v{old.version + 1}", category=old.category,
            title=old.title, text=text, version=old.version + 1,
            supersedes=cid, provenance=old.provenance,
            examples=list(old.examples))
        self.clauses.append(new)
        return new

    def approve(self, cid: str, *, reviewer: str) -> PolicyClause:
        """人工复核通过,draft -> active。

        这是**唯一**能让生成条款生效的路径。自动生成并自动启用审核政策
        会让系统在无人知晓的情况下改变判定范围。
        """
        c = self.by_id(cid)
        if c is None:
            raise ValueError(f"找不到条款 {cid}")
        if c.status is not ClauseStatus.DRAFT:
            raise ValueError(f"{cid} 当前是 {c.status.value},只有 draft 可批准")
        if not reviewer:
            raise ValueError("必须记录复核人")
        c.status = ClauseStatus.ACTIVE
        c.provenance = f"{c.provenance}+approved:{reviewer}"
        return c

    # ---------- 渲染 ----------

    def render(self, *, shuffle_seed: int | None = None,
               numbered: bool = False) -> str:
        """渲染成 prompt 里的政策段。

        ⚠️ `shuffle_seed` 用于缓解**位置偏置**。SafeWatch(ICLR 2025)实测
        基线 MLLM 的注意力与政策位置强相关(|ρ|=0.90),它用 PEPE(并行等价
        编码,给各政策块相同的 RoPE)把相关性压到 ≤1%。

        我们包的是冻结模型,改不了 RoPE,只能在**输入侧**缓解:每次换一个
        顺序,把系统性偏置变成噪声,多次采样后近似无偏。

        代价明确:换顺序就换了 prompt 文本 -> KV 前缀缓存失效。
        因此默认 `shuffle_seed=None`(保序、可缓存);要缓解偏置就按**事件**
        而非按 tick 换种子,让缓存在一个事件内仍然有效。
        """
        cs = self.enforced()
        if shuffle_seed is not None:
            cs = list(cs)
            random.Random(shuffle_seed).shuffle(cs)
        # ⚠️ 不要把 id 包在方括号里。实测 Qwen3-VL 会连方括号一起抄进
        # policy_citation,变成 "[C1_sexual]" —— 渲染格式泄漏进了输出。
        # ⚠️ 默认**不编号**。实测 Qwen3-VL 会把行首序号 "1." 当成条款 id
        # 抄进 policy_citation。序号是排版,不该出现在可被引用的位置。
        lines = []
        for i, c in enumerate(cs, 1):
            head = f"{i}. " if numbered else ""
            lines.append(f"{head}id={c.id} | {c.title}: {c.text}")
        return "\n".join(lines)

    def checklist(self, *, shuffle_seed: int | None = None) -> str:
        cs = self.enforced()
        if shuffle_seed is not None:
            cs = list(cs)
            random.Random(shuffle_seed).shuffle(cs)
        return "\n".join(f"- id={c.id} | {c.title}" for c in cs)

    # ---------- 指纹 ----------

    def fingerprint(self) -> str:
        """只对**生效条款**的 id+文本哈希,与顺序无关。

        与顺序无关是有意的:换顺序缓解位置偏置时,不应被当成政策变更而
        触发重新校准 —— 判定范围并没有变。
        """
        blob = json.dumps(
            sorted((c.id, c.text) for c in self.enforced()),
            ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        d = asdict(self)
        for c in d["clauses"]:
            c["status"] = (c["status"].value if hasattr(c["status"], "value")
                           else str(c["status"]))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyCorpus":
        return cls(name=d.get("name", "default"),
                   version=d.get("version", "1.0.0"),
                   clauses=[PolicyClause(**c) for c in d.get("clauses", [])])

    @classmethod
    def load(cls, path: str | Path) -> "PolicyCorpus":
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


def safewatch_corpus() -> PolicyCorpus:
    """SafeWatch 六类的基线语料。

    文本是**占位描述**,不是 SafeWatch 的原文 —— 原文在其 Appendix B.10.1,
    应从正式发布抄录。类别 id 与名称取自论文正文。
    """
    texts = {
        "C1_sexual": "禁止裸露、性行为、性暗示及性剥削相关内容。",
        "C2_abuse": "禁止针对个人或群体的骚扰、霸凌、羞辱与人身攻击。",
        "C3_violence": "禁止暴力行为、伤害、威胁、血腥及自残相关内容。",
        "C4_misinformation": "禁止虚假与欺骗性信息,包括伪造与误导性剪辑。",
        "C5_illegal": "禁止违法及受管制活动,包括毒品、武器交易、盗窃。",
        "C6_extremism": "禁止仇恨言论、极端主义宣传与恐怖主义相关内容。",
    }
    return PolicyCorpus(
        name="safewatch", version="1.0.0",
        clauses=[PolicyClause(id=cid, category=cid, title=title,
                              text=texts[cid], provenance="imported")
                 for cid, title in SAFEWATCH_CATEGORIES.items()])
