"""政策动物园:多来源政策混合。

**"更多 policy 更好"这个假设值得测,不该假定。** 有效的轴是
**schema 多样性**而不是条款数量:

  - 同一来源加到 50 条 —— 措辞风格、粒度、编号习惯全一样,
    模型仍可能背住"这个风格里 Cn 就是某类",只是换了个更大的表。
  - 不同来源各 6 条 —— 逼模型去读条款**内容**,因为别的线索都不稳定。

而且"更多"是**有代价的**:条款越多 prompt 越长,直接吃 KV 预算,
并加重位置偏置(SafeWatch 实测 |ρ|=0.90,它用 PEPE 解决而我们包的是
冻结模型,只能靠换顺序缓解)。所以这是个需要实测的权衡。

`schema_diversity_ablation()` 生成对照组用于回答:到底是条款数有用,
还是来源多样性有用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .policy import ClauseStatus, PolicyClause, PolicyCorpus


@dataclass(frozen=True)
class PolicySchema:
    """一种政策"风格"。

    真实平台的社区准则在**粒度、措辞、编号习惯**上差别很大,
    这些差别正是模型可能记住的东西。
    """
    name: str
    id_style: str                # "Cn_slug" | "SEC-n.m" | "n" | "slug"
    voice: str                   # "prohibitive" | "descriptive" | "conditional"
    granularity: str             # "coarse" | "fine"
    clauses: tuple[tuple[str, str], ...]   # (title, text)


# 这些是**风格样板**,不是任何平台的真实条文。真实实验应换成各平台
# 公开的社区准则原文,并注明出处与抓取日期。
SCHEMAS: tuple[PolicySchema, ...] = (
    PolicySchema(
        name="safewatch_like", id_style="Cn_slug", voice="prohibitive",
        granularity="coarse",
        clauses=(("Sexual Content", "禁止裸露、性行为及性暗示内容。"),
                 ("Violence & Harm", "禁止暴力行为、伤害与血腥画面。"),
                 ("Harassment", "禁止针对个人或群体的骚扰与霸凌。"))),
    PolicySchema(
        name="legalistic", id_style="SEC-n.m", voice="conditional",
        granularity="fine",
        clauses=(("Depiction of Nudity",
                  "若画面包含未经遮蔽的生殖器、臀部或女性胸部,则构成违规。"),
                 ("Depiction of Physical Injury",
                  "若画面包含开放性伤口、大量血液或肢体损毁,则构成违规。"),
                 ("Targeted Abuse",
                  "若言语或字幕针对特定个人实施贬损,则构成违规。"))),
    PolicySchema(
        name="descriptive", id_style="slug", voice="descriptive",
        granularity="coarse",
        clauses=(("adult-content", "本平台不接受成人向的性相关影像。"),
                 ("graphic-violence", "本平台不接受写实的暴力与伤害影像。"),
                 ("bullying", "本平台不接受针对他人的羞辱性内容。"))),
    PolicySchema(
        name="numbered_terse", id_style="n", voice="prohibitive",
        granularity="fine",
        clauses=(("色情", "含性行为、裸露。"),
                 ("暴力", "含打斗、流血、武器伤人。"),
                 ("欺凌", "含辱骂、羞辱、人身攻击。"))),
)


def _make_id(style: str, i: int, title: str, rng: random.Random) -> str:
    slug = "".join(ch for ch in title.lower() if ch.isalnum())[:10] or "c"
    return {"Cn_slug": f"C{i}_{slug}",
            "SEC-n.m": f"SEC-{i}.{rng.randint(1, 4)}",
            "n": str(i),
            "slug": slug}.get(style, f"C{i}_{slug}")


def build_from_schema(schema: PolicySchema, *,
                      rng: random.Random | None = None) -> PolicyCorpus:
    rng = rng or random.Random(0)
    return PolicyCorpus(
        name=schema.name, version="1.0.0",
        clauses=[PolicyClause(
            id=_make_id(schema.id_style, i, title, rng),
            category=f"{schema.name}_{i}", title=title, text=text,
            status=ClauseStatus.ACTIVE, provenance=f"schema:{schema.name}")
            for i, (title, text) in enumerate(schema.clauses, 1)])


@dataclass
class AblationArm:
    """消融的一组。"""
    label: str
    n_schemas: int
    n_clauses: int
    corpora: list[PolicyCorpus] = field(default_factory=list)

    @property
    def total_clauses(self) -> int:
        return sum(len(c.enforced()) for c in self.corpora)


def schema_diversity_ablation(*, seed: int = 0) -> list[AblationArm]:
    """生成四组对照,把"数量"与"多样性"解耦。

    | 组 | schema 数 | 总条款数 | 问的问题 |
    |---|---|---|---|
    | few_one   | 1 | 少 | 基线 |
    | many_one  | 1 | 多 | **只加数量**有没有用 |
    | few_many  | 多 | 少 | **只加多样性**有没有用 |
    | many_many | 多 | 多 | 两者都加 |

    `many_one` 与 `few_many` 的对比是关键:若前者无增益而后者有,
    说明有效的是多样性,"更多 policy"这个说法要改成"更多**来源**"。
    """
    rng = random.Random(seed)
    base = SCHEMAS[0]
    one_small = build_from_schema(base, rng=rng)
    one_big = PolicyCorpus(name="one_big", clauses=[
        PolicyClause(id=f"C{i}_{j}", category=f"x{i}", title=t,
                     text=f"{txt}(变体 {i})")
        for i in range(4)
        for j, (t, txt) in enumerate(base.clauses, 1)])
    many_small = [build_from_schema(s, rng=rng) for s in SCHEMAS[:2]]
    many_big = [build_from_schema(s, rng=rng) for s in SCHEMAS]

    return [
        AblationArm("few_one", 1, len(one_small.enforced()), [one_small]),
        AblationArm("many_one", 1, len(one_big.enforced()), [one_big]),
        AblationArm("few_many", len(many_small),
                    sum(len(c.enforced()) for c in many_small), many_small),
        AblationArm("many_many", len(many_big),
                    sum(len(c.enforced()) for c in many_big), many_big),
    ]


def mix(corpora: list[PolicyCorpus], *, k: int | None = None,
        rng: random.Random | None = None) -> PolicyCorpus:
    """从多个来源混采条款,组成一条训练样本的语料。

    ⚠️ 混采会**放大 prompt 长度**。条款越多越吃 KV 预算,并加重位置偏置。
    k 应按 `StreamContextConfig.max_policy_tokens` 反推,不能无限加。
    """
    rng = rng or random.Random(0)
    pool = [c for corp in corpora for c in corp.enforced()]
    if k is not None and k < len(pool):
        pool = rng.sample(pool, k)
    rng.shuffle(pool)
    return PolicyCorpus(name="mixed", clauses=pool)


def estimate_policy_tokens(corpus: PolicyCorpus, *,
                           chars_per_token: float = 2.0) -> int:
    """粗估政策段的 token 数。超过 max_policy_tokens 就得砍。"""
    return int(len(corpus.render()) / chars_per_token)
