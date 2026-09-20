"""政策增强:逼模型**读**政策而不是**背**政策。

在固定语料上 SFT,模型可以记住"C1 就是色情"从此不再读条款文本。
那样归纳出的新政策不会改变它的判决 —— 整条政策归纳链(docs/10)白做,
因为中间层在适配时是**不重训**的,它只是读新政策。

六种增强,每一种针对一条可被记住的线索:

| 增强 | 堵住的记忆线索 | 默认 |
| --- | --- | --- |
| `shuffle` | 条款位置 | ✅ |
| `rename_ids` | 条款 id 与类别的固定对应 | ✅ |
| `subset` | "语料里总是这六条" | ✅ |
| `paraphrase` | 条款的固定措辞 | 需提供改写器 |
| `inject_novel` | "没见过的 id 一定不重要" | ✅ |
| `mismatch` | **内容与政策的固定对应** | ✅ |

最后一条最重要:有一定比例的样本,政策**不覆盖**画面里的内容,
正确动作是 `clear` 或 `uncovered`。没有它,模型学到的是"看见这种画面
就 flag",政策文本变成装饰。
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable

from ..policy import ClauseStatus, PolicyClause, PolicyCorpus


@dataclass
class AugConfig:
    shuffle: bool = True
    rename_ids: bool = True
    subset: bool = True
    min_clauses: int = 2
    paraphrase: Callable[[str], str] | None = None
    inject_novel: bool = True
    n_novel: int = 1
    mismatch_rate: float = 0.25        # 政策不覆盖内容的样本比例
    keep_safe_exit: bool = True        # 始终保留"其余放行"出口

    def __post_init__(self):
        if not 0.0 <= self.mismatch_rate <= 1.0:
            raise ValueError("mismatch_rate 必须在 [0,1]")
        if self.min_clauses < 1:
            raise ValueError("min_clauses 至少为 1")


SAFE_EXIT = PolicyClause(
    id="Z0_ok", category="Z0_ok", title="其余一律放行",
    text="未被上述条款覆盖的画面均不违规,应回 clear。")

NOVEL_POOL = [
    ("车牌", "禁止出现可识别的车牌号码。"),
    ("商标", "禁止出现未授权的第三方商标。"),
    ("坐标", "禁止出现精确的地理坐标或住址。"),
    ("二维码", "禁止出现可扫描的二维码。"),
    ("闪烁", "禁止出现每秒超过三次的强烈闪烁。"),
]


def _rand_id(rng: random.Random, n: int = 6) -> str:
    return (rng.choice(string.ascii_uppercase)
            + str(rng.randint(1, 9)) + "_"
            + "".join(rng.choice(string.ascii_lowercase) for _ in range(n)))


@dataclass
class AugmentedSample:
    """一条增强后的训练样本。"""
    corpus: PolicyCorpus
    target_clause_id: str | None      # 应被引用的条款;None = 政策不覆盖
    is_mismatch: bool
    id_map: dict[str, str]            # 原 id -> 新 id,便于回溯

    @property
    def expected_action(self) -> str:
        return "clear" if self.is_mismatch else "flag"


def augment(corpus: PolicyCorpus, target_id: str | None, *,
            cfg: AugConfig | None = None,
            rng: random.Random | None = None) -> AugmentedSample:
    """为一条训练样本生成一个扰动后的政策语料。

    Args:
        target_id: 该样本内容实际违反的条款。None 表示这条内容本身安全。

    ⚠️ `mismatch` 时会把目标条款**移除**,于是同一段视频配上不同政策,
    正确答案不同。这是唯一能教会模型"答案由政策决定"的信号。
    """
    cfg = cfg or AugConfig()
    rng = rng or random.Random(0)
    clauses = [c for c in corpus.enforced() if c.id != SAFE_EXIT.id]

    mismatch = bool(target_id) and rng.random() < cfg.mismatch_rate
    keep_id = None if mismatch else target_id

    # 子集采样:必须保留目标条款(除非是 mismatch)
    if cfg.subset and len(clauses) > cfg.min_clauses:
        must = [c for c in clauses if c.id == keep_id]
        rest = [c for c in clauses if c.id != keep_id and c.id != target_id]
        k = rng.randint(max(cfg.min_clauses - len(must), 0), len(rest))
        clauses = must + rng.sample(rest, k)
    elif mismatch:
        clauses = [c for c in clauses if c.id != target_id]

    if cfg.inject_novel:
        for title, text in rng.sample(NOVEL_POOL,
                                      min(cfg.n_novel, len(NOVEL_POOL))):
            clauses.append(PolicyClause(id=_rand_id(rng), category="novel",
                                        title=title, text=text))

    id_map: dict[str, str] = {}
    out: list[PolicyClause] = []
    for c in clauses:
        new_id = _rand_id(rng) if cfg.rename_ids else c.id
        id_map[c.id] = new_id
        text = cfg.paraphrase(c.text) if cfg.paraphrase else c.text
        out.append(PolicyClause(id=new_id, category=c.category, title=c.title,
                                text=text, status=ClauseStatus.ACTIVE))

    if cfg.keep_safe_exit:
        out.append(SAFE_EXIT)
    if cfg.shuffle:
        rng.shuffle(out)

    return AugmentedSample(
        corpus=PolicyCorpus(name="aug", clauses=out),
        target_clause_id=id_map.get(keep_id) if keep_id else None,
        is_mismatch=mismatch, id_map=id_map)


def augment_batch(corpus: PolicyCorpus, targets: list[str | None], *,
                  cfg: AugConfig | None = None,
                  seed: int = 0) -> list[AugmentedSample]:
    rng = random.Random(seed)
    return [augment(corpus, t, cfg=cfg, rng=rng) for t in targets]


def mismatch_stats(samples: list[AugmentedSample]) -> dict:
    """核对增强是否真的生效。

    ⚠️ 尤其要看 `unique_id_ratio`。它接近 1 才说明 id 确实被随机化了 ——
    否则模型仍能把 id 与类别绑死,增强形同虚设。
    """
    n = len(samples) or 1
    ids = [i for s in samples for i in s.id_map.values()]
    sizes = [len(s.corpus.enforced()) for s in samples]
    return {
        "n": len(samples),
        "mismatch_rate": round(sum(s.is_mismatch for s in samples) / n, 3),
        "unique_id_ratio": round(len(set(ids)) / max(len(ids), 1), 3),
        "clauses_min": min(sizes) if sizes else 0,
        "clauses_max": max(sizes) if sizes else 0,
        "has_safe_exit": all(
            any(c.id == SAFE_EXIT.id for c in s.corpus.enforced())
            for s in samples),
    }
