"""数据集级不变量校验。

逐条记录的合法性由 schema.py 的 __post_init__ 保证;这里查的是**跨记录**
的性质,尤其是拼接痕迹是否携带标签信息——那会让整个 benchmark 失效。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .schema import ALL_POOLS, ClipRecord, load


@dataclass
class Finding:
    level: str          # "error" | "warn"
    check: str
    detail: str

    def __str__(self) -> str:
        mark = "✗" if self.level == "error" else "!"
        return f"{mark} [{self.check}] {self.detail}"


def check_pool_disjoint(records: list[ClipRecord]) -> list[Finding]:
    """同一 id 不得出现在多个 pool。"""
    pools = defaultdict(set)
    for r in records:
        pools[r.id].add(r.pool)
    bad = {i: p for i, p in pools.items() if len(p) > 1}
    if bad:
        return [Finding("error", "pool_disjoint",
                        f"{len(bad)} 个 id 跨池出现,例: {list(bad.items())[:3]}")]
    return []


def check_splice_balanced(records: list[ClipRecord], *,
                          tol: float = 0.15) -> list[Finding]:
    """正负例的剪辑点数必须同分布。

    否则一个纯剪辑点计数器就能猜标签,检测延迟测的将是剪辑检测延迟。
    """
    spliced = [r for r in records if r.splice is not None]
    if not spliced:
        return []
    by_label = defaultdict(list)
    for r in spliced:
        by_label[r.is_safe].append(r.splice.n_cuts)
    if len(by_label) < 2:
        return [Finding("error", "splice_balanced",
                        "只有一种标签的视频被拼接过 —— 剪辑点直接泄漏标签")]
    means = {k: sum(v) / len(v) for k, v in by_label.items()}
    lo, hi = min(means.values()), max(means.values())
    if hi > 0 and (hi - lo) / hi > tol:
        return [Finding("error", "splice_balanced",
                        f"正负例平均剪辑数差异 {(hi-lo)/hi:.1%} > {tol:.0%}; "
                        f"safe={means.get(True):.2f} unsafe={means.get(False):.2f}")]
    return []


def _tvd(hist_a: list[float], hist_b: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(hist_a, hist_b))


def _norm(counts: list[int]) -> list[float]:
    tot = sum(counts)
    return [c / tot for c in counts] if tot else counts


def check_cut_position_leakage(records: list[ClipRecord], *, bins: int = 10,
                               n_perm: int = 500, alpha: float = 0.01,
                               seed: int = 0) -> list[Finding]:
    """剪辑点的**相对位置分布**不得与标签相关。

    用置换检验而非固定 TVD 阈值:小样本下随机位置本身就有可观的 TVD,
    固定阈值会在干净数据上误报,而误报几次之后这条检查就没人看了。
    """
    import random as _random

    spliced = [r for r in records if r.splice is not None]
    if len(spliced) < 20:
        return []

    obs = []  # (归一化位置, is_safe)
    for r in spliced:
        dur = r.media.get("duration_s") or 0
        if dur <= 0:
            continue
        for c in r.splice.cuts:
            obs.append((min(c.t_s / dur, 0.999), r.is_safe))
    if len(obs) < 40:
        return []

    def tvd_of(labels: list[bool]) -> float:
        h = {True: [0] * bins, False: [0] * bins}
        for (pos, _), lab in zip(obs, labels):
            h[lab][int(pos * bins)] += 1
        return _tvd(_norm(h[True]), _norm(h[False]))

    labels = [lab for _, lab in obs]
    observed = tvd_of(labels)

    rng = _random.Random(seed)
    shuffled = list(labels)
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if tvd_of(shuffled) >= observed:
            n_ge += 1
    p = (n_ge + 1) / (n_perm + 1)

    if p < alpha:
        return [Finding("error", "cut_position_leakage",
                        f"剪辑位置分布与标签相关: TVD={observed:.3f}, "
                        f"置换检验 p={p:.4f} < {alpha}")]
    if p < 10 * alpha:
        return [Finding("warn", "cut_position_leakage",
                        f"剪辑位置分布可疑: TVD={observed:.3f}, p={p:.4f}")]
    return []


def check_calibration_sample_size(records: list[ClipRecord], *,
                                  target_recall: float = 0.95,
                                  delta: float = 0.10) -> list[Finding]:
    """每个类别的校准正例是否够做 conformal 下界。

    精确二项(Clopper-Pearson)零失败下界: p_L = delta**(1/n) >= target
    => n >= ln(delta) / ln(target)。target=0.95, delta=0.10 时 n>=45。
    """
    n_min = math.ceil(math.log(delta) / math.log(target_recall))
    per_cat = Counter()
    for r in records:
        if r.pool != "calibration" or r.is_safe:
            continue
        for e in r.events:
            per_cat[e.category] += 1
    out = []
    for cat, n in sorted(per_cat.items()):
        if n < n_min:
            out.append(Finding("warn", "calib_sample_size",
                               f"类别 {cat}: 校准正例 {n} < {n_min} "
                               f"(recall≥{target_recall:.0%} @ {1-delta:.0%} 置信,零失败)"))
    if not per_cat:
        out.append(Finding("warn", "calib_sample_size", "校准池中没有正例"))
    return out


def check_axis_coverage(records: list[ClipRecord]) -> list[Finding]:
    """四个难度轴每个 bin 都要有样本,否则该轴不可分析。"""
    out = []
    for axis in ("temporal_sparsity", "perceptual_subtlety",
                 "modality_locus", "context_dependence"):
        seen = Counter(r.axes.get(axis, {}).get("bin") for r in records)
        seen.pop(None, None)
        if len(seen) < 2:
            out.append(Finding("warn", "axis_coverage",
                               f"{axis} 只有 {len(seen)} 个 bin 有样本: {dict(seen)}"))
    return out


def check_json_schema(records: list[ClipRecord]) -> list[Finding]:
    """逐条记录对照 spec/manifest.schema.json。

    schema 只能查**单条**记录的合法性;跨记录的分布性质(拼接是否泄漏)
    由本模块其他检查负责。两层不可互相替代。
    """
    try:
        import json
        import jsonschema
    except ImportError:
        return [Finding("warn", "json_schema", "未安装 jsonschema,跳过逐条校验")]
    from pathlib import Path
    sp = Path(__file__).resolve().parents[1] / "spec/manifest.schema.json"
    if not sp.exists():
        return [Finding("warn", "json_schema", f"schema 不存在: {sp}")]
    v = jsonschema.Draft202012Validator(json.loads(sp.read_text()))
    bad = []
    for r in records:
        errs = list(v.iter_errors(r.to_json()))
        if errs:
            bad.append(f"{r.id}: {errs[0].message[:70]}")
    if bad:
        return [Finding("error", "json_schema",
                        f"{len(bad)} 条不合 schema,例: {bad[:2]}")]
    return []


CHECKS = (check_json_schema, check_pool_disjoint, check_splice_balanced, check_cut_position_leakage,
          check_calibration_sample_size, check_axis_coverage)


def validate(manifest: str) -> list[Finding]:
    records = list(load(manifest, pool=ALL_POOLS))
    findings: list[Finding] = []
    for c in CHECKS:
        findings.extend(c(records))
    return findings


if __name__ == "__main__":
    import sys
    fs = validate(sys.argv[1])
    for f in fs:
        print(f)
    n_err = sum(1 for f in fs if f.level == "error")
    print(f"\n{len(fs)} 条发现, {n_err} 个 error")
    sys.exit(1 if n_err else 0)
