"""难度因子的分箱规则。

连续量是真值,bin 只是方便切片的派生值,任何时候可以用这里的规则重算。
改分箱规则不需要重新标注数据。
"""
from __future__ import annotations

TEMPORAL_SPARSITY_EDGES = [(0.01, "dense"), (0.001, "sparse")]  # 下限用 ultra_sparse
PERCEPTUAL_EDGES = [(0.05, "salient"), (0.01, "moderate")]      # 按 pixel_area_frac
CONTEXT_EDGES = [(0.0, "self_contained"), (10.0, "short_context")]


def temporal_sparsity_bin(ratio: float) -> str:
    """needle 总时长 / 视频时长。"""
    for edge, name in TEMPORAL_SPARSITY_EDGES:
        if ratio >= edge:
            return name
    return "ultra_sparse"


def perceptual_subtlety_bin(pixel_area_frac: float, requires_ocr: bool = False) -> str:
    """需要 OCR 才能判定的一律算 subtle——小字对廉价模型等价于不可见。"""
    if requires_ocr:
        return "subtle"
    for edge, name in PERCEPTUAL_EDGES:
        if pixel_area_frac >= edge:
            return name
    return "subtle"


def context_dependence_bin(window_required_s: float) -> str:
    if window_required_s <= CONTEXT_EDGES[0][0]:
        return "self_contained"
    if window_required_s <= CONTEXT_EDGES[1][0]:
        return "short_context"
    return "long_context"


def modality_locus_bin(decisive: list[str]) -> str:
    if len(decisive) > 1:
        return "multi"
    m = decisive[0] if decisive else "composition"
    return {"pixel": "visual", "motion": "visual", "speech": "speech",
            "ocr": "text", "composition": "composition"}.get(m, "composition")
