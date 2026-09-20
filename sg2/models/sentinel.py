"""Sentinel 实现:多通道融合 -> 单一连续风险分。

通道按配置开关。每路都便宜,合起来把安全/不安全分数的 KL 间隔拉大 ——
而延迟界是该间隔的函数,所以**加通道比加模型划算**(docs/01_MODELS.md §1.2)。
"""
from __future__ import annotations

import numpy as np

from ..config import SentinelConfig
from ..registry import build, register
from .base import SentinelOutput


class DegenerateEmbedding(RuntimeError):
    """嵌入空间区分不了相邻帧,去重无法工作。

    静默去掉 90% 的帧而系统照跑,是最难发现的一类失败。宁可在标定时报错。
    """


@register("sentinel", "multichannel")
class MultiChannelSentinel:
    """always-on 打分器。

    去重是**内容自适应**的,因此是可被对手放大的成本通道:注入运动噪声
    就能让去重失效、解码量爆炸。`max_decode_fps_multiplier` 是运行时硬上限。
    """

    def __init__(self, cfg: SentinelConfig | None = None, *,
                 encoder=None, fusion=None, prototype: np.ndarray | None = None):
        self.cfg = cfg or SentinelConfig()
        self.encoder = encoder or build("encoder", self.cfg.encoder.name,
                                        self.cfg.encoder)
        self.fusion = fusion
        self._proto = prototype
        self.reset()

    def reset(self) -> None:
        self._ref = getattr(self, "_ref", None)
        self._last_emb: np.ndarray | None = None
        self._decoded = 0
        self._skipped = 0
        self._window_start_s: float | None = None
        self._window_decoded = 0

    # ---------- 通道 ----------

    def _vision_score(self, frame: np.ndarray) -> tuple[float, bool]:
        """返回 (分数, 是否真的解码了)。去重命中时复用上一帧。"""
        emb = self.encoder.encode(frame[None])[0]
        if (self._last_emb is not None
                and float(emb @ self._last_emb) > 1.0 - self.cfg.dedup_embedding_delta):
            self._skipped += 1
            return self._proto_sim(self._last_emb), False
        self._last_emb = emb
        self._decoded += 1
        return self._proto_sim(emb), True

    def calibrate_dedup(self, frames: np.ndarray, *,
                        keep_frac: float = 0.3,
                        tol: float = 0.12) -> float:
        """从数据标定去重阈值,并就地生效。

        ⚠️ 去重阈值**依赖编码器**。同一个 0.02 在不同嵌入空间里含义完全不同
        —— 实测:直方图编码器下两张互不相关的随机图余弦也 >0.98,
        硬编码阈值会把它们判为重复。

        做法:取相邻帧余弦相似度的分位数,使大约 `keep_frac` 的帧被保留解码。

        Args:
            frames: (N,H,W,3) 一段代表性的连续帧。
            keep_frac: 期望解码的帧比例。
        Returns:
            标定出的 delta(= 1 - 相似度阈值)。
        """
        if len(frames) < 3:
            raise ValueError("至少需要 3 帧来标定")
        emb = self.encoder.encode(frames)
        sims = np.sum(emb[1:] * emb[:-1], axis=1)

        # 去重发生在 sim > thr,故被解码的比例 = F(thr) = keep_frac,
        # 阈值取 keep_frac 分位数(不是 1-keep_frac —— 方向反了会全部去重)
        thr = float(np.quantile(sims, keep_frac))

        # 退化判据:直接检查**标定出的阈值能否达到要求的解码比例**,
        # 而不是用分布展布这类代理量。
        #
        # ⚠️ 初版用 10-90 分位展布判退化,在真实视频上是错的:真实序列
        # 绝大多数相邻帧在镜头内(sim≈0.99),少数在镜头边界(sim≈0.79),
        # 分布天然严重偏斜 —— 展布小是正常形状,不是退化。实测 SigLIP2
        # 在真实帧上 sims∈[0.791,0.997] 判别力充足,却被展布判据误杀。
        achieved = float((sims < thr).mean())
        if abs(achieved - keep_frac) > tol:
            raise DegenerateEmbedding(
                f"阈值无法达到目标解码比例:要 {keep_frac:.0%},实得 "
                f"{achieved:.0%}(sims 全在 [{sims.min():.3f}, {sims.max():.3f}],"
                f"去重点 {thr:.4f})。该嵌入空间区分不了这些帧 —— "
                "换一个真实的视觉编码器,或把 channels.vision 关掉。")

        self.cfg.dedup_embedding_delta = max(1e-6, 1.0 - thr)
        return self.cfg.dedup_embedding_delta

    def _proto_sim(self, emb: np.ndarray) -> float:
        if self._proto is None:
            return 0.0
        return float(emb @ self._proto)

    def _ood(self, emb: np.ndarray) -> float:
        """离参考集有多远。1 = 完全没见过。

        用与参考嵌入的**最大**相似度:只要像其中任何一个就不算 OOD。
        没有参考集时返回 0(不声称知道),而不是 1(不乱报警)。
        """
        if self._ref is None or len(self._ref) == 0:
            return 0.0
        return float(np.clip(1.0 - (self._ref @ emb).max(), 0.0, 1.0))

    def set_reference(self, embs: np.ndarray) -> None:
        """装载 OOD 参考集。应来自 pool="compile",**不能用校准池**。"""
        e = np.atleast_2d(embs)
        self._ref = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _codec_score(codec: np.ndarray | None) -> float:
        """压缩域:相对码率 + 突变。不解码即可得,成本接近 0。"""
        if codec is None or len(codec) == 0:
            return 0.0
        c = np.atleast_2d(codec)
        return float(np.clip(c[:, 0].mean() - 1.0, -3.0, 3.0))

    @staticmethod
    def _text_score(text: str | None) -> float:
        """ASR/OCR/弹幕 的占位打分。真实实现接一个文本风险分类器。"""
        return 0.0 if not text else min(len(text) / 500.0, 1.0)

    # ---------- 预算护栏 ----------

    def _budget_ok(self, t_s: float) -> bool:
        """每分钟解码帧数的硬上限,防成本放大攻击。"""
        if self._window_start_s is None or t_s - self._window_start_s >= 60.0:
            self._window_start_s = t_s
            self._window_decoded = 0
        cap = self.cfg.sample_fps * 60.0 * self.cfg.max_decode_fps_multiplier
        return self._window_decoded < cap

    # ---------- 主入口 ----------

    def score_frame(self, t_s: float, *, frame: np.ndarray | None = None,
                    codec: np.ndarray | None = None,
                    text: str | None = None) -> SentinelOutput:
        ch: dict[str, float] = {}
        decoded = False

        if self.cfg.channels.codec:
            ch["codec"] = self._codec_score(codec)
        if self.cfg.channels.vision and frame is not None:
            if self._budget_ok(t_s):
                ch["vision"], decoded = self._vision_score(frame)
                if decoded:
                    self._window_decoded += 1
            else:
                ch["vision"] = self._proto_sim(self._last_emb) \
                    if self._last_emb is not None else 0.0
        for flag, key in (("asr", "asr"), ("ocr", "ocr"), ("chat", "chat")):
            if getattr(self.cfg.channels, flag):
                ch[key] = self._text_score(text)

        if self.fusion is not None:
            vec = np.array([[ch.get(k, 0.0) for k in sorted(ch)]])
            score = float(self.fusion.score(vec)[0])
        else:
            score = float(sum(ch.values()))

        ood = self._ood(self._last_emb) if self._last_emb is not None else 0.0
        return SentinelOutput(t_s=t_s, score=score, channels=ch,
                              decoded=decoded, ood=ood)

    @property
    def stats(self) -> dict:
        tot = self._decoded + self._skipped
        return {"decoded": self._decoded, "skipped": self._skipped,
                "dedup_rate": self._skipped / tot if tot else 0.0}


def build_sentinel(cfg: SentinelConfig, **kw):
    return build("sentinel", cfg.name, cfg, **kw)
