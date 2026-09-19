"""对抗性泄漏审计:用 sentinel 实际会读的特征去攻击数据集。

validate.py 查的是**规划层**的统计性质(切点数/位置/时长同分布)。
本模块查的是**渲染后**的实际信号:编码器在场景切换处强制插 I 帧,所以切点
在压缩域一定可见。可见不等于泄漏 —— 只要正负例切点同分布,关键帧模式就
不含标签信息。这里用分类器实证这一点。

判读:
  AUC ≈ 0.5            -> 干净
  AUC 明显 > 0.5       -> 泄漏,整个 benchmark 失效

⚠️ 小样本下检验力很低。n=40 时 AUC 的置信区间约 ±0.15,只能算冒烟测试。
本模块会报告置换检验 p 值和所需样本量提示,不让你把"没测出来"当成"没有"。
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import extract
from .schema import ALL_POOLS, ClipRecord, load

FEATURE_NAMES = ("关键帧数", "关键帧密度", "间隔均值", "间隔std", "间隔min",
                 "间隔max", "I帧率", "B帧率", "时长", "总帧数")


@dataclass
class AuditResult:
    n: int
    auc: float
    auc_std: float
    accuracy: float
    majority_baseline: float
    p_value: float
    top_features: list[str]
    underpowered: bool

    @property
    def leaked(self) -> bool:
        """泄漏判定:AUC 显著高于 0.5。"""
        return self.p_value < 0.05 and self.auc > 0.5

    def __str__(self) -> str:
        head = "✗ 检出泄漏" if self.leaked else "✓ 未检出泄漏"
        s = (f"{head}  n={self.n}  AUC={self.auc:.3f}±{self.auc_std:.3f}  "
             f"准确率={self.accuracy:.3f} (多数类 {self.majority_baseline:.3f})  "
             f"p={self.p_value:.3f}")
        if self.underpowered:
            s += (f"\n  ⚠️ 样本量不足:n={self.n} 时检验力很低,"
                  f"「未检出」不等于「不存在」。建议 n≥200 再下结论。")
        if self.leaked:
            s += f"\n  最可疑特征: {', '.join(self.top_features)}"
        return s


def codec_summary(video_path: str, duration_s: float) -> list[float]:
    """把一条视频的压缩域信号压成定长摘要向量。"""
    f = extract(video_path)
    kf = f.t_s[f.is_keyframe == 1]
    gaps = np.diff(kf) if len(kf) > 1 else np.array([0.0])
    return [len(kf), len(kf) / max(duration_s, 1e-6),
            float(gaps.mean()), float(gaps.std()),
            float(gaps.min()), float(gaps.max()),
            float((f.pict_type == 0).mean()), float((f.pict_type == 2).mean()),
            duration_s, len(f)]


class _FeatureCache:
    """按 (视频路径, mtime, 大小) 缓存压缩域摘要。

    抽取实测 0.79s/条,40 条即 31s。审计会被反复运行(改了规划就要重跑),
    没有缓存的话每次都白付这笔钱。
    """

    def __init__(self, cache_dir: str | None, video_dir: str):
        self.path = Path(cache_dir or video_dir).parent / ".sg2_codec_cache.json"
        try:
            self.data = json.loads(self.path.read_text())
        except Exception:
            self.data = {}
        self.dirty = False

    @staticmethod
    def _key(clip_id: str, video: str) -> str:
        try:
            st = Path(video).stat()
            sig = f"{clip_id}:{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            sig = clip_id
        return hashlib.sha256(sig.encode()).hexdigest()[:24]

    def get(self, clip_id: str, video: str, duration_s: float) -> list[float]:
        k = self._key(clip_id, video)
        if k not in self.data:
            self.data[k] = codec_summary(video, duration_s)
            self.dirty = True
        return self.data[k]

    def flush(self) -> None:
        if self.dirty:
            try:
                self.path.write_text(json.dumps(self.data))
            except OSError:
                pass


def audit_codec_leakage(manifest: str, video_dir: str, *, n_perm: int = 200,
                        seed: int = 0, n_jobs: int | None = None,
                        cache_dir: str | None = None) -> AuditResult:
    """用压缩域摘要特征尝试预测 safe/unsafe。

    Returns:
        AuditResult。`leaked=True` 表示数据集不可用。
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as e:
        raise RuntimeError("需要 scikit-learn:pip install scikit-learn") from e

    records: list[ClipRecord] = list(load(manifest, pool=ALL_POOLS))
    cache = _FeatureCache(cache_dir, video_dir)
    X, y = [], []
    for r in records:
        X.append(cache.get(r.id, f"{video_dir}/{r.id}.mp4",
                           r.media["duration_s"]))
        y.append(0 if r.is_safe else 1)
    cache.flush()
    X, y = np.asarray(X), np.asarray(y)

    if len(set(y)) < 2:
        raise ValueError("只有一种标签,无法审计")

    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("某一类样本太少,无法交叉验证")
    # 小数据上 n_jobs=-1 反而更慢:每次拟合只有几十棵浅树,派发开销高于计算。
    # 实测 n=40 时 n_jobs=-1 为 996ms/次,n_jobs=1 为 359ms/次,慢 2.8 倍。
    if n_jobs is None:
        n_jobs = 1 if len(y) < 500 else -1
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                 n_jobs=n_jobs)
    # 零分布只需形状,用更小的森林 —— 300 棵 x 200 次置换 x 5 折 = 30 万次
    # 拟合,对一个日常工具来说不可接受。
    clf_perm = RandomForestClassifier(n_estimators=60, random_state=seed,
                                      n_jobs=n_jobs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        obs = float(auc.mean())

        # 置换检验:打乱标签重跑,看观测 AUC 在零分布里的位置
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(n_perm):
            yp = rng.permutation(y)
            m = float(cross_val_score(clf_perm, X, yp, cv=cv,
                                      scoring="roc_auc").mean())
            if m >= obs:
                ge += 1
        p = (ge + 1) / (n_perm + 1)

        clf.fit(X, y)
        top = [FEATURE_NAMES[i]
               for i in np.argsort(clf.feature_importances_)[::-1][:3]]

    return AuditResult(n=len(y), auc=obs, auc_std=float(auc.std()),
                       accuracy=float(acc.mean()),
                       majority_baseline=float(max(y.mean(), 1 - y.mean())),
                       p_value=p, top_features=top, underpowered=len(y) < 200)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python -m sg2.audit <manifest.jsonl> <video_dir>")
        raise SystemExit(2)
    res = audit_codec_leakage(sys.argv[1], sys.argv[2])
    print(res)
    raise SystemExit(1 if res.leaked else 0)
