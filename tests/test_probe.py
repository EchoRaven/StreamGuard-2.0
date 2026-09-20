"""rolling attention probe。

重点不在"能训出来",而在两条容易悄悄错掉的性质:
  - 在线量**必须能降**(单调量无法撤销,见 anytime.py 同类教训)
  - 通道开了却没给探针**必须报错**(静默少一路会让融合头输入维度变)
"""
import numpy as np
import pytest

from sg2.config import ChannelConfig, SentinelConfig
from sg2.models.sentinel import MultiChannelSentinel
from sg2.probe import (RollingAttentionProbe, global_attn_score,
                       mean_pool_score)

D, W = 16, 5


def _mk(rng, n, harmful, direction, strength=3.0):
    Z = rng.normal(0, 1, (n, D))
    if harmful:
        s = rng.integers(0, max(1, n - W))
        Z[s:s + W] += strength * direction
    return Z


def _corpus(rng, n_seq, seq_len, direction):
    seqs = [_mk(rng, seq_len, i % 2 == 0, direction) for i in range(n_seq)]
    y = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n_seq)])
    return seqs, y


def _auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


@pytest.fixture
def trained():
    rng = np.random.default_rng(0)
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    seqs, y = _corpus(rng, 120, 60, d)
    return RollingAttentionProbe(dim=D, window=W, epochs=150).fit(seqs, y), d


def test_separates_held_out(trained):
    probe, direction = trained
    rng = np.random.default_rng(99)
    seqs, y = _corpus(rng, 200, 60, direction)
    sc = np.array([probe.score_sequence(Z) for Z in seqs])
    assert _auroc(sc[y == 1], sc[y == 0]) > 0.95


def test_param_count_is_tiny(trained):
    """q + w + b + log_temp = 2d+2。探针便宜是它全部的意义。"""
    probe, _ = trained
    assert probe.n_params == 2 * D + 2


def test_dim_mismatch_raises(trained):
    probe, _ = trained
    with pytest.raises(ValueError, match="维度"):
        probe.online().update(np.zeros(D + 1))


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="未训练"):
        RollingAttentionProbe(dim=D).score_sequence(np.zeros((10, D)))


def test_shorter_than_window_is_legal(trained):
    """真实视频开头就只有几帧,短序列是合法输入而非错误。"""
    probe, _ = trained
    assert np.isfinite(probe.score_sequence(np.zeros((2, D))))


# ---------- 在线语义:本模块最重要的两条 ----------

def test_online_value_can_decrease(trained):
    """⚠️ 逐 tick 的 value 必须能降 —— 否则无法 clear、无法闭合事件。"""
    probe, direction = trained
    rng = np.random.default_rng(7)
    on = probe.online()
    # 前段有害,后段良性 -> 有害片段滑出窗口后分数必须回落
    Z = np.vstack([_mk(rng, W, True, direction), rng.normal(0, 1, (30, D))])
    vals = [on.update(z) for z in Z]
    assert min(vals[W + W:]) < max(vals[:W]), "有害片段滑出窗口后分数没有回落"


def test_running_max_is_monotone_and_not_for_decisions(trained):
    """running_max 单调 —— 保留它只为对齐离线数字,不做判决。"""
    probe, direction = trained
    rng = np.random.default_rng(7)
    on = probe.online()
    seen = []
    for z in np.vstack([_mk(rng, W, True, direction), rng.normal(0, 1, (20, D))]):
        on.update(z)
        seen.append(on.running_max)
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_running_max_before_any_frame_raises(trained):
    probe, _ = trained
    with pytest.raises(RuntimeError, match="还没喂过帧"):
        _ = probe.online().running_max


def test_warm_flag_tracks_window_fill(trained):
    probe, _ = trained
    on = probe.online()
    for i in range(W - 1):
        on.update(np.zeros(D))
        assert not on.warm, f"第 {i+1} 帧不该 warm"
    on.update(np.zeros(D))
    assert on.warm


def test_reset_clears_online_state(trained):
    """跨流复用探针状态会串味。"""
    probe, direction = trained
    rng = np.random.default_rng(3)
    on = probe.online()
    for z in _mk(rng, 20, True, direction):
        on.update(z)
    on.reset()
    assert on.stats["seen"] == 0 and not on.warm


# ---------- 科学主张:窗口买的是失配下的鲁棒性 ----------

def test_rolling_beats_global_attention_under_length_shift():
    """在**短序列上训**、**长序列上用** —— 这正是 Gemini probes 的部署条件。

    窗口买的不是精度,是 query 失配时的鲁棒性。用同一组学到的 q/w 对比,
    唯一差别是有没有窗口。
    """
    rng = np.random.default_rng(1)
    direction = rng.normal(size=D); direction /= np.linalg.norm(direction)
    seqs, y = _corpus(rng, 120, 30, direction)        # 训练:短
    probe = RollingAttentionProbe(dim=D, window=W, epochs=150).fit(seqs, y)
    q, w, b = probe._q, probe._w, probe._b

    te, ty = _corpus(rng, 200, 1500, direction)       # 部署:长 50 倍
    roll = np.array([probe.score_sequence(Z) for Z in te])
    glob = np.array([global_attn_score(Z, q, w, b) for Z in te])
    mean = np.array([mean_pool_score(Z, w, b) for Z in te])

    a_roll = _auroc(roll[ty == 1], roll[ty == 0])
    a_glob = _auroc(glob[ty == 1], glob[ty == 0])
    a_mean = _auroc(mean[ty == 1], mean[ty == 0])
    assert a_roll > a_glob, f"rolling {a_roll:.3f} 没赢过 global {a_glob:.3f}"
    assert a_roll > a_mean, f"rolling {a_roll:.3f} 没赢过 mean {a_mean:.3f}"


# ---------- 接线 ----------

def test_channel_on_without_probe_raises():
    """⚠️ 静默少一路通道会让融合头输入维度悄悄变,分数不可比。"""
    cfg = SentinelConfig(channels=ChannelConfig(rolling_probe=True),
                         encoder=__import__("sg2.config", fromlist=["x"])
                         .EncoderConfig(name="hist"))
    with pytest.raises(ValueError, match="rolling_probe"):
        MultiChannelSentinel(cfg)


def test_config_rejects_probe_without_vision():
    with pytest.raises(ValueError, match="rolling_probe"):
        SentinelConfig(channels=ChannelConfig(vision=False, rolling_probe=True))


def test_sentinel_emits_probe_channel(trained):
    from sg2.config import EncoderConfig
    probe, _ = trained

    class _Enc:                       # 固定维度的假编码器
        def encode(self, frames):
            n = len(frames)
            out = np.asarray(frames, dtype=float).reshape(n, -1)[:, :D]
            if out.shape[1] < D:
                out = np.pad(out, ((0, 0), (0, D - out.shape[1])))
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    cfg = SentinelConfig(channels=ChannelConfig(rolling_probe=True),
                         encoder=EncoderConfig(name="hist"))
    s = MultiChannelSentinel(cfg, encoder=_Enc(), probe=probe)
    rng = np.random.default_rng(0)
    out = [s.score_frame(t, frame=rng.random((8, 8, 3))) for t in range(6)]
    assert all("probe" in o.channels for o in out)
    assert s.stats["probe"]["seen"] == 6


def test_vectorized_pool_matches_naive_reference():
    """⚠️ `_pool` 被向量化过(90s -> 2s)。必须与朴素双层循环逐位等价。

    优化核心计算而不留等价性测试,是最容易引入静默错误的改法。
    """
    rng = np.random.default_rng(11)
    probe = RollingAttentionProbe(dim=D, window=W)
    probe._q = rng.normal(0, 1, D)
    probe._w = rng.normal(0, 1, D)
    probe._b = 0.37

    def naive(Z):
        n, w_ = len(Z), min(probe.window, len(Z))
        s = (Z @ probe._q) / np.sqrt(probe.dim)
        best = (-np.inf, None, None, 0)
        for t in range(n - w_ + 1):
            a = np.exp(s[t:t+w_] - s[t:t+w_].max()); a /= a.sum()
            pooled = a @ Z[t:t+w_]
            val = float(pooled @ probe._w + probe._b)
            if val > best[0]:
                best = (val, pooled, a, t)
        return best[1], best[2], best[3]

    for n in (3, W, W + 1, 50, 300):
        Z = rng.normal(0, 1, (n, D))
        pv, av, tv = probe._pool(Z)
        pn, an, tn = naive(Z)
        assert tv == tn, f"n={n}: 选中的窗不同 {tv} vs {tn}"
        np.testing.assert_allclose(av, an, atol=1e-12, err_msg=f"n={n}: 权重")
        np.testing.assert_allclose(pv, pn, atol=1e-12, err_msg=f"n={n}: 池化向量")


# ---------- 温度:注意力会不会静默失效 ----------

def test_attention_not_degenerate_on_normalised_embeddings():
    """⚠️ 归一化嵌入 + 固定 1/√d 会让 softmax 退化成均匀平均。

    实测触发过:SigLIP2 的 ‖z‖=1、d=1152,logit 展布只有 0.002,
    有效样本数 1500/1500 —— 注意力整个是死的,而系统照跑、AUROC 还不错。
    可学温度就是为了堵这个。
    """
    rng = np.random.default_rng(4)
    direction = rng.normal(size=D); direction /= np.linalg.norm(direction)

    def unit(n, harmful):
        Z = _mk(rng, n, harmful, direction)
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)   # L2 归一化

    seqs = [unit(40, i % 2 == 0) for i in range(120)]
    y = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(120)])
    probe = RollingAttentionProbe(dim=D, window=W, epochs=150).fit(seqs, y)

    diag = probe.attention_diagnostics(unit(400, True))
    assert not diag["degenerate"], (
        f"注意力退化成均匀平均:有效样本 {diag['effective_n']:.0f}/{diag['n']}, "
        f"logit 展布 {diag['logit_spread']:.2e}, 温度 {diag['temperature']:.2f}")


def test_fixed_sqrt_d_would_degenerate():
    """反证:把温度按死成 1(即纯 1/√d),在归一化嵌入上必然退化。

    这条固定的是**问题存在**,不是我们的修复 —— 没有它,上面那条测试
    通过了也不知道是修好了还是问题本来就不存在。
    """
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(400, D))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    probe = RollingAttentionProbe(dim=D, window=W)
    probe._q = rng.normal(0, 0.02, D)
    probe._w = rng.normal(0, 0.02, D)
    probe._log_temp = 0.0                     # τ=1,退回教科书的 1/√d
    diag = probe.attention_diagnostics(Z)
    assert diag["degenerate"], "预期退化但没退化 —— 反证失效,上面那条测试就没有意义"


def test_diagnostics_reports_temperature_and_effective_n(trained):
    probe, direction = trained
    rng = np.random.default_rng(8)
    d = probe.attention_diagnostics(_mk(rng, 100, True, direction))
    assert d["n"] == 100 and 1.0 <= d["effective_n"] <= 100.0
    assert d["temperature"] > 0


def test_global_attn_baseline_needs_temperature():
    """不传温度时 global_attn 会塌成 mean_pool —— 拿它当对照等于没对照。"""
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(300, D)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    q = rng.normal(0, 0.02, D); w = rng.normal(0, 0.02, D)
    assert global_attn_score(Z, q, w, 0.0, temperature=1.0) == pytest.approx(
        mean_pool_score(Z, w, 0.0), abs=1e-4)
    assert global_attn_score(Z, q, w, 0.0, temperature=50.0) != pytest.approx(
        mean_pool_score(Z, w, 0.0), abs=1e-4)


def test_attention_adds_nothing_over_windowed_mean():
    """★ 实测负结果:注意力在窗内不贡献,增益全部来自"开窗 + 取 max"。

    真实 SigLIP2 嵌入 + 真实 TikTok 帧上,三个 needle 难度、四个序列长度,
    `rolling` 相对 `rolling_mean`(同样开窗取 max,但窗内用平均)的 AUROC
    差值最大 +0.001。见 docs/17_PROBE.md。

    这条测试钉住的是**结论仍然成立**:哪天改了实现让注意力真的起作用了,
    它会红,提醒去更新文档里的数字,而不是让文档悄悄变成错的。
    """
    from sg2.probe import rolling_mean_score
    rng = np.random.default_rng(21)
    direction = rng.normal(size=D); direction /= np.linalg.norm(direction)

    def unit(n, harmful):
        Z = _mk(rng, n, harmful, direction, strength=1.5)
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)

    seqs = [unit(30, i % 2 == 0) for i in range(120)]
    y = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(120)])
    probe = RollingAttentionProbe(dim=D, window=W, epochs=150).fit(seqs, y)
    w, b = probe._w, probe._b

    te = [unit(400, i % 2 == 0) for i in range(200)]
    ty = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(200)])
    a_roll = _auroc(*[[probe.score_sequence(Z) for Z, t in zip(te, ty) if t == k]
                      for k in (1, 0)])
    a_mean = _auroc(*[[rolling_mean_score(Z, w, b, W) for Z, t in zip(te, ty) if t == k]
                      for k in (1, 0)])
    assert abs(a_roll - a_mean) < 0.03, (
        f"注意力现在有贡献了(rolling {a_roll:.3f} vs 窗内平均 {a_mean:.3f}) —— "
        "去更新 docs/17_PROBE.md 的结论")
