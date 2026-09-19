"""配置 / 注册表 / 编码器 / sentinel / CUSUM 的测试。

这些都是**可配置骨架**的一部分:骨架出错时表现为"跑得动但学错东西",
所以约束要在构造时就拦住,而不是靠文档提醒。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import (CoverageConfig, CusumConfig, EncoderConfig, SG2Config,
                        StreamContextConfig)
from sg2.cusum import CusumController
from sg2.models.base import Sentinel, VisionEncoder
from sg2.models.encoders import build_encoder
from sg2.models.sentinel import DegenerateEmbedding, build_sentinel
from sg2.registry import UnknownComponent, available, build

RNG = np.random.default_rng(0)


def _shots(n_shots=12, per_shot=5, drift=0.02, size=32):
    """镜头内渐变、镜头间跳变 —— 真实视频的结构。"""
    fs = []
    for _ in range(n_shots):
        base = RNG.random((size, size, 3)).astype(np.float32)
        for _k in range(per_shot):
            fs.append(np.clip(base + RNG.normal(0, drift, base.shape), 0, 1
                              ).astype(np.float32))
    return np.stack(fs)


# ==================== 配置 ====================

def test_n_min_matches_documented_45():
    assert SG2Config().calibration.n_min == 45


def test_rho_zero_is_refused():
    """rho=0 会让优雅退化定理失效。"""
    with pytest.raises(ValueError, match="优雅退化|rho"):
        CoverageConfig(rho=0.0)


def test_sink_tokens_zero_is_refused():
    with pytest.raises(ValueError, match="sink"):
        StreamContextConfig(sink_tokens=0)


def test_beta_ge_gamma_is_refused():
    from sg2.config import RewardConfigC
    with pytest.raises(ValueError, match="退化|beta_fa"):
        RewardConfigC(beta_fa=1.0, gamma_hit=1.0)


def test_turing_rejects_bf16_and_flash_attn():
    """sm_75 没有 bf16 也没有 FA2。不检查就会在跑起来后以 NaN 形式暴露。"""
    p = SG2Config().validate_for_device((7, 5))
    assert any("bfloat16" in x for x in p)
    assert any("FlashAttention" in x for x in p)


def test_ampere_has_no_complaints():
    assert SG2Config().validate_for_device((8, 0)) == []


def test_override_by_dotted_path():
    c = SG2Config().override("midtier.lora.rank", 32)
    assert c.midtier.lora.rank == 32


def test_override_revalidates():
    """覆盖后必须重跑该段校验,否则能绕过结构性约束。"""
    with pytest.raises(ValueError):
        SG2Config().override("coverage.rho", 0.0)


def test_override_unknown_key_is_refused():
    with pytest.raises(ValueError, match="没有"):
        SG2Config().override("midtier.nonexistent", 1)


def test_roundtrip_yaml(tmp_path):
    c = SG2Config().override("cusum.h", 7.5)
    p = c.to_yaml(tmp_path / "c.yaml")
    assert SG2Config.from_yaml(p).cusum.h == 7.5


def test_unknown_config_key_is_refused():
    with pytest.raises(ValueError, match="未知配置项"):
        SG2Config.from_dict({"cusum": {"h": 1.0, "typo_here": 3}})


# ==================== 注册表 ====================

def test_encoders_are_registered():
    assert {"hist", "random", "siglip2"} <= set(available("encoder"))


def test_unknown_component_lists_options():
    with pytest.raises(UnknownComponent, match="可用"):
        build("encoder", "does-not-exist")


@pytest.mark.parametrize("name", ["hist", "random"])
def test_encoder_satisfies_protocol_and_normalizes(name):
    e = build_encoder(EncoderConfig(name=name))
    assert isinstance(e, VisionEncoder)
    z = e.encode(RNG.random((4, 16, 16, 3)).astype(np.float32))
    assert z.shape == (4, e.dim)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_siglip2_constructs_without_downloading():
    """惰性导入:构造时不应触发权重下载或 import torch。"""
    assert build_encoder(EncoderConfig(name="siglip2")).__class__.__name__ \
        == "SigLIP2Encoder"


# ==================== Sentinel ====================

def _sent(name="random", **kw):
    from sg2.config import ChannelConfig, SentinelConfig
    return build_sentinel(SentinelConfig(
        encoder=EncoderConfig(name=name),
        channels=ChannelConfig(codec=False, vision=True)), **kw)


def test_sentinel_satisfies_protocol():
    assert isinstance(_sent(), Sentinel)


def test_degenerate_embedding_is_detected():
    """帧完全相同时无论什么阈值都达不到目标解码率 —— 必须报错。

    判据是**目标达成率**而非分布展布。回归:初版用 10-90 分位展布,
    在真实视频上误杀 —— 真实序列绝大多数相邻帧在镜头内(sim≈0.99),
    少数在镜头边界(sim≈0.79),偏斜是正常形状不是退化。实测 SigLIP2
    在真实帧上 sims∈[0.791,0.997] 判别力充足却被展布判据拒绝。
    """
    same = np.stack([np.full((16, 16, 3), 0.5, np.float32)] * 30)
    with pytest.raises(DegenerateEmbedding, match="目标解码比例"):
        _sent().calibrate_dedup(same)


def test_skewed_but_usable_distribution_is_accepted():
    """偏斜分布(镜头内密集 + 少数边界)必须被接受。"""
    d = _sent().calibrate_dedup(_shots(), keep_frac=0.3)
    assert 0.0 < d < 1.0


def test_dedup_calibration_is_monotone_in_keep_frac():
    cal = _shots()
    rates = []
    for kf in (0.2, 0.5, 0.8):
        s = _sent(); s.calibrate_dedup(cal, keep_frac=kf); s.reset()
        test = _shots(6, 5)
        for i, f in enumerate(test):
            s.score_frame(float(i), frame=f)
        rates.append(s.stats["decoded"])
    assert rates == sorted(rates), f"解码量未随 keep_frac 单调增: {rates}"


def test_decode_budget_caps_cost_amplification():
    """成本放大攻击:每帧都不同时,解码量必须被硬上限挡住。"""
    from sg2.config import ChannelConfig, SentinelConfig
    cfg = SentinelConfig(encoder=EncoderConfig(name="random"),
                         channels=ChannelConfig(codec=False, vision=True),
                         sample_fps=0.5, max_decode_fps_multiplier=2.0)
    s = build_sentinel(cfg)
    s.cfg.dedup_embedding_delta = 1e-9          # 去重完全失效(攻击者视角)
    for i in range(300):
        s.score_frame(i * 0.5, frame=RNG.random((16, 16, 3)).astype(np.float32))
    cap = cfg.sample_fps * 60.0 * cfg.max_decode_fps_multiplier
    assert s.stats["decoded"] <= cap * 4        # 150s -> 最多 3 个窗口


def test_sentinel_score_is_continuous_not_binary():
    """CUSUM 需要连续统计量。1-bit 信号做输入很糟。"""
    s = _sent(prototype=RNG.normal(0, 1, 128))
    vals = {s.score_frame(float(i), frame=f).score
            for i, f in enumerate(_shots(4, 3))}
    assert len(vals) > 2, "分数取值过少,退化成了准二值信号"


# ==================== CUSUM ====================

def test_cusum_detects_change():
    c = CusumController(cfg=CusumConfig(h=5.0, drift=0.1), mu0=0.0, mu1=1.0)
    nu, tau = 60, None
    for t in range(300):
        if c.update(RNG.normal(1.0 if t >= nu else 0.0, 1.0), float(t)):
            tau = t
            break
    assert tau is not None and 0 <= tau - nu < 60


def test_cusum_rarely_alarms_on_safe_stream():
    c = CusumController(cfg=CusumConfig(h=5.0, drift=0.1), mu0=0.0, mu1=1.0)
    n_fa = 0
    for t in range(3000):
        if c.update(RNG.normal(0, 1), float(t)):
            n_fa += 1
            c.reset()
    assert n_fa < 30, f"纯安全流误报 {n_fa} 次,平均间隔过短"


def test_cusum_refuses_inverted_means():
    with pytest.raises(ValueError, match="mu1|反方向"):
        CusumController(mu0=1.0, mu1=0.5)


def test_higher_threshold_gives_longer_expected_delay():
    lo = CusumController(cfg=CusumConfig(h=3.0)).expected_delay_windows()
    hi = CusumController(cfg=CusumConfig(h=9.0)).expected_delay_windows()
    assert hi > lo


def test_jitter_makes_schedule_nondeterministic():
    """确定性调度对按采样表定时投放的对手没有保证。"""
    c = CusumController(cfg=CusumConfig(jitter_s=0.4))
    ts = {round(c.next_sample_time(0.0, base_fps=0.5), 6) for _ in range(20)}
    assert len(ts) > 15, "抖动未生效"


def test_zero_jitter_is_deterministic():
    c = CusumController(cfg=CusumConfig(jitter_s=0.0))
    ts = {round(c.next_sample_time(0.0, base_fps=0.5), 6) for _ in range(10)}
    assert len(ts) == 1
