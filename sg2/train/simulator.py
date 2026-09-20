"""流式训练数据生成:**复用运行时的采样器**。

Streaming 难训,不是一个问题而是七个:

| # | 问题 | 本模块怎么处理 |
| --- | --- | --- |
| 1 | 采样率不匹配 | **调用与运行时同一份采样逻辑**(抖动/去重/突发) |
| 2 | 因果性 | 窗口只含 [t-w, t] |
| 3 | KV 驱逐状态 | 序列足够长以**真的触发驱逐**,并按运行时规则截断 |
| 4 | 时序信用分配 | SFT 给不了 -> 产出**轨迹**供 RL |
| 5 | 非平稳 | 采**连续片段**而非 i.i.d. 窗口,事件上下文照常累积 |
| 6 | 标签粒度 | ν 来自合成拼接(SafeWatch 无时间戳) |
| 7 | 暴露偏差 | 支持用**当前策略**自己 rollout(DAgger 式) |

⚠️ 第 1 条是根子。训练用"均匀抽 16 帧"而部署是"0.5fps 滑窗+抖动+去重",
模型在训练时从没见过部署时的输入分布 —— 而这种不匹配**不会报错**,
只会让线上指标莫名低于离线。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..buffer import RingBuffer
from ..config import SG2Config
from ..cusum import CusumController


@dataclass
class SimTick:
    """模拟出的一个 tick。字段与运行时的 Tick 对齐。"""
    t_s: float
    frame_idx: int
    decoded: bool
    alarmed: bool
    escalated: bool
    window_t_s: tuple[float, ...]      # 该 tick 送进中间层的帧时刻
    evicted: int = 0
    in_event: bool = False             # 真值:此刻是否落在 [ν, ν_end]
    label: str = "clear"               # 该 tick 的目标动作


@dataclass
class SimConfig:
    """模拟参数。默认值**取自运行时配置**,不另起一套。"""
    sentinel_fps: float = 0.5
    jitter_s: float = 0.4
    burst_fps: float = 4.0
    rewind_s: float = 30.0
    window_s: float = 16.0
    max_window_frames: int = 2         # real 分辨率下 11GB 卡的实测上限
    dedup_skip_rate: float = 0.0       # 感知去重跳帧比例
    escalation_rate: float = 0.05
    partial_frac: float = 0.4          # 事件前多大比例时目标是 hold

    @classmethod
    def from_config(cls, cfg: SG2Config) -> "SimConfig":
        """⚠️ 从**同一份** SG2Config 派生,保证训练与部署一致。"""
        return cls(sentinel_fps=cfg.sentinel.sample_fps,
                   jitter_s=cfg.cusum.jitter_s,
                   burst_fps=cfg.cusum.burst_fps,
                   rewind_s=cfg.cusum.rewind_s,
                   window_s=cfg.midtier.context.vision_window_s,
                   escalation_rate=0.05)


class StreamSimulator:
    """把一条视频重放成**训练序列**。

    与 `Pipeline` 共用 `CusumController.next_sample_time` 与 `RingBuffer`,
    所以抖动、驱逐、突发采样的行为**逐字节一致**。
    """

    def __init__(self, cfg: SimConfig | None = None, *, seed: int = 0):
        self.cfg = cfg or SimConfig()
        self.seed = seed

    def sample_times(self, duration_s: float, *,
                     alarm_at: float | None = None) -> list[float]:
        """产生采样时刻 —— 用的是运行时那一份 `next_sample_time`。"""
        ctrl = CusumController(seed=self.seed)
        ctrl.cfg.jitter_s = self.cfg.jitter_s
        ctrl.cfg.burst_fps = self.cfg.burst_fps
        ts, t = [], 0.0
        while t < duration_s:
            ts.append(t)
            alarmed = alarm_at is not None and t >= alarm_at
            t = ctrl.next_sample_time(t, base_fps=self.cfg.sentinel_fps,
                                      alarmed=alarmed)
        return ts

    def simulate(self, *, duration_s: float, nu_s: float | None,
                 nu_end_s: float | None = None,
                 alarm_at: float | None = None) -> list[SimTick]:
        """重放一条流,产出带目标动作的 tick 序列。

        Args:
            nu_s / nu_end_s: 事件区间。None = 该流安全。
            alarm_at: CUSUM 告警时刻(模拟突发采样的切换点)。
        """
        rng = random.Random(self.seed)
        buf = RingBuffer(window_s=self.cfg.window_s)
        out: list[SimTick] = []
        end = nu_end_s if nu_end_s is not None else (
            nu_s + 2.0 if nu_s is not None else None)

        for i, t in enumerate(self.sample_times(duration_s,
                                                alarm_at=alarm_at)):
            # 感知去重:跳过的帧**不进 buffer 也不产 tick**,
            # 与运行时一致 —— 训练时不模拟去重,模型会以为帧是等距的
            if rng.random() < self.cfg.dedup_skip_rate:
                continue
            ev = buf.append(t, np.zeros((2, 2, 3), np.float32))

            alarmed = alarm_at is not None and t >= alarm_at
            escalated = alarmed or rng.random() < self.cfg.escalation_rate

            if escalated and alarmed:
                win = buf.resample(max(0.0, t - self.cfg.rewind_s), t,
                                   self.cfg.burst_fps)
            elif escalated:
                win = buf.rewind(t - 1e-6, t)
            else:
                win = []
            wt = tuple(f.t_s for f in win[-self.cfg.max_window_frames:])

            in_ev = (nu_s is not None and end is not None
                     and nu_s <= t <= end)
            out.append(SimTick(
                t_s=round(t, 4), frame_idx=i, decoded=True, alarmed=alarmed,
                escalated=escalated, window_t_s=wt, evicted=len(ev),
                in_event=in_ev, label=self._label(t, nu_s, end)))
        return out

    def _label(self, t: float, nu_s: float | None,
               end: float | None) -> str:
        """该 tick 的目标动作。

        ⚠️ "部分覆盖 -> hold" 是关键的一类:事件刚开始、证据还不全时,
        正确动作是**等**而不是猜。没有这类样本,模型学不会 hold
        (docs/08 §2.1)。
        """
        if nu_s is None or end is None:
            return "clear"
        if t < nu_s:
            return "clear"
        span = max(end - nu_s, 1e-9)
        if (t - nu_s) / span < self.cfg.partial_frac:
            return "hold"
        return "flag" if t <= end else "clear"


@dataclass
class RolloutTick(SimTick):
    """策略自己走出来的 tick。比 SimTick 多记**状态**。

    teacher forcing 下这些状态都是"假设前面全判对"推出来的;
    真实 rollout 下它们由模型自己的历史决定。
    """
    action: str = "clear"              # 策略实际输出的动作
    event_open: bool = False           # S2 里有没有累积证据
    event_tokens: int = 0              # S2 的占用
    n_since_open: int = 0              # 事件开着多少个 tick 了


def rollout(sim: "StreamSimulator", policy, *, duration_s: float,
            nu_s: float | None, nu_end_s: float | None = None,
            alarm_at: float | None = None) -> list[RolloutTick]:
    """让**策略自己驱动状态演化**,而不是假设它每步都对。

    这是 teacher forcing 的对照组。两者的差别在于 S2:
      teacher forcing —— 状态按"前面全判对"推
      rollout         —— 状态按策略**实际**做了什么推

    `policy(tick_state) -> action`,action ∈ hold/flag/clear/uncovered。
    """
    base = sim.simulate(duration_s=duration_s, nu_s=nu_s,
                        nu_end_s=nu_end_s, alarm_at=alarm_at)
    out: list[RolloutTick] = []
    event_open, ev_tokens, n_since = False, 0, 0

    for t in base:
        state = {"t_s": t.t_s, "in_event": t.in_event,
                 "escalated": t.escalated, "alarmed": t.alarmed,
                 "event_open": event_open, "event_tokens": ev_tokens,
                 "n_since_open": n_since, "target": t.label}
        act = policy(state) if t.escalated else "hold"

        # 状态按**实际动作**演化 —— 这正是 teacher forcing 拿不到的
        if act == "flag":
            event_open = True
            ev_tokens += 40
            n_since += 1
        elif act == "clear" and event_open:
            event_open = False
            ev_tokens = 0
            n_since = 0
        elif event_open:
            n_since += 1

        out.append(RolloutTick(
            t_s=t.t_s, frame_idx=t.frame_idx, decoded=t.decoded,
            alarmed=t.alarmed, escalated=t.escalated,
            window_t_s=t.window_t_s, evicted=t.evicted,
            in_event=t.in_event, label=t.label, action=act,
            event_open=event_open, event_tokens=ev_tokens,
            n_since_open=n_since))
    return out


def teacher_forced(sim: "StreamSimulator", **kw) -> list[RolloutTick]:
    """teacher forcing 对照:状态按**真值动作**演化。"""
    return rollout(sim, lambda st: st["target"], **kw)


def hindsight_relabel(ticks: list[RolloutTick]) -> list[tuple[dict, str]]:
    """DAgger 式回标:对策略**实际访问到的状态**给出正确动作。

    ⚠️ **专家是免费的** —— 合成拼接给了 ν,所以任意状态下"当时该做什么"
    直接算得出,不需要人工、也不需要 frontier 模型来当老师。
    这是 SafeWatch 无时间戳时合成流水线的又一个理由。
    """
    return [({"t_s": t.t_s, "event_open": t.event_open,
              "event_tokens": t.event_tokens, "n_since_open": t.n_since_open,
              "in_event": t.in_event, "alarmed": t.alarmed}, t.label)
            for t in ticks if t.escalated]


def state_shift(tf: list[RolloutTick], ro: list[RolloutTick]) -> dict:
    """量化 teacher forcing 与真实 rollout 的**状态分布偏移**。

    ⚠️ 偏移小 -> teacher forcing 够用,DAgger/RL 是过度设计。
       偏移大 -> 模型训练时从没见过它上线后会遇到的状态。
    先量再决定,别直接上 RL。
    """
    def dist(ts):
        n = len(ts) or 1
        return {"event_open": sum(t.event_open for t in ts) / n,
                "mean_event_tokens": float(np.mean([t.event_tokens for t in ts])),
                "mean_since_open": float(np.mean([t.n_since_open for t in ts]))}
    a, b = dist(tf), dist(ro)
    keys = set(a) | set(b)
    tvd = 0.5 * sum(abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-9)
                    for k in keys) / len(keys)
    mismatch = sum(1 for x, y in zip(tf, ro) if x.event_open != y.event_open)
    return {"teacher_forced": {k: round(v, 3) for k, v in a.items()},
            "rollout": {k: round(v, 3) for k, v in b.items()},
            "normalised_shift": round(tvd, 4),
            "event_state_mismatch": mismatch,
            "mismatch_rate": round(mismatch / max(len(tf), 1), 4)}


def resample_ticks(ticks: list[SimTick], *, target_mix: dict | None = None,
                   rng: random.Random | None = None) -> list[SimTick]:
    """按目标配比重采样 tick。

    ⚠️ **流式训练最根本的困难:正例极度稀缺。** 实测 0.5fps 下一个 4 秒的
    事件只落到 2 个 tick,而一条 120s 的流有 217 个 tick —— 自然采样下
    `flag`/`hold` 各只有 1 个,占 0.9%。按自然分布训练几乎学不到东西。

    但**不能简单过采样**:同一个 tick 重复多次会让模型记住那几帧。
    做法是以最受限的类定总量,不做有放回过采样(与 sft.py 一致)。
    """
    from collections import defaultdict
    target_mix = target_mix or {"flag": 0.25, "hold": 0.25, "clear": 0.5}
    rng = rng or random.Random(0)
    by = defaultdict(list)
    for t in ticks:
        by[t.label].append(t)
    feasible = [len(by.get(k, [])) / v for k, v in target_mix.items() if v > 0]
    if not feasible or min(feasible) == 0:
        return list(ticks)
    total = int(min(feasible))
    out: list[SimTick] = []
    for k, v in target_mix.items():
        pool = by.get(k, [])
        out.extend(rng.sample(pool, min(int(round(total * v)), len(pool))))
    rng.shuffle(out)
    return out


def simulate_many(sim: "StreamSimulator", specs: list[dict]) -> list[SimTick]:
    """重放多条流并汇总。

    **正例稀缺只能靠多条流解决,不能靠单条流过采样。**
    """
    out = []
    for i, sp in enumerate(specs):
        sim.seed = i
        out.extend(sim.simulate(**sp))
    return out


def sequence_stats(ticks: list[SimTick]) -> dict:
    """核对模拟是否真的复现了运行时的行为。"""
    from collections import Counter
    gaps = [b.t_s - a.t_s for a, b in zip(ticks, ticks[1:])]
    return {
        "n_ticks": len(ticks),
        "labels": dict(Counter(t.label for t in ticks)),
        "escalated": sum(t.escalated for t in ticks),
        "evictions": sum(t.evicted for t in ticks),
        "gap_mean": round(float(np.mean(gaps)), 3) if gaps else 0.0,
        "gap_std": round(float(np.std(gaps)), 3) if gaps else 0.0,
        "window_frames_max": max((len(t.window_t_s) for t in ticks),
                                 default=0),
        "has_hold": any(t.label == "hold" for t in ticks),
    }
