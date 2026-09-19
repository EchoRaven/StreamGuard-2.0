#!/usr/bin/env python3
"""环境自检:按**约束的先后顺序**逐层检查,先卡住的先报。

顺序不是随意的。实测踩过的坑:先盯着显存选模型,装完才发现驱动根本不支持
那个 CUDA 轮子 —— 显存不够只是跑不了大模型,驱动不对是什么都跑不了。

    python scripts/doctor.py [--config configs/turing_local.yaml]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "✓", "!", "✗"
_problems: list[str] = []


def say(mark: str, title: str, detail: str = "") -> None:
    print(f"  {mark} {title}" + (f"  —  {detail}" if detail else ""))
    if mark == FAIL:
        _problems.append(title)


# ---------------------------------------------------------------- 1. 驱动

def check_driver() -> tuple[int, int] | None:
    print("\n[1] NVIDIA 驱动 —— 比显存更早的约束")
    if not shutil.which("nvidia-smi"):
        say(FAIL, "找不到 nvidia-smi", "无 GPU 或驱动未装")
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        say(FAIL, "读不到驱动版本")
        return None
    ver = out[0].strip()
    major = int(ver.split(".")[0])
    say(OK, f"驱动 {ver}")

    # CUDA minor 版本兼容: 11.x runtime 需驱动 >= 450.80.02;12.x 需 >= 525
    if major >= 525:
        say(OK, "可用 cu12x 轮子")
    elif major >= 450:
        say(WARN, "只能用 cu11x 轮子",
            "cu121/cu124 会 RuntimeError: driver too old。"
            "靠 CUDA minor 兼容装 cu118")
    else:
        say(FAIL, f"驱动 {ver} 过旧", "cu11x 也需要 >= 450.80.02")
    return (major, 0)


# ---------------------------------------------------------------- 2. torch

def check_torch():
    print("\n[2] PyTorch")
    try:
        import torch
    except ImportError:
        say(FAIL, "未安装 torch")
        return None
    say(OK, f"torch {torch.__version__}", f"CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        say(FAIL, "torch.cuda.is_available() == False",
            "多半是轮子的 CUDA 版本高于驱动所能支持的")
        return None
    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    arches = torch.cuda.get_arch_list()
    say(OK, f"{torch.cuda.device_count()} 张卡",
        ", ".join(f"sm_{a}{b}" for a, b in sorted(caps)))
    for a, b in sorted(caps):
        tag = f"sm_{a}{b}"
        if tag not in arches:
            say(FAIL, f"{tag} 不在该 torch 的编译架构表里", f"支持: {arches}")
    return sorted(caps)[0]


# ---------------------------------------------------------------- 3. 能力

def check_capability(cap):
    print("\n[3] 计算能力带来的限制")
    if cap is None:
        return
    sm = cap[0] * 10 + cap[1]
    if sm < 80:
        say(WARN, f"sm_{sm}: 无 bfloat16", "所有 dtype 必须用 float16")
        say(WARN, f"sm_{sm}: 无 FlashAttention-2", "attn_impl 用 sdpa")
    else:
        say(OK, f"sm_{sm}: bf16 与 FA2 均可用")

    try:
        import torch
        x = torch.randn(256, 256, device="cuda:0", dtype=torch.float16)
        say(OK if torch.isfinite(x @ x).all() else FAIL, "fp16 矩阵乘")
        free, total = torch.cuda.mem_get_info(0)
        gb = free / 1024 ** 3
        say(OK if gb > 8 else WARN, f"cuda:0 可用显存 {gb:.1f} GB",
            "" if gb > 16 else "8B 模型放不下,需 2B/4B 或量化")
    except Exception as e:                       # noqa: BLE001
        say(FAIL, "显存/算子检查失败", str(e)[:70])


# ---------------------------------------------------------------- 4. 依赖

def check_deps():
    print("\n[4] 依赖")
    import importlib
    for mod, why in [("transformers", "模型加载"), ("numpy", "核心"),
                     ("sklearn", "机制 C 线性头"), ("jsonschema", "格式校验"),
                     ("yaml", "配置"), ("imageio_ffmpeg", "合成与压缩域")]:
        try:
            m = importlib.import_module(mod)
            say(OK, f"{mod} {getattr(m, '__version__', '')}".strip())
        except ImportError:
            say(WARN if mod != "numpy" else FAIL, f"缺 {mod}", why)

    try:
        import torch
        import transformers
        tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        pv = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if tv >= (5, 0) and pv < (2, 5):
            say(FAIL, f"transformers {transformers.__version__} 需要 torch>=2.5",
                f"当前 {torch.__version__};升 torch 或降 transformers")
    except Exception:                            # noqa: BLE001
        pass

    say(OK if shutil.which("ffmpeg") else WARN, "ffmpeg",
        "" if shutil.which("ffmpeg") else "imageio-ffmpeg 自带静态版可替代")


# ---------------------------------------------------------------- 5. 配置

def check_config(path: str | None, cap):
    print("\n[5] 配置与硬件是否匹配")
    if not path:
        say(WARN, "未指定 --config", "跳过")
        return
    from sg2.config import SG2Config
    cfg = SG2Config.from_yaml(path)
    probs = cfg.validate_for_device(cap)
    if probs:
        for p in probs:
            say(FAIL, p)
    else:
        say(OK, f"{path} 与本机匹配")
    say(OK, f"midtier = {cfg.midtier.model_id}",
        f"{cfg.midtier.dtype} / {cfg.midtier.attn_impl}")


# ---------------------------------------------------------------- 6. 磁盘

def check_disk():
    print("\n[6] 磁盘")
    for p in ("/", "/data", "/data2"):
        if Path(p).exists():
            u = shutil.disk_usage(p)
            gb = u.free / 1024 ** 3
            say(OK if gb > 50 else WARN, f"{p} 可用 {gb:.0f} GB",
                "" if gb > 50 else "权重+数据可能放不下")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    print("=" * 62)
    print("SG2 环境自检 —— 按约束先后顺序")
    print("=" * 62)
    check_driver()
    cap = check_torch()
    check_capability(cap)
    check_deps()
    check_config(a.config, cap)
    check_disk()

    print("\n" + "=" * 62)
    if _problems:
        print(f"{FAIL} {len(_problems)} 个阻塞项:")
        for p in _problems:
            print(f"    - {p}")
        return 1
    print(f"{OK} 无阻塞项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
