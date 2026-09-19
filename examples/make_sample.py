"""生成样例 manifest。两个变体:clean 应全绿,dirty 必须被抓到。"""
import json, random, sys
sys.path.insert(0, '.')
from sg2 import bins

random.seed(7)
CATS = ["C1_violence", "C2_sexual", "C3_selfharm", "C4_extremism"]

def rec(i, safe, *, n_cuts, cut_bias=0.0, pool="eval", cat=None, dur=3600.0):
    needle = round(random.uniform(0.8, 3.0), 2)
    t0 = round(random.uniform(60, dur - 60), 1)
    cat = cat or random.choice(CATS)
    area = round(random.choice([0.002, 0.02, 0.12]), 4)
    cuts = []
    for k in range(n_cuts):
        # cut_bias>0 时把 unsafe 的剪辑点往后堆 -> 人为制造位置泄漏
        base = random.random()
        pos = min(0.99, base * (1 - cut_bias) + cut_bias) if not safe else base
        cuts.append({"t_s": round(pos * dur, 1),
                     "kind": "needle" if (k == 0 and not safe) else "benign",
                     "insert_id": f"x-{i}-{k}"})
    ratio = 0.0 if safe else needle / dur
    return {
        "id": f"sg2-{i:05d}",
        "source": {"kind": "synthetic", "license": "internal"},
        "media": {"duration_s": dur, "fps": 30.0, "resolution": [1280, 720],
                  "codec": "h264", "sha256": "0" * 64,
                  "reencode": {"crf": 23, "preset": "medium", "gop": 250}},
        "label": {"safe": safe, "categories": [] if safe else [cat],
                  "events": [] if safe else [{
                      "event_id": "e0", "category": cat,
                      "t_start_s": t0, "t_end_s": round(t0 + needle, 2),
                      "frame_start": int(t0 * 30), "frame_end": int((t0 + needle) * 30),
                      "severity": "high", "evidence_modality": ["pixel"],
                      "sufficient_alone": ["pixel"], "annotator_ids": ["a1", "a2"],
                      "agreement": 0.9}]},
        "axes": {
            "temporal_sparsity": {"needle_total_s": 0.0 if safe else needle,
                                  "video_duration_s": dur, "ratio": ratio,
                                  "bin": bins.temporal_sparsity_bin(ratio)},
            "perceptual_subtlety": {"min_evidence_pixel_area_frac": area,
                                    "evidence_contrast": 0.3,
                                    "evidence_span_frames": int(needle * 30),
                                    "requires_ocr": False,
                                    "bin": bins.perceptual_subtlety_bin(area)},
            "modality_locus": {"decisive": ["pixel"], "pixel_only_sufficient": True,
                               "bin": bins.modality_locus_bin(["pixel"])},
            "context_dependence": {"frame_alone_sufficient": True,
                                   "window_required_s": 0.0,
                                   "bin": bins.context_dependence_bin(0.0)},
        },
        "metadata_adversarial": {"title": "clip", "description": "",
                                 "asr_path": None, "chat_path": None,
                                 "condition": "aligned"},
        "splice": {"host_id": f"host-{i:04d}", "n_cuts": n_cuts, "cuts": cuts,
                   "audio_crossfade_ms": 40, "loudness_lufs": -23.0,
                   "reencoded_uniformly": True},
        "split": "calib" if pool == "calibration" else "test",
        "pool": pool,
    }

def write(path, *, dirty):
    rows = [{"__meta__": {"spec_version": "1.0"}}]
    for i in range(120):
        safe = i % 2 == 0
        pool = "calibration" if i % 3 == 0 else "eval"
        if dirty:
            # 缺陷1: unsafe 拼得更多  缺陷2: unsafe 剪辑点偏后
            n = 2 if safe else 6
            rows.append(rec(i, safe, n_cuts=n, cut_bias=0.0 if safe else 0.6, pool=pool))
        else:
            rows.append(rec(i, safe, n_cuts=4, pool=pool))
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

write("examples/manifest.clean.jsonl", dirty=False)
write("examples/manifest.dirty.jsonl", dirty=True)
print("已生成 clean / dirty 两份样例")
