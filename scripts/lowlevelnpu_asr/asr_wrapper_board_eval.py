#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import visiong as vg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the visiong.ASR wrapper on precomputed feature tensors.")
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--acoustic-model", type=Path, required=True)
    parser.add_argument("--acoustic-vocab", type=Path, required=True)
    parser.add_argument("--p2c-model", type=Path, required=True)
    parser.add_argument("--p2c-input-vocab", type=Path, required=True)
    parser.add_argument("--p2c-output-vocab", type=Path, required=True)
    parser.add_argument("--feature-frames", type=int, default=600)
    parser.add_argument("--feature-bins", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--segment-topk", type=int, default=6)
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    parser.add_argument("--char-lm", type=Path, default=None)
    parser.add_argument("--char-lm-scale", type=float, default=0.0)
    parser.add_argument("--char-beam-size", type=int, default=6)
    parser.add_argument("--char-topk", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    cols = len(seq_b) + 1
    prev = list(range(cols))
    for i, item_a in enumerate(seq_a, start=1):
        curr = [i] + [0] * (cols - 1)
        for j, item_b in enumerate(seq_b, start=1):
            cost = 0 if item_a == item_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def summarize_latency(values_ms: Sequence[float]) -> Dict[str, float]:
    if not values_ms:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.feature_manifest.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]

    asr = vg.ASR(
        str(args.acoustic_model.resolve()),
        str(args.acoustic_vocab.resolve()),
        str(args.p2c_model.resolve()),
        str(args.p2c_input_vocab.resolve()),
        str(args.p2c_output_vocab.resolve()),
        feature_frames=args.feature_frames,
        feature_bins=args.feature_bins,
        max_tokens=args.max_tokens,
        segment_topk=args.segment_topk,
        candidate_temperature=args.candidate_temperature,
        char_lm_path="" if args.char_lm is None else str(args.char_lm.resolve()),
        char_lm_scale=args.char_lm_scale,
        char_beam_size=args.char_beam_size,
        char_topk=args.char_topk,
    )
    if not asr.is_initialized():
        raise RuntimeError("failed to initialize visiong.ASR")

    feature_root = args.feature_manifest.resolve().parent
    total_pinyin_err = 0
    total_pinyin_ref = 0
    total_char_err = 0
    total_char_ref = 0
    exact_matches = 0
    wall_times: List[float] = []
    acoustic_times: List[float] = []
    p2c_times: List[float] = []
    rerank_times: List[float] = []
    total_times: List[float] = []
    samples: List[Dict[str, object]] = []

    for index, row in enumerate(rows):
        feature = np.load((feature_root / row["relative_feature_path"]).resolve()).astype(np.float32, copy=False)
        start = time.perf_counter()
        result = asr.infer(feature)
        wall_ms = (time.perf_counter() - start) * 1000.0

        ref_text = row["text"]
        ref_pinyin_tokens = [token for token in row["pinyin_text"].strip().split() if token]
        pred_pinyin_tokens = list(result.pinyin_tokens)

        pinyin_error = edit_distance(ref_pinyin_tokens, pred_pinyin_tokens)
        char_error = edit_distance(list(ref_text), list(result.text))

        total_pinyin_err += pinyin_error
        total_pinyin_ref += len(ref_pinyin_tokens)
        total_char_err += char_error
        total_char_ref += len(ref_text)
        if result.text == ref_text:
            exact_matches += 1

        if index >= args.warmup_runs:
            wall_times.append(wall_ms)
            acoustic_times.append(float(result.acoustic_run_us) / 1000.0)
            p2c_times.append(float(result.p2c_run_us) / 1000.0)
            rerank_times.append(float(result.rerank_run_us) / 1000.0)
            total_times.append(float(result.total_run_us) / 1000.0)

        samples.append(
            {
                "id": row["id"],
                "reference": ref_text,
                "reference_pinyin": row["pinyin_text"],
                "pred_pinyin": result.pinyin,
                "prediction": result.text,
                "pinyin_error": int(pinyin_error),
                "char_error": int(char_error),
                "wall_ms": float(wall_ms),
                "acoustic_npu_ms": float(result.acoustic_run_us) / 1000.0,
                "p2c_npu_ms": float(result.p2c_run_us) / 1000.0,
                "rerank_ms": float(result.rerank_run_us) / 1000.0,
                "total_run_ms": float(result.total_run_us) / 1000.0,
                "used_tokens": int(result.used_tokens),
            }
        )

    summary = {
        "sample_count": len(rows),
        "sentence_accuracy": float(exact_matches / max(len(rows), 1)),
        "pinyin_token_error_rate": float(total_pinyin_err / max(total_pinyin_ref, 1)),
        "char_error_rate": float(total_char_err / max(total_char_ref, 1)),
        "has_char_lm": bool(asr.has_char_lm),
        "char_lm_scale": float(asr.char_lm_scale),
        "char_beam_size": int(asr.char_beam_size),
        "char_topk": int(asr.char_topk),
        "latency_warmup_runs": int(args.warmup_runs),
        "wall_latency_ms": summarize_latency(wall_times),
        "acoustic_npu_latency_ms": summarize_latency(acoustic_times),
        "p2c_npu_latency_ms": summarize_latency(p2c_times),
        "rerank_latency_ms": summarize_latency(rerank_times),
        "wrapper_total_latency_ms": summarize_latency(total_times),
        "samples": samples,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
