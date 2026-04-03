#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import visiong as vg

from common import feature_tensor_from_audio, read_wav_pcm16, softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS-CNN keyword spotting with visiong.LowLevelNPU on the board.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument(
        "--frontend-backend",
        choices=("auto", "python", "native"),
        default="auto",
        help="auto: use native frontend when available, otherwise python; python: force the numpy frontend; native: require the native visiong frontend",
    )
    parser.add_argument(
        "--input-strategy",
        choices=("compat", "array-only", "packed-fallback"),
        default="compat",
        help="compat: use packed fallback on old runtimes; array-only: always use set_input_array(); packed-fallback: always pack when possible",
    )
    parser.add_argument("--strict-array-only", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()
    if args.strict_array_only:
        args.input_strategy = "array-only"
    return args


def read_labels(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(manifest_path: Path) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append((manifest_path.parent / row["relative_path"], row["label"]))
    return rows


def tensor_info_to_dict(tensor_info) -> Dict[str, object]:
    return {
        "index": int(tensor_info.index),
        "name": str(tensor_info.name),
        "dims": [int(value) for value in tensor_info.dims],
        "format": str(tensor_info.format),
        "type": str(tensor_info.type),
        "quant_type": str(tensor_info.quant_type),
        "zero_point": int(tensor_info.zero_point),
        "scale": float(tensor_info.scale),
        "num_elements": int(tensor_info.num_elements),
        "size_bytes": int(tensor_info.size_bytes),
        "size_with_stride_bytes": int(tensor_info.size_with_stride_bytes),
        "w_stride": int(tensor_info.w_stride),
        "h_stride": int(tensor_info.h_stride),
        "pass_through": bool(tensor_info.pass_through),
    }


def adapt_input_layout(features_nchw: np.ndarray, input_info: Dict[str, object]) -> np.ndarray:
    tensor_format = str(input_info.get("format", "")).upper()
    if tensor_format == "NHWC":
        return np.transpose(features_nchw, (0, 2, 3, 1)).astype(np.float32, copy=False)
    return features_nchw.astype(np.float32, copy=False)


def build_pass_through_payload(features_nchw: np.ndarray, input_info: Dict[str, object]) -> Optional[bytes]:
    tensor_type = str(input_info.get("type", "")).upper()
    if not bool(input_info.get("pass_through", False)):
        return None
    if tensor_type not in ("INT8", "UINT8"):
        return None

    scale = float(input_info.get("scale", 1.0) or 1.0)
    zero_point = int(input_info.get("zero_point", 0))
    tensor_format = str(input_info.get("format", "")).upper()
    w_stride = int(input_info.get("w_stride", 0))
    dtype = np.int8 if tensor_type == "INT8" else np.uint8

    if tensor_format == "NHWC":
        logical = np.transpose(features_nchw, (0, 2, 3, 1)).astype(np.float32, copy=False)
        batch, height, width, channels = logical.shape
        padded_width = max(w_stride, width)
        packed = np.zeros((batch, height, padded_width, channels), dtype=dtype)
        quant = np.round(logical / scale + zero_point)
        if tensor_type == "INT8":
            quant = np.clip(quant, -128, 127).astype(np.int8)
        else:
            quant = np.clip(quant, 0, 255).astype(np.uint8)
        packed[:, :, :width, :] = quant
        return packed.tobytes()

    if tensor_format == "NCHW":
        logical = features_nchw.astype(np.float32, copy=False)
        batch, channels, height, width = logical.shape
        padded_width = max(w_stride, width)
        packed = np.zeros((batch, channels, height, padded_width), dtype=dtype)
        quant = np.round(logical / scale + zero_point)
        if tensor_type == "INT8":
            quant = np.clip(quant, -128, 127).astype(np.int8)
        else:
            quant = np.clip(quant, 0, 255).astype(np.uint8)
        packed[:, :, :, :width] = quant
        return packed.tobytes()

    return None


def choose_input_strategy(
    npu: vg.LowLevelNPU,
    features_nchw: np.ndarray,
    input_info: Dict[str, object],
    input_strategy: str,
) -> str:
    if input_strategy == "array-only":
        features = adapt_input_layout(features_nchw, input_info)
        npu.set_input_array(0, features, quantize_if_needed=True, zero_pad=True, sync_to_device=True)
        return "set_input_array"

    payload = build_pass_through_payload(features_nchw, input_info)
    if payload is not None and input_strategy in ("compat", "packed-fallback"):
        npu.set_input_bytes(0, payload, zero_pad=True, sync_to_device=True)
        return "packed_pass_through_fallback"

    features = adapt_input_layout(features_nchw, input_info)
    npu.set_input_array(0, features, quantize_if_needed=True, zero_pad=True, sync_to_device=True)
    return "set_input_array"


def summarize_latency(values: List[float], warmup_count: int) -> Dict[str, float]:
    trimmed = values[min(max(warmup_count, 0), len(values)) :]
    if not trimmed:
        trimmed = values[:]
    if not trimmed:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    array = np.asarray(trimmed, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(np.max(array)),
    }


def infer_single(
    npu: vg.LowLevelNPU,
    wav_path: Path,
    labels: List[str],
    input_info: Dict[str, object],
    frontend_backend: str,
    frontend,
    input_strategy: str,
) -> Dict[str, object]:
    start_total = time.perf_counter()
    start_frontend = time.perf_counter()
    audio = read_wav_pcm16(wav_path)
    if frontend_backend == "native":
        features_nchw = np.asarray(frontend.compute_from_float(audio, output_format="nchw"), dtype=np.float32)
    else:
        features_nchw = feature_tensor_from_audio(audio).astype(np.float32)
    after_frontend = time.perf_counter()
    frontend_ms = (after_frontend - start_frontend) * 1000.0

    start_input = time.perf_counter()
    actual_input_strategy = choose_input_strategy(npu, features_nchw, input_info, input_strategy)
    after_input = time.perf_counter()
    input_prepare_ms = (after_input - start_input) * 1000.0

    start_run_wall = time.perf_counter()
    npu.run(sync_outputs=True)
    after_run_wall = time.perf_counter()
    run_wall_ms = (after_run_wall - start_run_wall) * 1000.0

    start_postprocess = time.perf_counter()
    logits = np.asarray(
        npu.output_array(0, dequantize_if_needed=True, sync_from_device=False),
        dtype=np.float32,
    ).reshape(-1)
    probs = softmax(logits)
    pred_index = int(np.argmax(probs))
    after_postprocess = time.perf_counter()
    postprocess_ms = (after_postprocess - start_postprocess) * 1000.0
    total_ms = (after_postprocess - start_total) * 1000.0
    return {
        "wav": str(wav_path),
        "pred": labels[pred_index],
        "confidence": float(np.max(probs)),
        "frontend_ms": frontend_ms,
        "input_prepare_ms": input_prepare_ms,
        "npu_ms": float(npu.last_run_us) / 1000.0,
        "run_wall_ms": run_wall_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
        "input_strategy": actual_input_strategy,
        "frontend_backend": frontend_backend,
        "logits": logits.tolist(),
    }


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    labels = read_labels(args.labels.resolve())
    npu = vg.LowLevelNPU(str(model_path))
    if not npu.is_initialized():
        raise RuntimeError("LowLevelNPU failed to initialize from {}".format(model_path))

    sdk_versions = npu.sdk_versions()
    input_info = tensor_info_to_dict(npu.input_tensor(0))
    output_info = tensor_info_to_dict(npu.output_tensor(0))

    native_frontend_available = hasattr(vg, "KwsLogMelFrontend")
    selected_frontend_backend = args.frontend_backend
    frontend = None
    if selected_frontend_backend == "auto":
        selected_frontend_backend = "native" if native_frontend_available else "python"
    if selected_frontend_backend == "native":
        if not native_frontend_available:
            raise RuntimeError("Native frontend requested but visiong.KwsLogMelFrontend is not available in this runtime.")
        frontend = vg.KwsLogMelFrontend()

    print("SDK versions: api={api} driver={driver}".format(api=sdk_versions["api"], driver=sdk_versions["driver"]))
    print("Input tensor: {}".format(input_info))
    print("Output tensor: {}".format(output_info))
    print("Warmup count for latency stats: {}".format(max(args.warmup_count, 0)))
    print("Frontend backend mode: {}".format(selected_frontend_backend))
    print("Input policy mode: {}".format(args.input_strategy))

    if args.wav is not None:
        result = infer_single(
            npu,
            args.wav.resolve(),
            labels,
            input_info,
            selected_frontend_backend,
            frontend,
            args.input_strategy,
        )
        print("wav={}".format(result["wav"]))
        print("pred={} confidence={:.4f}".format(result["pred"], result["confidence"]))
        print(
            "frontend_ms={:.3f} input_prepare_ms={:.3f} npu_ms={:.3f} postprocess_ms={:.3f} total_ms={:.3f} frontend_backend={} input_strategy={}".format(
                result["frontend_ms"],
                result["input_prepare_ms"],
                result["npu_ms"],
                result["postprocess_ms"],
                result["total_ms"],
                result["frontend_backend"],
                result["input_strategy"],
            )
        )
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0

    if args.manifest is None:
        raise ValueError("Either --wav or --manifest must be provided.")

    entries = load_manifest(args.manifest.resolve())
    if args.limit > 0:
        entries = entries[: args.limit]
    print("Evaluating {} utterances".format(len(entries)))

    preview: List[Dict[str, object]] = []
    correct = 0
    frontend_ms_values: List[float] = []
    input_prepare_ms_values: List[float] = []
    npu_ms_values: List[float] = []
    run_wall_ms_values: List[float] = []
    postprocess_ms_values: List[float] = []
    total_ms_values: List[float] = []
    strategy_counts: Dict[str, int] = {}
    frontend_backend_counts: Dict[str, int] = {}

    for wav_path, truth in entries:
        result = infer_single(
            npu,
            wav_path,
            labels,
            input_info,
            selected_frontend_backend,
            frontend,
            args.input_strategy,
        )
        is_correct = result["pred"] == truth
        correct += int(is_correct)
        frontend_ms_values.append(float(result["frontend_ms"]))
        input_prepare_ms_values.append(float(result["input_prepare_ms"]))
        npu_ms_values.append(float(result["npu_ms"]))
        run_wall_ms_values.append(float(result["run_wall_ms"]))
        postprocess_ms_values.append(float(result["postprocess_ms"]))
        total_ms_values.append(float(result["total_ms"]))
        strategy = str(result["input_strategy"])
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        backend_name = str(result["frontend_backend"])
        frontend_backend_counts[backend_name] = frontend_backend_counts.get(backend_name, 0) + 1
        if len(preview) < args.preview_count:
            preview.append(
                {
                    "wav": wav_path.name,
                    "truth": truth,
                    "pred": result["pred"],
                    "confidence": round(float(result["confidence"]), 4),
                    "frontend_ms": round(float(result["frontend_ms"]), 3),
                    "input_prepare_ms": round(float(result["input_prepare_ms"]), 3),
                    "npu_ms": round(float(result["npu_ms"]), 3),
                    "postprocess_ms": round(float(result["postprocess_ms"]), 3),
                    "total_ms": round(float(result["total_ms"]), 3),
                    "frontend_backend": backend_name,
                    "input_strategy": strategy,
                    "correct": is_correct,
                }
            )

    accuracy = correct / max(len(entries), 1)
    avg_frontend_ms = float(np.mean(frontend_ms_values)) if frontend_ms_values else 0.0
    avg_input_prepare_ms = float(np.mean(input_prepare_ms_values)) if input_prepare_ms_values else 0.0
    avg_npu_ms = float(np.mean(npu_ms_values)) if npu_ms_values else 0.0
    avg_run_wall_ms = float(np.mean(run_wall_ms_values)) if run_wall_ms_values else 0.0
    avg_postprocess_ms = float(np.mean(postprocess_ms_values)) if postprocess_ms_values else 0.0
    avg_total_ms = float(np.mean(total_ms_values)) if total_ms_values else 0.0
    latency_stats = {
        "warmup_count": min(max(args.warmup_count, 0), len(entries)),
        "frontend_ms": summarize_latency(frontend_ms_values, args.warmup_count),
        "input_prepare_ms": summarize_latency(input_prepare_ms_values, args.warmup_count),
        "npu_ms": summarize_latency(npu_ms_values, args.warmup_count),
        "run_wall_ms": summarize_latency(run_wall_ms_values, args.warmup_count),
        "postprocess_ms": summarize_latency(postprocess_ms_values, args.warmup_count),
        "total_ms": summarize_latency(total_ms_values, args.warmup_count),
    }

    print("Preview predictions:")
    for item in preview:
        print(
            "  {wav}: truth={truth} pred={pred} conf={confidence:.4f} frontend_ms={frontend_ms:.3f} input_prepare_ms={input_prepare_ms:.3f} npu_ms={npu_ms:.3f} postprocess_ms={postprocess_ms:.3f} total_ms={total_ms:.3f} frontend_backend={frontend_backend} input_strategy={input_strategy} correct={correct}".format(
                **item
            )
        )
    print("accuracy={:.4f}".format(accuracy))
    print("avg_frontend_ms={:.3f}".format(avg_frontend_ms))
    print("avg_input_prepare_ms={:.3f}".format(avg_input_prepare_ms))
    print("avg_npu_ms={:.3f}".format(avg_npu_ms))
    print("avg_run_wall_ms={:.3f}".format(avg_run_wall_ms))
    print("avg_postprocess_ms={:.3f}".format(avg_postprocess_ms))
    print("avg_total_ms={:.3f}".format(avg_total_ms))
    print("frontend_backend_counts={}".format(frontend_backend_counts))
    print("input_strategy_counts={}".format(strategy_counts))
    print(
        "post_warmup_total_ms_mean={mean_ms:.3f} median={median_ms:.3f} p95={p95_ms:.3f} max={max_ms:.3f}".format(
            **latency_stats["total_ms"]
        )
    )

    if args.summary_json is not None:
        payload = {
            "model": str(model_path),
            "labels": labels,
            "sample_count": len(entries),
            "accuracy": accuracy,
            "avg_frontend_ms": avg_frontend_ms,
            "avg_input_prepare_ms": avg_input_prepare_ms,
            "avg_npu_ms": avg_npu_ms,
            "avg_run_wall_ms": avg_run_wall_ms,
            "avg_postprocess_ms": avg_postprocess_ms,
            "avg_total_ms": avg_total_ms,
            "frontend_backend_mode": selected_frontend_backend,
            "frontend_backend_counts": frontend_backend_counts,
            "input_strategy_counts": strategy_counts,
            "input_strategy_mode": args.input_strategy,
            "latency_stats": latency_stats,
            "sdk_versions": {"api": sdk_versions["api"], "driver": sdk_versions["driver"]},
            "input_tensor": input_info,
            "output_tensor": output_info,
            "preview": preview,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("summary_json={}".format(args.summary_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
