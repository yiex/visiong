# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark one model per process, excluding initialization from timing.
每个进程测试一个模型，计时不包含模型初始化。
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import visiong as vg


def stats(values):
    values = sorted(values)
    return {
        "min_ms": min(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(0.95 * len(values)))],
        "fps": 1000 / statistics.mean(values),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--type", default="yolo26", choices=["yolo26", "yolo11"])
    p.add_argument("--image", default="bus.jpg")
    p.add_argument("--labels", default="coco_80_labels_list.txt")
    p.add_argument("--loops", type=int, default=60)
    p.add_argument("--output", default="benchmark.json")
    p.add_argument("--low-level", action="store_true")
    args = p.parse_args()
    if args.loops < 1:
        p.error("--loops must be positive")
    img = vg.ImageBuffer.load(args.image)
    result = {
        "model": args.model,
        "type": args.type,
        "image": args.image,
        "loops": args.loops,
        "library": vg.__version__,
        "module": vg.__file__,
    }
    if args.low_level:
        model = vg.LowLevelNPU(args.model)
        result["sdk"] = model.sdk_versions()
        result["outputs"] = [
            {
                name: getattr(t, name)
                for name in [
                    "name",
                    "dims",
                    "format",
                    "type",
                    "scale",
                    "zero_point",
                    "size_bytes",
                    "size_with_stride_bytes",
                    "w_stride",
                    "h_stride",
                ]
            }
            for t in model.output_tensors()
        ]
        model.set_input_image(0, img, color_order="rgb", keep_aspect=True, pad_value=114)
        for _ in range(5):
            model.run()
        times = []
        wall = []
        for _ in range(args.loops):
            start = time.perf_counter()
            model.run()
            wall.append((time.perf_counter() - start) * 1000)
            times.append(model.last_run_us / 1000)
        result["rknn_run"] = stats(times)
        result["run_with_sync"] = stats(wall)
    else:
        model = vg.NPU(args.type, args.model, args.labels, box=0.25)
        result["input"] = [model.model_width, model.model_height]
        for _ in range(5):
            detections = model.infer(img)
        times = []
        for _ in range(args.loops):
            start = time.perf_counter()
            detections = model.infer(img)
            times.append((time.perf_counter() - start) * 1000)
        result["inference"] = stats(times)
        result["detections"] = [
            {"class_id": d.class_id, "label": d.label, "score": d.score, "box": list(d.box)}
            for d in detections
        ]
        for d in detections:
            img.draw_rectangle(d.box, (0, 255, 0), 2)
        img.save(str(Path(args.output).with_suffix(".png")))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
