# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare IVE LK with OpenCV PyrLK on known translations.
使用已知平移量的图像，对比 IVE LK 与 OpenCV PyrLK 的准确性和耗时。
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

import cv2
import numpy as np
import visiong as vg


def elapsed(call, loops, signature=None):
    values = []
    observations = []
    for _ in range(loops):
        start = time.perf_counter()
        result = call()
        values.append((time.perf_counter() - start) * 1000)
        if signature is not None:
            observations.append(signature(result))
    timing = {"min_ms": min(values), "mean_ms": statistics.mean(values)}
    if observations:
        timing["unique_outputs"] = len(set(observations))
    return result, timing


def summarize(flow, valid, truth):
    flow = [list(map(float, point)) for point in flow]
    valid = [bool(value) for value in valid]
    error = [math.hypot(point[0] - truth[0], point[1] - truth[1]) for point in flow]
    accepted = sorted(value for value, keep in zip(error, valid) if keep)
    kept_flow = [point for point, keep in zip(flow, valid) if keep]
    return {
        "points": len(valid),
        "valid": sum(valid),
        "median_flow": (
            [statistics.median(p[c] for p in kept_flow) for c in (0, 1)]
            if kept_flow
            else None
        ),
        "mean_error_px": statistics.mean(accepted) if accepted else None,
        "p95_error_px": (
            accepted[min(len(accepted) - 1, int(0.95 * len(accepted)))]
            if accepted
            else None
        ),
        "accurate_within_1px": sum(
            keep and value <= 1 for keep, value in zip(valid, error)
        ),
        "flow": flow,
        "valid_mask": valid,
    }


def make_case(width, height, flat=False):
    generator = random.Random(1106)
    image = np.frombuffer(
        bytearray(generator.randrange(256) for _ in range(width * height)),
        dtype=np.uint8,
    )
    image = image.reshape(height, width)
    image = cv2.GaussianBlur(image, (5, 5), 0.8)
    mask = np.zeros_like(image)
    mask[48:-48, 48:-48] = 255
    points = cv2.goodFeaturesToTrack(image, 80, 0.02, 12, mask=mask).reshape(-1, 2)
    points = np.asarray(
        [[round(float(x)), round(float(y))] for x, y in points], dtype=np.float32
    )
    if flat:
        image[:] = 127
    return image, points


def compare(image, points, dx, dy, loops):
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    first = vg.ImageBuffer.from_numpy(image, "gray8")
    second = vg.ImageBuffer.from_numpy(shifted, "gray8")
    ive = vg.IVE()
    integer_points = [(int(x), int(y)) for x, y in points]
    entry = {
        "shape": list(image.shape),
        "translation": [dx, dy],
        "input_points": integer_points,
    }
    try:
        vectors, timing = elapsed(
            lambda: ive.lk_optical_flow(first, second, integer_points),
            loops,
            lambda output: tuple(
                (
                    v.status,
                    v.mv_x if v.status == 0 else 0,
                    v.mv_y if v.status == 0 else 0,
                )
                for v in output
            ),
        )
        displacement = [[v.mv_x / 128, v.mv_y / 128] for v in vectors]
        valid = [v.status == 0 for v in vectors]
        entry["ive"] = summarize(displacement, valid, (dx, dy))
        entry["ive"]["timing"] = timing
        entry["ive"]["raw_status"] = [v.status for v in vectors]
    except RuntimeError as error:
        entry["ive"] = {"error": str(error)}

    initial = points.reshape(-1, 1, 2)
    for level in (0, 3):

        def software():
            return cv2.calcOpticalFlowPyrLK(
                image,
                shifted,
                initial,
                None,
                winSize=(21, 21),
                maxLevel=level,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
                minEigThreshold=0.001,
            )

        (tracked, status, _), timing = elapsed(software, loops)
        displacement = tracked.reshape(-1, 2) - points
        key = "opencv_level_" + str(level)
        entry[key] = summarize(displacement, status.ravel().astype(bool), (dx, dy))
        entry[key]["timing"] = timing
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_flow_comparison.json"))
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be positive")
    cv2.setNumThreads(1)
    report = {
        "visiong": vg.__version__,
        "opencv": cv2.__version__,
        "loops": args.loops,
        "cases": [],
    }
    sizes = [(320, 240)] if args.quick else [(320, 240), (322, 242), (640, 480)]
    translations = (
        [(0, 0)]
        if args.quick
        else [(0, 0), (1, 0), (4, -3), (-7, 5), (12, 7), (24, -16)]
    )
    for width, height in sizes:
        for flat in (False, True):
            image, points = make_case(width, height, flat)
            for dx, dy in translations:
                if flat and (dx, dy) != (0, 0):
                    continue
                result = compare(image, points, dx, dy, args.loops)
                result["flat"] = flat
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                brief = {"size": [width, height], "flat": flat, "shift": [dx, dy]}
                for name in ("ive", "opencv_level_0", "opencv_level_3"):
                    brief[name] = {
                        key: value
                        for key, value in result[name].items()
                        if key
                        in ("error", "valid", "mean_error_px", "median_flow", "timing")
                    }
                print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
