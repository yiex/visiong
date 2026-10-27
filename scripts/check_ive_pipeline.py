# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate feature detection, hardware pyramids and frame-to-frame LK.
验证特征检测、硬件金字塔及帧间 LK。
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import visiong as vg
from check_ive_components import interrupt_count
from compare_ive_optical_flow import make_case

cv2.setNumThreads(1)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("ive_pipeline.json"))
parser.add_argument("--repeat", type=int, default=1000)
parser.add_argument("--resize-cycles", type=int, default=20)
args = parser.parse_args()
if args.repeat <= 0 or args.resize_cycles <= 0:
    parser.error("--repeat and --resize-cycles must be positive")
processor = vg.IVE()
report = []
frames = []
expected_corners = []
for width, height in [(320, 240), (322, 242), (640, 480)]:
    image, features = make_case(width, height)
    first = vg.ImageBuffer.from_numpy(image, "gray8")
    frames.append(first)
    count = interrupt_count()
    corners = processor.st_corner(first, max_corners=80, min_dist=12)
    repeated = [
        tuple(processor.st_corner(first, max_corners=80, min_dist=12)) for _ in range(5)
    ]
    valid = bool(corners) and all(
        0 <= x < width and 0 <= y < height for x, y in corners
    )
    spacing = min(
        [
            math.hypot(x - xx, y - yy)
            for i, (x, y) in enumerate(corners)
            for xx, yy in corners[i + 1 :]
        ]
        or [0]
    )
    row = dict(
        name="st_corner",
        size=[width, height],
        count=len(corners),
        inside=valid,
        minimum_spacing=spacing,
        unique=len(set(repeated)),
        interrupts=interrupt_count() - count,
        passed=valid and spacing >= 12 and len(set(repeated)) == 1,
    )
    report.append(row)
    expected_corners.append(tuple(corners))
    print(json.dumps(row), flush=True)
    mask = (
        np.asarray(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=np.float32,
        )
        / 256
    )
    count = interrupt_count()
    pyramid = processor.create_pyramid(first, 4)
    errors = []
    for index in range(1, len(pyramid)):
        previous = pyramid[index - 1].to_numpy().copy()
        actual = pyramid[index].to_numpy().copy()
        expected = cv2.filter2D(previous, -1, mask)[::2, ::2][
            : actual.shape[0], : actual.shape[1]
        ]
        errors.append(int(cv2.absdiff(actual[3:-3, 3:-3], expected[3:-3, 3:-3]).max()))
    row = dict(
        name="pyramid",
        size=[width, height],
        shapes=[list(level.to_numpy().shape) for level in pyramid],
        maximum_errors=errors,
        passed=len(pyramid) == 4 and all(error <= 1 for error in errors),
        interrupts=interrupt_count() - count,
    )
    report.append(row)
    print(json.dumps(row), flush=True)
    points = [(int(x), int(y)) for x, y in features]
    translated = []
    count = interrupt_count()
    for dx, dy in [(0, 0), (4, -3), (-7, 5), (12, 7), (1, 0)] * 10:
        matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(
            image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
        )
        second = vg.ImageBuffer.from_numpy(shifted, "gray8")
        result = processor.lk_optical_flow(first, second, points)
        errors = [
            math.hypot(vector.mv_x / 128 - dx, vector.mv_y / 128 - dy)
            for vector in result
            if vector.status == 0
        ]
        translated.append(
            dict(
                shift=[dx, dy],
                valid=len(errors),
                max_error=max(errors) if errors else None,
            )
        )
    row = dict(
        name="changing_frames_lk",
        size=[width, height],
        calls=len(translated),
        results=translated,
        passed=all(
            result["valid"] == 80 and result["max_error"] <= 0.1
            for result in translated
        ),
        interrupts=interrupt_count() - count,
    )
    report.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "results"}), flush=True)
    if width == 320:
        count = interrupt_count()
        signatures = set()
        passed = True
        for _ in range(args.repeat):
            result = processor.lk_optical_flow(first, first, points)
            signature = tuple((v.status, v.mv_x, v.mv_y) for v in result)
            signatures.add(signature)
            passed = (
                passed
                and len(result) == 80
                and all(status == 0 and x == 0 and y == 0 for status, x, y in signature)
            )
        row = dict(
            name="stationary_lk",
            calls=args.repeat,
            points=80,
            unique=len(signatures),
            passed=passed and len(signatures) == 1,
            interrupts=interrupt_count() - count,
        )
        report.append(row)
        print(json.dumps(row), flush=True)
        counts = [1, 2, 7, 8, 9, 16, 32, 64, 80, 128, 500]
        for size in counts:
            result = processor.lk_optical_flow(first, first, [points[0]] * size)
            assert len(result) == size and all(
                v.status == 0 and v.mv_x == 0 and v.mv_y == 0 for v in result
            )
        report.append(dict(name="point_counts", counts=counts, passed=True))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
count = interrupt_count()
failures = []
for cycle in range(args.resize_cycles):
    for image, expected in zip(frames, expected_corners):
        for repeat in range(3):
            corners = tuple(processor.st_corner(image, max_corners=80, min_dist=12))
            if corners != expected or len(set(corners)) != len(corners):
                failures.append(
                    dict(cycle=cycle, size=[image.width, image.height], repeat=repeat)
                )
row = dict(
    name="resized_corner_state",
    calls=args.resize_cycles * 9,
    failures=failures,
    passed=not failures,
    interrupts=interrupt_count() - count,
)
report.append(row)
print(json.dumps(row), flush=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
if not all(row["passed"] and row.get("interrupts", 1) > 0 for row in report):
    raise SystemExit(1)
