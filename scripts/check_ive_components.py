# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check IVE operators against independent image references.
使用独立图像参考结果检查 IVE 算子。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import visiong as vg


def interrupt_count():
    for line in Path("/proc/interrupts").read_text().splitlines():
        if line.rstrip().endswith(" rve"):
            return int(line.split()[1])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_components.json"))
    parser.add_argument("--loops", type=int, default=10)
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be positive")
    cv2.setNumThreads(1)
    processor = vg.IVE()
    report = []
    for width, height in [(320, 240), (322, 242), (640, 480)]:
        generator = random.Random(231)
        a = np.frombuffer(
            bytes(generator.randrange(256) for _ in range(width * height)), np.uint8
        ).reshape(height, width)
        b = np.frombuffer(
            bytes(generator.randrange(256) for _ in range(width * height)), np.uint8
        ).reshape(height, width)
        first = vg.ImageBuffer.from_numpy(a, "gray8")
        second = vg.ImageBuffer.from_numpy(b, "gray8")
        binary = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)[1]
        binary_buffer = vg.ImageBuffer.from_numpy(binary, "gray8")
        kernel = np.asarray(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=np.float32,
        )
        gaussian = cv2.filter2D(a, -1, kernel / 256)
        identity = [0] * 25
        identity[12] = 1
        operations = [
            ("dma", lambda: processor.dma(first), a, 0, 0),
            (
                "threshold",
                lambda: processor.threshold(first, 128),
                cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)[1],
                0,
                0,
            ),
            ("absdiff", lambda: processor.sub(first, second), cv2.absdiff(a, b), 0, 0),
            (
                "and",
                lambda: processor.logic_op(first, second, vg.LogicOp.AND),
                cv2.bitwise_and(a, b),
                0,
                0,
            ),
            (
                "or",
                lambda: processor.logic_op(first, second, vg.LogicOp.OR),
                cv2.bitwise_or(a, b),
                0,
                0,
            ),
            (
                "xor",
                lambda: processor.logic_op(first, second, vg.LogicOp.XOR),
                cv2.bitwise_xor(a, b),
                0,
                0,
            ),
            (
                "average",
                lambda: processor.add(first, second),
                cv2.addWeighted(a, 0.5, b, 0.5, 0),
                0,
                1,
            ),
            (
                "cast_identity",
                lambda: processor.cast_8bit_to_8bit(first, 0, 1, 1),
                a,
                0,
                0,
            ),
            (
                "map_invert",
                lambda: processor.map(first, list(range(255, -1, -1))),
                cv2.bitwise_not(a),
                0,
                0,
            ),
            ("filter_identity", lambda: processor.filter(first, identity), a, 2, 0),
            (
                "filter_gaussian",
                lambda: processor.filter(
                    first, kernel.astype(np.int8).ravel().tolist()
                ),
                gaussian,
                2,
                1,
            ),
            (
                "sobel_x",
                lambda: processor.sobel(first)[0],
                cv2.Sobel(a, cv2.CV_16S, 1, 0, ksize=3),
                2,
                0,
            ),
            (
                "sobel_y",
                lambda: processor.sobel(first)[1],
                cv2.Sobel(a, cv2.CV_16S, 0, 1, ksize=3),
                2,
                0,
            ),
            (
                "median_3",
                lambda: processor.ordered_stat_filter(
                    first, vg.OrdStatFilterMode.MEDIAN
                ),
                cv2.medianBlur(a, 3),
                2,
                0,
            ),
            (
                "max_3",
                lambda: processor.ordered_stat_filter(first, vg.OrdStatFilterMode.MAX),
                cv2.dilate(a, np.ones((3, 3), dtype=np.uint8)),
                2,
                0,
            ),
            (
                "min_3",
                lambda: processor.ordered_stat_filter(first, vg.OrdStatFilterMode.MIN),
                cv2.erode(a, np.ones((3, 3), dtype=np.uint8)),
                2,
                0,
            ),
        ]
        for size in (3, 5):
            mask = np.ones((size, size), dtype=np.uint8)
            operations.extend(
                [
                    (
                        "dilate_binary_" + str(size),
                        lambda size=size: processor.dilate(binary_buffer, size),
                        cv2.dilate(binary, mask),
                        2,
                        0,
                    ),
                    (
                        "erode_binary_" + str(size),
                        lambda size=size: processor.erode(binary_buffer, size),
                        cv2.erode(binary, mask),
                        2,
                        0,
                    ),
                ]
            )
        for name, call, expected, border, tolerance in operations:
            start_irq = interrupt_count()
            signatures = []
            durations = []
            try:
                worst = 0
                bad = 0
                for _ in range(args.loops):
                    start = time.perf_counter()
                    actual = call().to_numpy().copy()
                    durations.append((time.perf_counter() - start) * 1000)
                    signatures.append(hashlib.sha256(actual.tobytes()).hexdigest())
                    left = actual[border:-border, border:-border] if border else actual
                    right = (
                        expected[border:-border, border:-border] if border else expected
                    )
                    difference = cv2.absdiff(left, right)
                    peak = int(difference.max())
                    worst = max(worst, peak)
                    bad += int(
                        cv2.countNonZero(cv2.compare(difference, tolerance, cv2.CMP_GT))
                    )
                row = dict(
                    name=name,
                    size=[width, height],
                    passed=bad == 0
                    and len(set(signatures)) == 1
                    and interrupt_count() > start_irq,
                    differing_pixels=bad,
                    max_difference=worst,
                    tolerance=tolerance,
                    excluded_border=border,
                    unique=len(set(signatures)),
                    loops=args.loops,
                    min_ms=min(durations),
                    mean_ms=sum(durations) / len(durations),
                    interrupts=interrupt_count() - start_irq,
                )
            except Exception as error:
                row = dict(
                    name=name, size=[width, height], passed=False, error=str(error)
                )
            report.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        expected_histogram = collections.Counter(a.tobytes())
        start_irq = interrupt_count()
        actual_histogram = processor.hist(first)
        row = dict(
            name="histogram",
            size=[width, height],
            passed=all(
                actual_histogram[index] == expected_histogram[index]
                for index in range(256)
            ),
            sum=sum(actual_histogram),
            interrupts=interrupt_count() - start_irq,
        )
        report.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(row["passed"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
