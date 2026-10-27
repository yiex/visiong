# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check additional IVE functions on controlled inputs.
使用受控输入检查其他 IVE 功能。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
import visiong as vg
from check_ive_components import interrupt_count

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("ive_advanced.json"))
args = parser.parse_args()
cv2.setNumThreads(1)
processor = vg.IVE()
width, height = 320, 240
blank = np.full((height, width), 80, dtype=np.uint8)
first = vg.ImageBuffer.from_numpy(blank, "gray8")
square = blank.copy()
square[80:120, 100:140] = 220
second = vg.ImageBuffer.from_numpy(square, "gray8")
binary = np.zeros((height, width), dtype=np.uint8)
binary[40:60, 50:80] = 255
binary[120:150, 160:200] = 255
binary_buffer = vg.ImageBuffer.from_numpy(binary, "gray8")
report = []


def check(name, call):
    count = interrupt_count()
    try:
        row = call()
        row.update(name=name, interrupts=interrupt_count() - count)
    except Exception as error:
        row = dict(
            name=name,
            error=str(error),
            passed=False,
            interrupts=interrupt_count() - count,
        )
    report.append(row)
    print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def ncc():
    result = processor.ncc(first, first)
    return dict(value=result, passed=abs(result - 1) < 1e-6)


def integral():
    results = []
    for mode, value in [(vg.IntegOutCtrl.SUM, 80), (vg.IntegOutCtrl.SQSUM, 6400)]:
        result = processor.integral(first, mode).to_numpy().copy()
        expected = width * height * value
        samples = [
            int(result[y, x]) for y, x in [(0, 0), (10, 15), (height - 1, width - 1)]
        ]
        truth = [
            (y + 1) * (x + 1) * value
            for y, x in [(0, 0), (10, 15), (height - 1, width - 1)]
        ]
        results.append(
            dict(
                mode=int(mode.value),
                samples=samples,
                expected=truth,
                passed=samples == truth,
            )
        )
    return dict(results=results, passed=all(row["passed"] for row in results))


def canny():
    empty = processor.canny(first, 100, 50).to_numpy().copy()
    edges = processor.canny(second, 100, 50).to_numpy().copy()
    count = int(cv2.countNonZero(edges[5:-5, 5:-5]))
    return dict(
        blank_edges=int(cv2.countNonZero(empty[5:-5, 5:-5])),
        square_edges=count,
        passed=cv2.countNonZero(empty[5:-5, 5:-5]) == 0 and count > 100,
    )


def lbp():
    result = processor.lbp(first).to_numpy().copy()
    return dict(
        nonzero=int(cv2.countNonZero(result[5:-5, 5:-5])),
        passed=cv2.countNonZero(result[5:-5, 5:-5]) == 0,
    )


def gradient():
    x, y = processor.norm_grad(second)
    a = x.to_numpy().copy()
    b = y.to_numpy().copy()
    mask = np.asarray(
        [
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1],
        ],
        dtype=np.float32,
    )
    horizontal = cv2.filter2D(square, cv2.CV_32F, mask) / 128
    vertical = cv2.filter2D(square, cv2.CV_32F, mask.T) / 128
    errors = []
    for actual, expected in [(a, horizontal), (b, vertical)]:
        peak = max(
            abs(int(actual[y, x]) - max(-128, min(127, float(expected[y, x]))))
            for y in range(5, height - 5)
            for x in range(5, width - 5)
        )
        errors.append(peak)
    return dict(
        x_dtype=a.dtype.str,
        y_dtype=b.dtype.str,
        maximum_errors=errors,
        passed=all(error <= 1 for error in errors),
    )


def sad():
    result, threshold = processor.sad(first, first, vg.SadMode.MB_4X4, 1)
    data = result.to_numpy().copy()
    return dict(
        shape=list(data.shape),
        dtype=data.dtype.str,
        nonzero=int(cv2.countNonZero(data)),
        passed=cv2.countNonZero(data) == 0,
    )


def ccl():
    result = processor.ccl(binary_buffer, min_area=1)
    return dict(count=len(result), passed=len(result) == 2)


def gmm():
    model = vg.IVEModel(width, height)
    foreground, background = processor.gmm(first, model, True)
    for _ in range(10):
        foreground, background = processor.gmm(first, model, False)
    quiet = foreground.to_numpy().copy()
    foreground, background = processor.gmm(second, model, False)
    moving = foreground.to_numpy().copy()
    count = int(cv2.countNonZero(moving[80:120, 100:140]))
    return dict(
        quiet=int(cv2.countNonZero(quiet)),
        moving=count,
        total=int(cv2.countNonZero(moving)),
        passed=cv2.countNonZero(quiet) == 0
        and count == 1600
        and cv2.countNonZero(moving) == 1600,
    )


def gmm2():
    model = vg.IVEModel(width, height)
    factor = vg.ImageBuffer.from_numpy(
        np.full((height, width), 0x0808, dtype=np.uint16), "u16c1"
    )
    foreground, background = processor.gmm2(first, factor, model, True)
    for _ in range(10):
        foreground, background = processor.gmm2(first, factor, model, False)
    quiet = foreground.to_numpy().copy()
    foreground, background = processor.gmm2(second, factor, model, False)
    moving = foreground.to_numpy().copy()
    count = int(cv2.countNonZero(moving[80:120, 100:140]))
    return dict(
        quiet=int(cv2.countNonZero(quiet)),
        moving=count,
        total=int(cv2.countNonZero(moving)),
        passed=cv2.countNonZero(quiet) == 0
        and count == 1600
        and cv2.countNonZero(moving) == 1600,
    )


def background_codebook():
    results = []
    for width, height in [(320, 240), (322, 242), (640, 480)]:
        background = np.full((height, width), 80, dtype=np.uint8)
        changed = background.copy()
        changed[80:120, 100:140] = 220
        first = vg.ImageBuffer.from_numpy(background, "gray8")
        second = vg.ImageBuffer.from_numpy(changed, "gray8")
        model = vg.IVEModel(width, height)
        for frame in range(60):
            assert processor.match_bg_model(first, model, frame) is None
        flag, image = processor.update_bg_model(first, model, 60)
        quiet = flag.to_numpy().copy()
        initial_background = image.to_numpy().copy()
        flag, image = processor.update_bg_model(second, model, 63)
        moving = flag.to_numpy().copy()
        final_background = image.to_numpy().copy()
        assert quiet.shape == (height, width) and cv2.countNonZero(quiet) == 0
        assert (
            cv2.countNonZero(moving) == 1600
            and cv2.countNonZero(moving[80:120, 100:140]) == 1600
        )
        assert (
            int(initial_background[90, 110]) == 80
            and int(final_background[90, 110]) == 80
        )
        results.append(
            dict(size=[width, height], foreground_pixels=int(cv2.countNonZero(moving)))
        )
    return dict(passed=True, results=results)


def repeated_models():
    results = []
    for mode in ("gmm", "gmm2"):
        signatures = []
        counts = []
        factor = vg.ImageBuffer.from_numpy(
            np.full((height, width), 0x0808, dtype=np.uint16), "u16c1"
        )
        for _ in range(5):
            model = vg.IVEModel(width, height)
            call = (
                (lambda image, first_frame: processor.gmm(image, model, first_frame))
                if mode == "gmm"
                else (
                    lambda image, first_frame: processor.gmm2(
                        image, factor, model, first_frame
                    )
                )
            )
            call(first, True)
            for _ in range(10):
                call(first, False)
            foreground, background = call(second, False)
            data = foreground.to_numpy().copy()
            signatures.append(hashlib.sha256(data.tobytes()).hexdigest())
            counts.append(int(cv2.countNonZero(data)))
        results.append(
            dict(mode=mode, unique=len(set(signatures)), foreground_counts=counts)
        )
    return dict(
        passed=all(
            row["unique"] == 1 and row["foreground_counts"] == [1600] * 5
            for row in results
        ),
        results=results,
    )


def gmm_geometry():
    results = []
    for mode in ("gmm", "gmm2"):
        for width, height in [(320, 240), (322, 242), (640, 480)]:
            base = np.full((height, width), 80, dtype=np.uint8)
            factor = vg.ImageBuffer.from_numpy(
                np.full((height, width), 0x0808, dtype=np.uint16), "u16c1"
            )
            for left, top, right, bottom in [
                (100, 80, 140, 120),
                (3, 4, 41, 37),
                (width - 45, height - 51, width - 2, height - 3),
            ]:
                image = vg.ImageBuffer.from_numpy(base, "gray8")
                model = vg.IVEModel(width, height)

                def call(image, initial):
                    if mode == "gmm":
                        return processor.gmm(image, model, initial)
                    return processor.gmm2(image, factor, model, initial)

                call(image, True)
                for _ in range(10):
                    call(image, False)
                changed = base.copy()
                changed[top:bottom, left:right] = 220
                foreground, background = call(
                    vg.ImageBuffer.from_numpy(changed, "gray8"), False
                )
                actual = foreground.to_numpy().copy()
                truth = np.zeros((height, width), dtype=np.uint8)
                truth[top:bottom, left:right] = 255
                differences = int(cv2.countNonZero(cv2.absdiff(actual, truth)))
                assert differences == 0
                results.append(
                    dict(
                        mode=mode,
                        size=[width, height],
                        rectangle=[left, top, right, bottom],
                        differences=differences,
                    )
                )
    return dict(passed=True, results=results)


for name, call in [
    ("ncc", ncc),
    ("integral_sum_sqsum", integral),
    ("canny", canny),
    ("lbp", lbp),
    ("norm_grad", gradient),
    ("sad", sad),
    ("ccl", ccl),
    ("gmm", gmm),
    ("gmm2", gmm2),
    ("background_codebook", background_codebook),
    ("repeated_models", repeated_models),
    ("gmm_geometry", gmm_geometry),
]:
    check(name, call)
if not all(row["passed"] for row in report):
    raise SystemExit(1)
