# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate extended IVE controls against independent numeric references.
使用独立数值参考检查扩展 IVE 控制参数。

Run with the regular target library. Missing interfaces and unresolved numeric
conventions are reported as unverified. Interrupts supplement numeric checks.
在目标常规库上运行；接口缺失和未确定的数值约定记为未验证，中断计数仅作为补充。
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time


np = None
cv2 = None
vg = None


def interrupt_count():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def image(data, pixel_format="gray8"):
    return vg.ImageBuffer.from_numpy(data, pixel_format)


def array(buffer):
    return buffer.to_numpy().copy()


def checksum(data):
    return hashlib.sha256(data.tobytes()).hexdigest()


def compare(actual, expected):
    if list(actual.shape) != list(expected.shape):
        return dict(shape_matches=False, different=None, maximum=None)
    if (
        actual.dtype.str == expected.dtype.str
        and actual.dtype.kind in "uib"
        and actual.tobytes() == expected.tobytes()
    ):
        return dict(shape_matches=True, different=0, maximum=0)
    differences = [
        abs(int(left) - int(right))
        for left, right in zip(actual.ravel().tolist(), expected.ravel().tolist())
    ]
    return dict(
        shape_matches=True,
        different=sum(value != 0 for value in differences),
        maximum=max(differences, default=0),
    )


def sad_reference(first, second, block):
    height, width = len(first), len(first[0])
    return [
        [
            sum(
                abs(first[y][x] - second[y][x])
                for y in range(top, top + block)
                for x in range(left, left + block)
            )
            for left in range(0, width, block)
        ]
        for top in range(0, height, block)
    ]


def signed_byte(value):
    return value - 256 if value & 128 else value


def pack_gradient(horizontal, vertical):
    return (horizontal & 255) | ((vertical & 255) << 8)


def point_geometry(points, truth, tolerance):
    distances = [min(math.hypot(x - tx, y - ty) for tx, ty in truth) for x, y in points]
    missing = sum(
        not any(math.hypot(x - tx, y - ty) <= tolerance for x, y in points)
        for tx, ty in truth
    )
    return dict(
        missing=missing,
        unexpected=sum(value > tolerance for value in distances),
        maximum_distance=max(distances, default=None),
    )


class Report:
    def __init__(self, path, loops):
        self.path = path
        self.loops = loops
        self.rows = []
        self.processor = vg.IVE()

    def emit(self, row):
        self.rows.append(row)
        self.path.write_text(
            json.dumps(
                dict(
                    schema=2,
                    counts=dict(Counter(item["status"] for item in self.rows)),
                    results=self.rows,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

    def missing(self, name, reason):
        self.emit(dict(name=name, status="unverified", reason=reason))

    def require(self, *names):
        missing = [name for name in names if not hasattr(self.processor, name)]
        if missing:
            self.missing(
                "missing_" + "_".join(missing), "Required interface unavailable"
            )
        return not missing

    def run(self, name, function, hardware=True):
        before = interrupt_count()
        start = time.perf_counter()
        try:
            row = function()
            if row.get("status") not in ("passed", "failed", "unverified"):
                raise ValueError("Check did not return an explicit acceptance status")
        except Exception as error:
            row = dict(status="failed", error=str(error))
        after = interrupt_count()
        delta = after - before if before is not None and after is not None else None
        row.update(
            name=name,
            interrupts=delta,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        if hardware and row["status"] == "passed":
            if delta is None:
                row.update(status="unverified", reason="Hardware counter unavailable")
            elif delta <= 0:
                row.update(status="failed", reason="No hardware interrupt observed")
        self.emit(row)

    def rejected(self, name, call):
        def check():
            before = interrupt_count()
            try:
                call()
            except (ValueError, TypeError):
                after = interrupt_count()
                if before is None or after is None:
                    return dict(
                        status="unverified",
                        reason="Rejection observed, but the hardware counter is unavailable",
                    )
                return dict(
                    status="passed" if before == after else "failed",
                    reason="Invalid controls must be rejected before hardware submission",
                )
            return dict(status="failed", reason="Invalid controls were accepted")

        self.run(name, check, hardware=False)


def check_norm_grad(report, width, height):
    if not report.require("norm_grad_config"):
        return
    kernels = [
        (
            "default",
            [],
            [
                [-1, -2, 0, 2, 1],
                [-4, -8, 0, 8, 4],
                [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4],
                [-1, -2, 0, 2, 1],
            ],
            (0, 6, 7, 8),
        ),
        (
            "sobel3",
            [0] * 6 + [-1, 0, 1, 0, 0, -2, 0, 2, 0, 0, -1, 0, 1] + [0] * 6,
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0],
                [0, -2, 0, 2, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            (0, 3, 4),
        ),
    ]
    roi = (
        slice(height // 2 - 8, height // 2 + 8),
        slice(width // 2 - 8, width // 2 + 8),
    )
    for dx, dy in [(2, 0), (0, -2), (2, -2), (-2, 2)]:
        source = np.asarray(
            [
                [
                    max(
                        0,
                        min(255, 128 + dx * (x - width // 2) + dy * (y - height // 2)),
                    )
                    for x in range(width)
                ]
                for y in range(height)
            ],
            dtype=np.uint8,
        )
        buffer = image(source)
        for kernel_name, mask, matrix, norms in kernels:
            if mask:
                mask = [coefficient for row in matrix for coefficient in row]
            kernel = np.asarray(matrix, dtype=np.float32)
            raw_h = cv2.filter2D(source, cv2.CV_32F, kernel)[roi]
            raw_v = cv2.filter2D(source, cv2.CV_32F, kernel.T)[roi]
            for norm in norms:
                divisor = 1 << norm
                references = []
                for raw in (raw_h, raw_v):
                    rows = raw.tolist()
                    if any(int(value) % divisor for row in rows for value in row):
                        raise ValueError(
                            "Gradient fixture requires exact integral scaling"
                        )
                    references.append(
                        np.asarray(
                            [
                                [
                                    max(-128, min(127, int(value) // divisor))
                                    for value in row
                                ]
                                for row in rows
                            ],
                            dtype=np.int8,
                        )
                    )
                for mode_name in ("HOR_AND_VER", "HOR", "VER", "COMBINE"):
                    mode = getattr(vg.NormGradOutCtrl, mode_name)

                    def check(
                        mode=mode,
                        mode_name=mode_name,
                        norm=norm,
                        mask=mask,
                        references=references,
                    ):
                        signatures, observations = [], []
                        for _ in range(report.loops):
                            outputs = report.processor.norm_grad_config(
                                buffer, mode, norm, mask
                            )
                            required = [
                                mode_name in ("HOR_AND_VER", "HOR"),
                                mode_name in ("HOR_AND_VER", "VER"),
                                mode_name == "COMBINE",
                            ]
                            validity = [value.is_valid() for value in outputs]
                            if validity != required:
                                return dict(
                                    status="failed",
                                    valid_outputs=validity,
                                    expected=required,
                                )
                            arrays = [
                                array(value) if valid else None
                                for value, valid in zip(outputs, validity)
                            ]
                            signatures.append(
                                tuple(
                                    checksum(value) if value is not None else None
                                    for value in arrays
                                )
                            )
                            comparisons = []
                            for index in (0, 1):
                                if required[index]:
                                    comparisons.append(
                                        compare(arrays[index][roi], references[index])
                                    )
                                    if arrays[index].dtype.str != "|i1":
                                        return dict(
                                            status="failed",
                                            reason="Expected signed 8-bit gradient",
                                        )
                            if required[2]:
                                expected = np.asarray(
                                    [
                                        [
                                            pack_gradient(h, v)
                                            for h, v in zip(row_h, row_v)
                                        ]
                                        for row_h, row_v in zip(
                                            references[0].tolist(),
                                            references[1].tolist(),
                                        )
                                    ],
                                    dtype=np.uint16,
                                )
                                comparisons.append(compare(arrays[2][roi], expected))
                                if arrays[2].dtype.str != "<u2":
                                    return dict(
                                        status="failed",
                                        reason="Expected packed U16 gradient",
                                    )
                            observations.extend(comparisons)
                        passed = len(set(signatures)) == 1 and all(
                            row["different"] == 0 for row in observations
                        )
                        return dict(
                            status="passed" if passed else "failed",
                            comparisons=observations,
                            unique_outputs=len(set(signatures)),
                            reference="OpenCV filter2D with exact integer scaling",
                            compared_region="central 16x16",
                        )

                    report.run(
                        "norm_%s_%s_n%d_d%d_%d_%dx%d"
                        % (kernel_name, mode_name, norm, dx, dy, width, height),
                        check,
                    )
    report.rejected(
        "norm_invalid_shift_%dx%d" % (width, height),
        lambda: report.processor.norm_grad_config(buffer, vg.NormGradOutCtrl.HOR, 16),
    )
    report.rejected(
        "norm_invalid_mask_%dx%d" % (width, height),
        lambda: report.processor.norm_grad_config(
            buffer, vg.NormGradOutCtrl.HOR, 7, [1, 2]
        ),
    )


def check_sad(report, width, height):
    if not report.require("sad_config"):
        return
    first_rows = [
        [(7 * x + 13 * y + (x * y) % 71) % 256 for x in range(width)]
        for y in range(height)
    ]
    second_rows = [
        [(19 * x + 3 * y + 17) % 256 for x in range(width)] for y in range(height)
    ]
    first, second = image(np.asarray(first_rows, dtype=np.uint8)), image(
        np.asarray(second_rows, dtype=np.uint8)
    )
    input_hashes = [checksum(array(first)), checksum(array(second))]
    for block, name in [(4, "MB_4X4"), (8, "MB_8X8"), (16, "MB_16X16")]:
        mode = getattr(vg.SadMode, name)
        if width % block or height % block:
            report.rejected(
                "sad_partial_%s_%dx%d" % (name, width, height),
                lambda mode=mode: report.processor.sad_config(
                    first, second, mode, 1024
                ),
            )
            continue
        values = sad_reference(first_rows, second_rows, block)
        for mode_name in ("SAD", "BOTH", "THRESH"):
            output_mode = getattr(vg.SadOutMode, mode_name)
            for bits in (8, 16):
                output_bits = getattr(vg.SadOutBits, "BITS" + str(bits))
                for threshold in (0, 1024, block * block * 255):

                    def check(
                        output_mode=output_mode,
                        output_bits=output_bits,
                        mode_name=mode_name,
                        bits=bits,
                        threshold=threshold,
                    ):
                        signatures, comparisons = [], []
                        for _ in range(report.loops):
                            outputs = report.processor.sad_config(
                                first,
                                second,
                                mode,
                                threshold,
                                17,
                                231,
                                output_mode,
                                output_bits,
                            )
                            validity = [value.is_valid() for value in outputs]
                            required = [mode_name != "THRESH", mode_name != "SAD"]
                            if validity != required:
                                return dict(
                                    status="failed",
                                    valid_outputs=validity,
                                    expected=required,
                                )
                            actual = [
                                array(value) if valid else None
                                for value, valid in zip(outputs, validity)
                            ]
                            signatures.append(
                                tuple(
                                    checksum(value) if value is not None else None
                                    for value in actual
                                )
                            )
                            if required[0]:
                                expected = np.asarray(
                                    [
                                        [
                                            value >> 8 if bits == 8 else value
                                            for value in row
                                        ]
                                        for row in values
                                    ],
                                    dtype=np.uint8 if bits == 8 else np.uint16,
                                )
                                comparisons.append(compare(actual[0], expected))
                                if actual[0].dtype.str != (
                                    "|u1" if bits == 8 else "<u2"
                                ):
                                    return dict(
                                        status="failed",
                                        reason="SAD output dtype mismatch",
                                    )
                            if required[1]:
                                expected = np.asarray(
                                    [
                                        [
                                            231 if value > threshold else 17
                                            for value in row
                                        ]
                                        for row in values
                                    ],
                                    dtype=np.uint8,
                                )
                                comparisons.append(compare(actual[1], expected))
                        passed = len(set(signatures)) == 1 and all(
                            row["different"] == 0 for row in comparisons
                        )
                        return dict(
                            status="passed" if passed else "failed",
                            comparisons=comparisons,
                            unique_outputs=len(set(signatures)),
                            reference="Full integer block sums; U8=sum>>8; threshold uses full sum",
                        )

                    report.run(
                        "sad_%s_%s_b%d_t%d_%dx%d"
                        % (name, mode_name, bits, threshold, width, height),
                        check,
                    )
    report.run(
        "sad_inputs_unchanged_%dx%d" % (width, height),
        lambda: dict(
            status=(
                "passed"
                if [checksum(array(first)), checksum(array(second))] == input_hashes
                else "failed"
            )
        ),
        hardware=False,
    )


def check_lbp(report, width, height):
    source = np.full((height, width), 127, dtype=np.uint8)
    columns = (width - 8) // 4
    samples = []
    for index, delta in enumerate(range(-255, 256)):
        x, y = 4 + (index % columns) * 4, 4 + (index // columns) * 4
        if y + 1 >= height:
            raise ValueError("LBP fixture is too small")
        center = -delta if delta < 0 else 0
        source[y - 1 : y + 2, x - 1 : x + 2] = center + delta
        source[y, x] = center
        samples.append((x, y, delta))
    buffer = image(source)
    for absolute in (False, True):
        thresholds = range(256) if absolute else range(-128, 128)
        for threshold in thresholds:

            def check(absolute=absolute, threshold=threshold):
                signatures, mismatch = [], 0
                for _ in range(report.loops):
                    output = array(report.processor.lbp(buffer, absolute, threshold))
                    signatures.append(checksum(output))
                    mismatch += sum(
                        int(output[y, x])
                        != (
                            255
                            if (abs(delta) if absolute else delta) >= threshold
                            else 0
                        )
                        for x, y, delta in samples
                    )
                return dict(
                    status=(
                        "passed"
                        if mismatch == 0 and len(set(signatures)) == 1
                        else "failed"
                    ),
                    samples=len(samples),
                    different=mismatch,
                    unique_outputs=len(set(signatures)),
                    reference="All eight neighbors share one independently chosen signed difference",
                )

            report.run(
                "lbp_%s_t%d_%dx%d"
                % ("abs" if absolute else "normal", threshold, width, height),
                check,
            )
    for absolute, threshold in [(False, -129), (False, 128), (True, -1), (True, 256)]:
        report.rejected(
            "lbp_reject_%s_%d_%dx%d" % (absolute, threshold, width, height),
            lambda absolute=absolute, threshold=threshold: report.processor.lbp(
                buffer, absolute, threshold
            ),
        )


def flow_fixture(width, height):
    generator = random.Random(281106)
    source = np.frombuffer(
        bytes(generator.randrange(256) for _ in range(width * height)), dtype=np.uint8
    ).reshape(height, width)
    source = cv2.GaussianBlur(source, (5, 5), 0.8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[48:-48, 48:-48] = 255
    features = cv2.goodFeaturesToTrack(source, 40, 0.02, 12, mask=mask)
    if features is None:
        raise ValueError("The optical flow fixture has no software reference corners")
    points = [
        (int(round(float(x))), int(round(float(y))))
        for x, y in features.reshape(-1, 2).tolist()
    ]
    return source, points


def check_flow(report, width, height):
    if not report.require("lk_optical_flow_config", "lk_optical_flow_single"):
        return
    source, points = flow_fixture(width, height)
    previous = image(source)
    cases = [("single", 0, 0, 0), ("single", 0, 1, -1)]
    cases.extend(
        ("pyramid", level, dx, dy) for level in range(4) for dx, dy in [(0, 0), (1, -1)]
    )
    cases.extend(("pyramid", level, 4, -3) for level in (1, 2, 3))
    cases.append(("initial", 3, 4, -3))
    for kind, level, dx, dy in cases:
        shifted = cv2.warpAffine(
            source,
            np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        current = image(shifted)
        coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        tracked, software_status, _ = cv2.calcOpticalFlowPyrLK(
            source,
            shifted,
            coordinates,
            None,
            winSize=(21, 21),
            maxLevel=level,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
            minEigThreshold=0.001,
        )
        software_errors = [
            math.hypot(float(x) - px - dx, float(y) - py - dy)
            for (x, y), (px, py) in zip(tracked.reshape(-1, 2).tolist(), points)
        ]

        def check(
            kind=kind,
            level=level,
            dx=dx,
            dy=dy,
            current=current,
            software_errors=software_errors,
            software_status=software_status,
        ):
            if not all(software_status.ravel().tolist()) or max(software_errors) > 0.2:
                return dict(
                    status="unverified",
                    reason="Software fixture failed the known translation check",
                    software_maximum_error=max(software_errors),
                )
            signatures, errors, raw_errors, valid_counts = [], [], [], []
            for _ in range(report.loops):
                if kind == "single":
                    vectors = report.processor.lk_optical_flow_single(
                        previous, current, points, 20, 100, 32
                    )
                    residual = []
                else:
                    initial = (
                        [(x + dx, y + dy) for x, y in points]
                        if kind == "initial"
                        else []
                    )
                    vectors, residual = report.processor.lk_optical_flow_config(
                        previous, current, points, level, 20, 100, 32, initial
                    )
                signature = [
                    (value.status, value.mv_x, value.mv_y) for value in vectors
                ]
                signatures.append((tuple(signature), tuple(residual)))
                valid_counts.append(sum(value.status == 0 for value in vectors))
                if len(vectors) != len(points) or (
                    kind != "single" and len(residual) != len(points)
                ):
                    return dict(
                        status="failed", reason="Optical flow result length mismatch"
                    )
                errors.extend(
                    math.hypot(value.mv_x / 128 - dx, value.mv_y / 128 - dy)
                    for value in vectors
                    if value.status == 0
                )
                raw_errors.extend(residual)
            passed = (
                min(valid_counts) == len(points)
                and len(set(signatures)) == 1
                and len(errors) > 0
                and max(errors) <= 0.2
            )
            if raw_errors:
                passed = passed and all(0 <= value <= 65535 for value in raw_errors)
                if dx == 0 and dy == 0:
                    passed = passed and all(value == 0 for value in raw_errors)
            return dict(
                status="passed" if passed else "failed",
                points=len(points),
                valid_counts=valid_counts,
                maximum_error_px=max(errors, default=None),
                software_maximum_error_px=max(software_errors),
                unique_outputs=len(set(signatures)),
                raw_error_min=min(raw_errors, default=None),
                raw_error_max=max(raw_errors, default=None),
                error_reference="Identical frames require zero residual; translated residual numeric convention remains separate",
            )

        report.run(
            "flow_%s_l%d_d%d_%d_%dx%d" % (kind, level, dx, dy, width, height), check
        )
    report.missing(
        "flow_translated_error_scale_%dx%d" % (width, height),
        "The nonzero U9Q7 error aggregation and interpolation convention needs an independent hardware algorithm specification; flow truth and zero residual are checked separately",
    )
    flat = image(np.full((height, width), 127, dtype=np.uint8))

    def flat_check():
        vectors, errors = report.processor.lk_optical_flow_config(flat, flat, points)
        return dict(
            status=(
                "passed"
                if len(vectors) == len(points)
                and all(value.status == -1 for value in vectors)
                else "failed"
            ),
            valid=sum(value.status == 0 for value in vectors),
            error_count=len(errors),
        )

    report.run("flow_flat_%dx%d" % (width, height), flat_check)
    for name, values in [
        ("level", dict(max_level=4)),
        ("iterations", dict(iterations=0)),
        ("eigen", dict(min_eigenvalue=256)),
        ("epsilon", dict(epsilon=65536)),
        ("initial_length", dict(initial_points=[(1, 1)])),
    ]:
        report.rejected(
            "flow_reject_%s_%dx%d" % (name, width, height),
            lambda values=values: report.processor.lk_optical_flow_config(
                previous, previous, points, **values
            ),
        )


def corner_fixture(width, height):
    source = np.zeros((height, width), dtype=np.uint8)
    truth = []
    for cy in (height // 3, height * 2 // 3):
        for cx in (width // 4, width // 2, width * 3 // 4):
            source[cy - 8 : cy + 8, cx - 8 : cx + 8] = 220
            truth.extend((x, y) for x in (cx - 8, cx + 7) for y in (cy - 8, cy + 7))
    return source, truth


def check_corners(report, width, height):
    if not report.require("st_corner_candidates", "st_corner_sorted"):
        return
    source, truth = corner_fixture(width, height)
    software = cv2.goodFeaturesToTrack(source, 100, 0.01, 8)
    software_points = [] if software is None else software.reshape(-1, 2).tolist()
    if (
        len(software_points) != len(truth)
        or point_geometry(software_points, truth, 2)["missing"]
    ):
        report.missing(
            "corner_fixture_%dx%d" % (width, height),
            "Software corner fixture did not produce the controlled rectangle corners",
        )
        return
    buffer = image(source)
    for kind in ("candidates", "sorted"):

        def check(kind=kind):
            signatures, observations = [], []
            for _ in range(report.loops):
                values = (
                    report.processor.st_corner_candidates(buffer, 25)
                    if kind == "candidates"
                    else report.processor.st_corner_sorted(buffer, 200, 12, 25)
                )
                signature = tuple(tuple(map(int, value)) for value in values)
                signatures.append(signature)
                points = [(x, y) for x, y, _ in signature]
                geometry = point_geometry(points, truth, 3)
                spacing = min(
                    (
                        math.hypot(x - px, y - py)
                        for i, (x, y) in enumerate(points)
                        for px, py in points[i + 1 :]
                    ),
                    default=None,
                )
                observations.append(
                    dict(
                        count=len(points),
                        geometry=geometry,
                        minimum_spacing=spacing,
                        lambda_range=[
                            min((v for _, _, v in signature), default=None),
                            max((v for _, _, v in signature), default=None),
                        ],
                    )
                )
                if (
                    geometry["missing"]
                    or geometry["unexpected"]
                    or any(not (0 <= value <= 255) for _, _, value in signature)
                ):
                    return dict(status="failed", observations=observations)
                if kind == "sorted" and (
                    len(points) != len(truth) or (spacing is not None and spacing < 12)
                ):
                    return dict(status="failed", observations=observations)
            return dict(
                status="passed" if len(set(signatures)) == 1 else "failed",
                unique_outputs=len(set(signatures)),
                observations=observations,
                reference="Known rectangle vertices confirmed by OpenCV goodFeaturesToTrack; 3-pixel localization tolerance",
            )

        report.run("corner_%s_geometry_%dx%d" % (kind, width, height), check)
    for maximum in (1, 8, 16):

        def limited(maximum=maximum):
            result = report.processor.st_corner_sorted(buffer, maximum, 12, 25)
            geometry = point_geometry([(x, y) for x, y, _ in result], truth, 3)
            return dict(
                status=(
                    "passed"
                    if len(result) == maximum and geometry["unexpected"] == 0
                    else "failed"
                ),
                count=len(result),
                geometry=geometry,
            )

        report.run("corner_limit_%d_%dx%d" % (maximum, width, height), limited)
    flat = image(np.full((height, width), 80, dtype=np.uint8))

    def empty():
        a = report.processor.st_corner_candidates(flat)
        b = report.processor.st_corner_sorted(flat)
        return dict(
            status="passed" if not a and not b else "failed",
            candidates=len(a),
            selected=len(b),
        )

    report.run("corner_flat_%dx%d" % (width, height), empty)
    report.missing(
        "corner_lambda_numeric_scale_%dx%d" % (width, height),
        "Candidate and selected score range/geometry are checked; the 8-bit lambda normalization has no independently verified numeric model",
    )


def check_model_lifecycle(report, width, height):
    if not hasattr(vg.IVEModel, "reset"):
        report.missing(
            "model_lifecycle",
            "Models must preserve dimensions and explicitly initialize/reset algorithm state",
        )
        return
    source = image(np.full((height, width), 80, dtype=np.uint8))
    factor = image(np.full((height, width), 0x0808, dtype=np.uint16), "u16c1")
    model = vg.IVEModel(width, height)
    report.rejected(
        "model_gmm_before_initialization",
        lambda: report.processor.gmm(source, model, False),
    )
    report.rejected(
        "model_gmm2_before_initialization",
        lambda: report.processor.gmm2(source, factor, model, False),
    )
    report.rejected(
        "model_codebook_before_training",
        lambda: report.processor.update_bg_model(source, model, 0),
    )
    for label, call in (
        ("gmm", lambda other: report.processor.gmm(source, other, True)),
        ("gmm2", lambda other: report.processor.gmm2(source, factor, other, True)),
        (
            "codebook_train",
            lambda other: report.processor.match_bg_model(source, other, 0),
        ),
        (
            "codebook_update",
            lambda other: report.processor.update_bg_model(source, other, 0),
        ),
    ):
        other = vg.IVEModel(width + 16 if width <= 4080 else width - 16, height)
        report.rejected(
            "model_dimensions_" + label, lambda other=other, call=call: call(other)
        )
    tiny = vg.IVEModel(width, height, 1)
    report.rejected(
        "model_capacity_gmm", lambda: report.processor.gmm(source, tiny, True)
    )
    report.rejected(
        "model_capacity_codebook",
        lambda: report.processor.match_bg_model(source, tiny, 0),
    )
    report.rejected("model_negative_capacity", lambda: vg.IVEModel(width, height, -1))

    def quiet_result(outputs):
        foreground, background = map(array, outputs)
        comparisons = dict(
            foreground=compare(foreground, np.zeros((height, width), dtype=np.uint8)),
            background=compare(
                background, np.full((height, width), 80, dtype=np.uint8)
            ),
        )
        return dict(
            status=(
                "passed"
                if all(row["different"] == 0 for row in comparisons.values())
                else "failed"
            ),
            comparisons=comparisons,
            reference="A constant image after initialization has no motion and preserves its background",
        )

    def initialize_gmm():
        report.processor.gmm(source, model, True)
        return quiet_result(report.processor.gmm(source, model, False))

    report.run("model_initialize_gmm", initialize_gmm)
    report.rejected(
        "model_gmm_to_gmm2_without_initialization",
        lambda: report.processor.gmm2(source, factor, model, False),
    )
    report.rejected(
        "model_gmm_to_codebook_without_reset",
        lambda: report.processor.match_bg_model(source, model, 0),
    )

    def initialize_gmm2():
        report.processor.gmm2(source, factor, model, True)
        return quiet_result(report.processor.gmm2(source, factor, model, False))

    report.run("model_reinitialize_as_gmm2", initialize_gmm2)
    report.rejected(
        "model_gmm2_to_gmm_without_initialization",
        lambda: report.processor.gmm(source, model, False),
    )

    def reset():
        before = interrupt_count()
        model.reset()
        after = interrupt_count()
        if before is None or after is None:
            return dict(
                status="unverified",
                reason="Reset performed, but the hardware counter is unavailable",
            )
        return dict(
            status=(
                "passed"
                if before == after and (model.width, model.height) == (width, height)
                else "failed"
            ),
            dimensions=[model.width, model.height],
        )

    report.run("model_reset_preserves_dimensions", reset, hardware=False)
    report.rejected(
        "model_after_reset_requires_initialization",
        lambda: report.processor.gmm2(source, factor, model, False),
    )
    report.rejected(
        "model_after_reset_requires_training",
        lambda: report.processor.update_bg_model(source, model, 0),
    )

    def retrain_codebook():
        for frame in range(60):
            report.processor.match_bg_model(source, model, frame)
        return quiet_result(report.processor.update_bg_model(source, model, 60))

    report.run("model_reset_then_codebook_training", retrain_codebook)
    report.rejected(
        "model_codebook_to_gmm_without_initialization",
        lambda: report.processor.gmm(source, model, False),
    )
    report.run("model_codebook_then_explicit_gmm_initialization", initialize_gmm)


def check_stateful(report, width, height):
    base = np.full((height, width), 80, dtype=np.uint8)
    baseline = image(base)
    zero = np.zeros((height, width), dtype=np.uint8)
    initialized_foreground = np.full((height, width), 255, dtype=np.uint8)
    factor_reference = np.full((height, width), 0x0808, dtype=np.uint16)
    factor = image(factor_reference, "u16c1")
    rectangles = [(3, 4, 41, 37), (width - 45, height - 51, width - 2, height - 3)]
    for kind in ("gmm", "gmm2", "codebook"):
        for value in (10, 220):
            for index, (left, top, right, bottom) in enumerate(rectangles):
                changed = base.copy()
                changed[top:bottom, left:right] = value
                current = image(changed)
                reference = cv2.threshold(
                    cv2.absdiff(base, changed), 32, 255, cv2.THRESH_BINARY
                )[1]

                def check(
                    kind=kind,
                    current=current,
                    changed=changed,
                    reference=reference,
                ):
                    signatures, observations = [], []
                    reference_description = (
                        "Codebook: 60 baseline-80 training frames; baseline at frame60 is quiet, "
                        "a 10/220 rectangle at frame63 is nonzero foreground while background remains80. "
                        "The rectangle mask comes from OpenCV absdiff + threshold; full frames are compared."
                        if kind == "codebook"
                        else "Measured single-model RV1106 contract with default GMM/GMM2 controls "
                        "and GMM2 factor0x0808: initialization80 gives FG255/BG80; ten baseline "
                        "frames give FG0/BG80; first 10/220 rectangle gives FG255 only in its mask "
                        "and same-call BG=current; identical repeat gives FG0/BG=current; returning80 "
                        "gives the rectangle FG255 once with BG80; repeating80 gives FG0/BG80. "
                        "Foreground is compared as exact 0/255 bytes and every background pixel is compared."
                    )
                    for _ in range(report.loops):
                        model = vg.IVEModel(width, height)
                        steps, held = [], []

                        def inspect_step(
                            label,
                            outputs,
                            expected_foreground,
                            expected_background,
                            normalize_foreground=False,
                        ):
                            foreground, background = outputs
                            actual_foreground, actual_background = array(
                                foreground
                            ), array(background)
                            signature = (
                                checksum(actual_foreground),
                                checksum(actual_background),
                            )
                            dtypes = [
                                actual_foreground.dtype.str,
                                actual_background.dtype.str,
                            ]
                            held.append((foreground, background, signature))
                            if normalize_foreground:
                                actual_foreground = (actual_foreground != 0).astype(
                                    np.uint8
                                )
                                expected_foreground = (expected_foreground != 0).astype(
                                    np.uint8
                                )
                            steps.append(
                                dict(
                                    frame=label,
                                    foreground=compare(
                                        actual_foreground, expected_foreground
                                    ),
                                    background=compare(
                                        actual_background, expected_background
                                    ),
                                    dtypes=dtypes,
                                    sha256=list(signature),
                                )
                            )

                        if kind == "codebook":
                            for frame in range(60):
                                if (
                                    report.processor.match_bg_model(
                                        baseline, model, frame
                                    )
                                    is not None
                                ):
                                    return dict(
                                        status="failed",
                                        reason="Training must return None",
                                    )
                            inspect_step(
                                "trained_baseline",
                                report.processor.update_bg_model(baseline, model, 60),
                                zero,
                                base,
                                normalize_foreground=True,
                            )
                            inspect_step(
                                "rectangle_once",
                                report.processor.update_bg_model(current, model, 63),
                                reference,
                                base,
                                normalize_foreground=True,
                            )
                        else:

                            def call(frame, first):
                                return (
                                    report.processor.gmm(frame, model, first)
                                    if kind == "gmm"
                                    else report.processor.gmm2(
                                        frame, factor, model, first
                                    )
                                )

                            # Controlled single-model sequences establish same-call
                            # background replacement and a quiet identical next frame.
                            # 受控单模型序列确认：本次调用替换背景，下一相同帧静止。
                            inspect_step(
                                "initialization",
                                call(baseline, True),
                                initialized_foreground,
                                base,
                            )
                            for frame in range(1, 11):
                                inspect_step(
                                    "train_%02d" % frame,
                                    call(baseline, False),
                                    zero,
                                    base,
                                )
                            for label, source, wanted_foreground, wanted_background in (
                                ("rectangle_once", current, reference, changed),
                                ("rectangle_repeat", current, zero, changed),
                                ("return_baseline_once", baseline, reference, base),
                                ("return_baseline_repeat", baseline, zero, base),
                            ):
                                inspect_step(
                                    label,
                                    call(source, False),
                                    wanted_foreground,
                                    wanted_background,
                                )
                        retained_outputs_stable = all(
                            (checksum(array(foreground)), checksum(array(background)))
                            == signature
                            for foreground, background, signature in held
                        )
                        inputs_preserved = (
                            compare(array(baseline), base)["different"] == 0
                            and compare(array(current), changed)["different"] == 0
                            and compare(array(factor), factor_reference)["different"]
                            == 0
                        )
                        signatures.append(tuple(tuple(row["sha256"]) for row in steps))
                        observations.append(
                            dict(
                                steps=steps,
                                retained_outputs_stable=retained_outputs_stable,
                                inputs_preserved=inputs_preserved,
                            )
                        )
                        if (
                            not retained_outputs_stable
                            or not inputs_preserved
                            or any(
                                row["foreground"]["different"] != 0
                                or row["background"]["different"] != 0
                                or row["dtypes"] != ["|u1", "|u1"]
                                for row in steps
                            )
                        ):
                            return dict(
                                status="failed",
                                observations=observations,
                                reference=reference_description,
                            )
                    return dict(
                        status="passed" if len(set(signatures)) == 1 else "failed",
                        unique_outputs=len(set(signatures)),
                        observations=observations,
                        reference=reference_description,
                    )

                report.run(
                    "state_%s_v%d_r%d_%dx%d" % (kind, value, index, width, height),
                    check,
                )


def pyramid_reference(source, levels, border):
    """Apply an exact integer Gaussian sum before floor-sized decimation.
    精确计算整数高斯卷积，再按向下取整的尺寸隔点采样。
    """
    weights = (1, 4, 6, 4, 1)
    kernel = np.asarray(
        [[vertical * horizontal for horizontal in weights] for vertical in weights],
        dtype=np.float32,
    )
    result = [source.copy()]
    for _ in range(1, levels):
        previous = result[-1]
        height, width = previous.shape[0] // 2, previous.shape[1] // 2
        sums = cv2.filter2D(previous, cv2.CV_32F, kernel, borderType=border)
        shifted = (sums.astype(np.int32) >> 8).astype(np.uint8)
        result.append(shifted[: height * 2 : 2, : width * 2 : 2].copy())
    return result


def check_pyramid(report, width, height):
    if not report.require("create_pyramid"):
        return
    generator = random.Random(4917)
    sources = {
        "constant": np.full((height, width), 173, dtype=np.uint8),
        "rows_columns": np.asarray(
            [
                [(17 * x + 31 * y + x * y % 29) % 256 for x in range(width)]
                for y in range(height)
            ],
            dtype=np.uint8,
        ),
        "checkerboard": np.asarray(
            [[255 * ((x + y) & 1) for x in range(width)] for y in range(height)],
            dtype=np.uint8,
        ),
        "noise": np.frombuffer(
            bytes(generator.randrange(256) for _ in range(width * height)),
            dtype=np.uint8,
        ).reshape(height, width),
    }
    for pattern, source in sources.items():
        buffer = vg.ImageBuffer.from_bytes(source.tobytes(), width, height, "gray8")
        for levels in (2, 3, 4):
            references = {
                name: pyramid_reference(source, levels, border)
                for name, border in (
                    ("replicate", cv2.BORDER_REPLICATE),
                    ("constant_zero", cv2.BORDER_CONSTANT),
                    ("reflect101", cv2.BORDER_REFLECT_101),
                )
            }

            def check(levels=levels, references=references):
                signatures, observations, valid = [], [], True
                retained_outputs_stable, inputs_preserved = True, True
                for _ in range(report.loops):
                    outputs = report.processor.create_pyramid(buffer, levels)
                    if len(outputs) != levels:
                        return dict(
                            status="failed",
                            returned_levels=len(outputs),
                            expected_levels=levels,
                        )
                    actual = [array(output) for output in outputs]
                    signatures.append(tuple(checksum(value) for value in actual))
                    comparisons = []
                    for level, value in enumerate(actual):
                        expected = references["replicate"][level]
                        margin = 0
                        interior = compare(
                            value[margin:-margin, margin:-margin] if margin else value,
                            (
                                expected[margin:-margin, margin:-margin]
                                if margin
                                else expected
                            ),
                        )
                        valid = (
                            valid
                            and interior["different"] == 0
                            and value.dtype.str == "|u1"
                        )
                        comparisons.append(
                            dict(
                                level=level,
                                shape=list(value.shape),
                                dtype=value.dtype.str,
                                interior=interior,
                                excluded_border=margin,
                                full={
                                    name: compare(value, reference[level])
                                    for name, reference in references.items()
                                },
                            )
                        )
                    observations.append(comparisons)
                    inputs_preserved = (
                        inputs_preserved
                        and compare(array(buffer), source)["different"] == 0
                    )
                    valid = valid and inputs_preserved
                    held_signatures = tuple(
                        checksum(array(output)) for output in outputs
                    )
                    alternate = image(
                        np.full((height + 2, width + 2), 91, dtype=np.uint8)
                    )
                    report.processor.create_pyramid(alternate, 2)
                    retained_outputs_stable = (
                        retained_outputs_stable
                        and held_signatures
                        == tuple(checksum(array(output)) for output in outputs)
                    )
                    valid = valid and retained_outputs_stable
                return dict(
                    status=(
                        "passed" if valid and len(set(signatures)) == 1 else "failed"
                    ),
                    observations=observations,
                    unique_outputs=len(set(signatures)),
                    reference="Full-frame integer Gaussian kernel sums with replicated edges, shifted right by eight, followed by even-coordinate decimation and floor dimensions",
                    retained_outputs_stable=retained_outputs_stable,
                    inputs_preserved=inputs_preserved,
                    border_contract="replicate_full_frame",
                )

            report.run(
                "pyramid_%s_levels%d_%dx%d" % (pattern, levels, width, height), check
            )
    report.emit(
        dict(
            name="pyramid_border_convention_%dx%d" % (width, height),
            status="unverified",
            reason="Alternative border-model observations remain exploratory; every numeric case separately verifies the full-frame replicated-edge contract",
        )
    )
    buffer = image(sources["constant"])
    for levels in (0, 1, 5):
        report.rejected(
            "pyramid_invalid_levels%d_%dx%d" % (levels, width, height),
            lambda levels=levels: report.processor.create_pyramid(buffer, levels),
        )
    too_small = image(np.zeros((62, 64), dtype=np.uint8))
    report.rejected(
        "pyramid_small_last_level",
        lambda: report.processor.create_pyramid(too_small, 4),
    )
    signed = image(sources["constant"].astype(np.int16), "s16c1")
    report.rejected(
        "pyramid_wrong_format", lambda: report.processor.create_pyramid(signed, 2)
    )


def self_test():
    assert sad_reference([[0, 2], [4, 6]], [[1, 2], [3, 8]], 2) == [[4]]
    assert sad_reference([[255] * 4] * 4, [[0] * 4] * 4, 4) == [[4080]]
    assert signed_byte(0) == 0 and signed_byte(127) == 127
    assert signed_byte(128) == -128 and signed_byte(255) == -1
    for horizontal in (-128, -1, 0, 1, 127):
        for vertical in (-128, -1, 0, 1, 127):
            value = pack_gradient(horizontal, vertical)
            assert signed_byte(value & 255) == horizontal
            assert signed_byte(value >> 8) == vertical
    assert point_geometry([(1, 1), (9, 9)], [(0, 0), (10, 10)], 2)["missing"] == 0
    assert point_geometry([], [(0, 0)], 2)["missing"] == 1
    assert point_geometry([(10, 10)], [(0, 0)], 2)["unexpected"] == 1
    print(
        json.dumps(
            dict(
                status="passed",
                reference_self_tests=[
                    "sad",
                    "signed_byte",
                    "gradient_pack",
                    "point_geometry",
                ],
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_extended.json"))
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--sizes", default="320x240,322x242,640x480")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=(
            "norm",
            "sad",
            "lbp",
            "flow",
            "corners",
            "pyramid",
            "model",
            "stateful",
        ),
        default=(
            "norm",
            "sad",
            "lbp",
            "flow",
            "corners",
            "pyramid",
            "model",
            "stateful",
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return 0
    if args.loops < 1:
        parser.error("--loops must be positive")
    try:
        sizes = [
            tuple(map(int, item.lower().split("x"))) for item in args.sizes.split(",")
        ]
        if any(len(size) != 2 or size[0] < 128 or size[1] < 128 for size in sizes):
            raise ValueError()
    except ValueError:
        parser.error(
            "--sizes requires comma-separated WIDTHxHEIGHT values, each at least 128"
        )
    global np, cv2, vg
    import cv2
    import numpy as np
    import visiong as vg

    cv2.setNumThreads(1)
    report = Report(args.output, args.loops)
    checks = dict(
        norm=check_norm_grad,
        sad=check_sad,
        lbp=check_lbp,
        flow=check_flow,
        corners=check_corners,
        pyramid=check_pyramid,
        model=check_model_lifecycle,
        stateful=check_stateful,
    )
    for width, height in sizes:
        for group in args.groups:
            checks[group](report, width, height)
    counts = Counter(row["status"] for row in report.rows)
    print(json.dumps(dict(counts=dict(counts))), flush=True)
    if counts["failed"]:
        return 1
    return 2 if args.require_complete and counts["unverified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
