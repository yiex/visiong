# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check IVE logic, subtraction and Canny against explicit software references.
使用明确的软件参考检查 IVE 逻辑、减法与 Canny 算子。

Exploratory SHIFT hypotheses remain unverified; its fixed byte contract is tested
separately. A scoped OpenCV match does not establish every hardware Canny rule.
SHIFT 探索性假设保持未验证，其固定字节契约另行测试；局部吻合 OpenCV 不证明全部硬件规则。
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np


SOBEL3 = [
    0,
    0,
    0,
    0,
    0,
    0,
    -1,
    0,
    1,
    0,
    0,
    -2,
    0,
    2,
    0,
    0,
    -1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]
SOBEL5 = [value * weight for weight in (1, 4, 6, 4, 1) for value in (-1, -2, 0, 2, 1)]
DIFFERENCE = [0] * 25
DIFFERENCE[11], DIFFERENCE[13] = -1, 1
KERNELS = {"sobel3": SOBEL3, "sobel5": SOBEL5, "difference": DIFFERENCE}
BORDERS = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant_zero": cv2.BORDER_CONSTANT,
    "reflect101": cv2.BORDER_REFLECT_101,
}


def interrupts():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def image_array(image):
    return image.to_numpy().copy()


def difference(actual, expected, border=0):
    if list(actual.shape) != list(expected.shape):
        return dict(shape_matches=False, different_pixels=None, maximum_difference=None)
    if border:
        actual, expected = (
            actual[border:-border, border:-border],
            expected[border:-border, border:-border],
        )
    delta = cv2.absdiff(actual, expected)
    return dict(
        shape_matches=True,
        different_pixels=int(cv2.countNonZero(delta)),
        maximum_difference=int(delta.max()),
    )


def boundary_difference(actual, expected, border):
    if list(actual.shape) != list(expected.shape):
        return None
    selected = np.ones(actual.shape, dtype=np.uint8)
    selected[border:-border, border:-border] = 0
    return int(
        cv2.countNonZero(
            cv2.bitwise_and(cv2.compare(actual, expected, cv2.CMP_NE), selected)
        )
    )


def pair_inputs(width, height):
    """The 256-square grid enumerates a=row and b=column exactly once.
    256 方阵以行为 a、列为 b，恰好枚举全部输入对。
    """
    left = bytes(
        ((y * width + x) // 256) % 256 for y in range(height) for x in range(width)
    )
    right = bytes((y * width + x) % 256 for y in range(height) for x in range(width))
    return np.frombuffer(left, dtype=np.uint8).reshape(height, width), np.frombuffer(
        right, dtype=np.uint8
    ).reshape(height, width)


def logic_reference(left, right, operation):
    formulas = dict(
        AND=lambda a, b: a & b,
        OR=lambda a, b: a | b,
        XOR=lambda a, b: a ^ b,
        ABS=lambda a, b: abs(a - b),
    )
    formula = formulas[operation]
    return np.frombuffer(
        bytes(formula(a, b) for a, b in zip(left.tobytes(), right.tobytes())),
        dtype=np.uint8,
    ).reshape(left.shape)


def shift_candidates(left, right):
    """Compare interpretations of the SDK's signed-bit right-shift wording.
    对照 SDK 关于保留符号位并右移的文字所对应的多种解释。
    """
    formulas = {
        "arithmetic_half_twos_complement": lambda d: (d >> 1) & 255,
        "logical_nine_bit_half": lambda d: (d & 511) >> 1,
        "truncate_half_twos_complement": lambda d: math.trunc(d / 2) & 255,
        "biased_255_half_floor": lambda d: (d + 255) // 2,
        "biased_256_half_floor": lambda d: (d + 256) // 2,
        "unsigned_byte_then_half": lambda d: (d & 255) >> 1,
        "signed_magnitude_half": lambda d: (abs(d) >> 1) | (128 if d < 0 else 0),
        "positive_saturated_half": lambda d: max(0, d >> 1),
        "positive_saturated_difference": lambda d: max(0, d),
        "wrapped_difference": lambda d: d & 255,
    }
    return {
        name: np.frombuffer(
            bytes(formula(a - b) for a, b in zip(left.tobytes(), right.tobytes())),
            dtype=np.uint8,
        ).reshape(left.shape)
        for name, formula in formulas.items()
    }


def shift_numeric_reference(left, right):
    """Halve the signed nine-bit difference with floor rounding, then keep its low byte.
    将九位有符号差值除以二并向负无穷取整，最后保留低八位。

    This is an observable byte contract, not a model of the internal circuit.
    这是可观测的字节契约，不代表内部电路实现。
    """
    return np.frombuffer(
        bytes(((a - b) // 2) & 255 for a, b in zip(left.tobytes(), right.tobytes())),
        dtype=np.uint8,
    ).reshape(left.shape)


def gradients(source, mask, border=cv2.BORDER_REPLICATE):
    """Compute signed derivatives using the same 25 coefficients and transpose.
    使用相同二十五个系数及其转置计算有符号导数。
    """
    kernel = np.asarray(mask, dtype=np.float32).reshape(5, 5)
    return cv2.filter2D(source, cv2.CV_16S, kernel, borderType=border), cv2.filter2D(
        source, cv2.CV_16S, kernel.T, borderType=border
    )


def canny_reference(source, mask, low, high, l2=False, border=cv2.BORDER_REPLICATE):
    dx, dy = gradients(source, mask, border)
    return cv2.Canny(dx, dy, float(low), float(high), L2gradient=l2)


def patterns(width, height):
    blank = np.full((height, width), 80, dtype=np.uint8)
    step = np.zeros((height, width), dtype=np.uint8)
    step[:, width // 2 :] = 20
    slanted = np.asarray(
        [
            [
                32 + (160 if 3 * x + 2 * y > 2 * width + height // 3 else 0)
                for x in range(width)
            ]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    cv2.line(
        slanted, (width // 8, height * 4 // 5), (width * 7 // 8, height // 7), 240, 2
    )
    corner = np.full((height, width), 24, dtype=np.uint8)
    corner[height // 5 : height * 3 // 5, width // 5 : width * 3 // 5] = 200
    corner[height * 2 // 5 : height * 4 // 5, width * 2 // 5 : width * 4 // 5] = 90
    weak = np.full((height, width), 32, dtype=np.uint8)
    left, right = width // 4, width * 2 // 5
    weak[height // 6 : height // 3, left:right] = 192
    weak[height // 3 : height * 5 // 6, left:right] = 52
    weak[height // 2 : height * 3 // 4, width * 3 // 5 : width * 4 // 5] = 52
    generator = random.Random(1106)
    noise = np.frombuffer(
        bytes(generator.randrange(256) for _ in range(width * height)), dtype=np.uint8
    ).reshape(height, width)
    return dict(
        blank=blank,
        step20=step,
        slanted=slanted,
        corner=corner,
        connected_weak=weak,
        noise=noise,
    )


class Checks:
    def __init__(self, vg, output, raw_dir, loops, max_edge_artifacts):
        self.vg, self.processor = vg, vg.IVE()
        self.output, self.raw_dir, self.loops = output, raw_dir, loops
        self.max_edge_artifacts, self.edge_artifacts = max_edge_artifacts, 0
        self.rows = []
        raw_dir.mkdir(parents=True, exist_ok=True)

    def image(self, source, pixel_format="gray8"):
        return self.vg.ImageBuffer.from_numpy(source, pixel_format)

    def emit(self, row):
        self.rows.append(row)
        self.output.write_text(
            json.dumps(
                dict(
                    schema=1,
                    opencv_version=cv2.__version__,
                    reference_derivatives="CV_16S filter2D with the supplied mask and its transpose; no preblur",
                    counts=dict(Counter(value["status"] for value in self.rows)),
                    results=self.rows,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

    def raw(self, name, data):
        contents = data.tobytes()
        filename = name + ".u8"
        (self.raw_dir / filename).write_bytes(contents)
        return dict(
            file=filename,
            bytes=len(contents),
            shape=list(data.shape),
            dtype=data.dtype.str,
            sha256=hashlib.sha256(contents).hexdigest(),
        )

    def capture(self, call, inputs):
        signatures, irqs, durations = [], [], []
        snapshots = [image_array(value) for value in inputs]
        preserved = True
        actual = None
        for _ in range(self.loops):
            before = interrupts()
            start = time.perf_counter()
            actual = image_array(call())
            durations.append((time.perf_counter() - start) * 1000)
            after = interrupts()
            irqs.append(
                after - before if after is not None and before is not None else None
            )
            signatures.append(hashlib.sha256(actual.tobytes()).hexdigest())
            preserved = preserved and all(
                difference(image_array(value), snapshot)["different_pixels"] == 0
                for value, snapshot in zip(inputs, snapshots)
            )
        return actual, dict(
            dtype=actual.dtype.str,
            shape=list(actual.shape),
            sha256=signatures[0],
            unique_outputs=len(set(signatures)),
            inputs_preserved=preserved,
            output_shape_matches_input=list(actual.shape) == list(snapshots[0].shape),
            interrupts_per_call=irqs,
            min_ms=min(durations),
        )

    def status(self, metadata, numeric_status):
        if (
            metadata["dtype"] != "|u1"
            or metadata["unique_outputs"] != 1
            or not metadata["inputs_preserved"]
            or not metadata["output_shape_matches_input"]
            or any(
                value is not None and value <= 0
                for value in metadata["interrupts_per_call"]
            )
        ):
            return "failed"
        if numeric_status == "passed" and any(
            value is None for value in metadata["interrupts_per_call"]
        ):
            return "unverified"
        return numeric_status

    def exact(self, name, call, inputs, expected, scope):
        try:
            actual, metadata = self.capture(call, inputs)
            comparison = difference(actual, expected)
            status = self.status(
                metadata, "passed" if comparison["different_pixels"] == 0 else "failed"
            )
            row = dict(
                name=name, status=status, scope=scope, comparison=comparison, **metadata
            )
        except Exception as error:
            row = dict(name=name, status="failed", error=str(error), scope=scope)
        self.emit(row)

    def reject(self, name, call):
        before = interrupts()
        try:
            call()
            row = dict(status="failed", reason="Invalid input was accepted")
        except (ValueError, TypeError) as error:
            after = interrupts()
            delta = after - before if after is not None and before is not None else None
            row = dict(
                status=(
                    "unverified"
                    if delta is None
                    else ("passed" if delta == 0 else "failed")
                ),
                exception=type(error).__name__,
                interrupts=delta,
            )
        except Exception as error:
            row = dict(status="failed", error=str(error))
        self.emit(dict(name=name, scope="Reject before hardware submission", **row))


def logic_call(checks, first, second, operation):
    if operation == "ABS":
        return checks.processor.sub(first, second, checks.vg.SubMode.ABS)
    return checks.processor.logic_op(
        first, second, getattr(checks.vg.LogicOp, operation)
    )


def check_logic(checks, width, height):
    left, right = pair_inputs(width, height)
    first, second = checks.image(left), checks.image(right)
    for operation in ("AND", "OR", "XOR", "ABS"):
        checks.exact(
            "%s_pairs_%dx%d" % (operation, width, height),
            lambda operation=operation: logic_call(checks, first, second, operation),
            [first, second],
            logic_reference(left, right, operation),
            (
                "All 65536 input pairs"
                if (width, height) == (256, 256)
                else "Row-varying pairs across padded image strides"
            ),
        )
        checks.exact(
            "%s_same_input_alias_%dx%d" % (operation, width, height),
            lambda operation=operation: logic_call(checks, first, first, operation),
            [first],
            logic_reference(left, left, operation),
            "The same ImageBuffer is supplied for both inputs; source bytes must remain intact",
        )
    check_shift(checks, first, second, left, right, width, height)
    check_shift_contract(checks, first, second, left, right, width, height)
    sequence = ["AND", "OR", "XOR", "ABS", "SHIFT", "AND", "ABS", "XOR", "OR", "AND"]
    steps = []
    try:
        for operation in sequence:
            call = (
                (lambda: checks.processor.sub(first, second, checks.vg.SubMode.SHIFT))
                if operation == "SHIFT"
                else (
                    lambda operation=operation: logic_call(
                        checks, first, second, operation
                    )
                )
            )
            actual, metadata = checks.capture(call, [first, second])
            comparison = (
                difference(actual, logic_reference(left, right, operation))
                if operation != "SHIFT"
                else None
            )
            status = checks.status(
                metadata,
                (
                    "passed"
                    if comparison is None or comparison["different_pixels"] == 0
                    else "failed"
                ),
            )
            steps.append(
                dict(
                    operation=operation,
                    status=status,
                    comparison=comparison,
                    sha256=metadata["sha256"],
                    metadata=metadata,
                )
            )
        checks.emit(
            dict(
                name="logic_mode_switch_%dx%d" % (width, height),
                status=(
                    "failed"
                    if any(step["status"] == "failed" for step in steps)
                    else (
                        "unverified"
                        if any(step["status"] == "unverified" for step in steps)
                        else "passed"
                    )
                ),
                scope="Known operations must remain exact after changing operation mode; this does not accept SHIFT arithmetic",
                steps=steps,
            )
        )
    except Exception as error:
        checks.emit(
            dict(
                name="logic_mode_switch_%dx%d" % (width, height),
                status="failed",
                error=str(error),
                steps=steps,
            )
        )


def check_shift(checks, first, second, left, right, width, height):
    try:
        actual, metadata = checks.capture(
            lambda: checks.processor.sub(first, second, checks.vg.SubMode.SHIFT),
            [first, second],
        )
        references = shift_candidates(left, right)
        comparisons = {
            name: difference(actual, reference)
            for name, reference in references.items()
        }
        equivalent = {}
        for name, reference in references.items():
            equivalent.setdefault(
                hashlib.sha256(reference.tobytes()).hexdigest(), []
            ).append(name)
        samples = [
            (0, 0),
            (0, min(width - 1, 1)),
            (1, 0),
            (0, min(width - 1, 255)),
            (min(height - 1, 255), 0),
            (min(height - 1, 127), min(width - 1, 128)),
        ]
        record = checks.raw("shift_pairs_%dx%d" % (width, height), actual)
        checks.emit(
            dict(
                name="SHIFT_pairs_%dx%d" % (width, height),
                status=checks.status(metadata, "unverified"),
                scope="SDK wording says one right shift to preserve the signed bit; representation and offset remain unresolved",
                candidates=comparisons,
                equivalent_candidate_groups=list(equivalent.values()),
                samples=[
                    dict(
                        row=y,
                        column=x,
                        a=int(left[y, x]),
                        b=int(right[y, x]),
                        actual=int(actual[y, x]),
                    )
                    for y, x in samples
                ],
                raw=record,
                **metadata
            )
        )
        alias, alias_metadata = checks.capture(
            lambda: checks.processor.sub(first, first, checks.vg.SubMode.SHIFT), [first]
        )
        values = sorted(set(alias.tobytes()))
        checks.emit(
            dict(
                name="SHIFT_same_input_alias_%dx%d" % (width, height),
                status=checks.status(
                    alias_metadata, "unverified" if len(values) == 1 else "failed"
                ),
                zero_difference_output_values=values,
                scope="Equal inputs must give a constant zero-difference encoding; its numeric encoding is not accepted here",
                **alias_metadata
            )
        )
        if (width, height) == (256, 256):
            reversed_output, reverse_metadata = checks.capture(
                lambda: checks.processor.sub(second, first, checks.vg.SubMode.SHIFT),
                [second, first],
            )
            comparison = difference(reversed_output, actual.T)
            checks.emit(
                dict(
                    name="SHIFT_pair_transpose_consistency",
                    status=checks.status(
                        reverse_metadata,
                        "passed" if comparison["different_pixels"] == 0 else "failed",
                    ),
                    scope="Pixelwise consistency under input exchange; this does not accept a numeric SHIFT model",
                    comparison=comparison,
                    **reverse_metadata
                )
            )
    except Exception as error:
        checks.emit(
            dict(
                name="SHIFT_pairs_%dx%d" % (width, height),
                status="failed",
                error=str(error),
            )
        )


def check_shift_contract(checks, first, second, left, right, width, height):
    """Check one fixed SHIFT mapping independently of exploratory candidate records.
    独立于探索性候选记录，检查一个固定的 SHIFT 映射。
    """
    checks.exact(
        "SHIFT_numeric_contract_pairs_%dx%d" % (width, height),
        lambda: checks.processor.sub(first, second, checks.vg.SubMode.SHIFT),
        [first, second],
        shift_numeric_reference(left, right),
        "Byte contract: signed nine-bit A-B, arithmetic right shift by one, low eight bits; full-pixel equality, without an internal-circuit claim",
    )
    checks.exact(
        "SHIFT_numeric_contract_same_input_alias_%dx%d" % (width, height),
        lambda: checks.processor.sub(first, first, checks.vg.SubMode.SHIFT),
        [first],
        np.zeros(left.shape, dtype=np.uint8),
        "The same input supplied twice must produce exactly zero in every byte under the fixed SHIFT contract",
    )


def threshold_pairs(quick=False):
    return (
        [(0, 0), (40, 200), (79, 80), (65535, 65535)]
        if quick
        else [
            (0, 0),
            (1, 1),
            (40, 200),
            (79, 79),
            (79, 80),
            (80, 80),
            (80, 81),
            (81, 81),
            (127, 128),
            (128, 128),
            (128, 129),
            (300, 900),
            (0, 65535),
            (65535, 65535),
        ]
    )


def check_canny_case(checks, source, name, mask_name, low, high, legacy=False):
    height, width = source.shape
    mask = KERNELS[mask_name]
    buffer = checks.image(source)
    identifier = "canny_%s_%s_%s_%d_%d_%dx%d" % (
        "default" if legacy else "ex",
        name,
        mask_name,
        low,
        high,
        width,
        height,
    )
    call = (
        (lambda: checks.processor.canny(buffer, high, low))
        if legacy
        else (lambda: checks.processor.canny_ex(buffer, high, low, mask))
    )
    try:
        references = {
            (border_name + ("_L2" if l2 else "_L1")): canny_reference(
                source, mask, low, high, l2, border
            )
            for border_name, border in BORDERS.items()
            for l2 in (False, True)
        }
    except Exception as error:
        checks.emit(
            dict(
                name=identifier,
                status="unverified",
                scope="Software derivative-Canny reference setup",
                reference_error=str(error),
            )
        )
        return
    try:
        actual, metadata = checks.capture(call, [buffer])
        primary = references["replicate_L1"]
        comparisons = {
            key: dict(
                full=difference(actual, reference),
                interior=difference(actual, reference, 4),
                nonzero_support_interior=difference(
                    cv2.compare(actual, 0, cv2.CMP_GT), reference, 4
                ),
                boundary_different_pixels=boundary_difference(actual, reference, 4),
            )
            for key, reference in references.items()
        }
        matched = comparisons["replicate_L1"]["interior"]["different_pixels"] == 0
        numeric_status = "passed" if matched else "unverified"
        scope = "Exact agreement with OpenCV derivative-overload L1 Canny, excluding four boundary pixels"
        analytic = None
        if name == "blank" or high == 65535:
            expected = np.zeros(source.shape, dtype=np.uint8)
            analytic = difference(actual, expected, 4)
            numeric_status = "passed" if analytic["different_pixels"] == 0 else "failed"
            scope = "Zero interior edges for a constant image or a threshold exceeding the supplied derivative bound"
        if (
            low == 0
            and high == 0
            and name != "blank"
            and cv2.countNonZero(primary[4:-4, 4:-4])
            and cv2.countNonZero(actual[4:-4, 4:-4]) == 0
        ):
            numeric_status = "failed"
        row = dict(
            name=identifier,
            status=checks.status(metadata, numeric_status),
            scope=scope,
            primary_reference="replicate_L1",
            reference_preblur=False,
            comparisons=comparisons,
            analytic_comparison=analytic,
            border_status="unverified",
            unresolved="Hardware NMS ties, angle-bin boundaries and hysteresis may differ; alternative model matches do not replace the primary reference",
            output_values=sorted(set(actual.tobytes())),
            low=low,
            high=high,
            mask=mask,
            input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
            **metadata
        )
        if not matched and checks.edge_artifacts < checks.max_edge_artifacts:
            row["raw_actual"] = checks.raw(identifier, actual)
            row["raw_primary_reference"] = checks.raw(
                identifier + "_opencv_L1", primary
            )
            checks.edge_artifacts += 1
        checks.emit(row)
    except Exception as error:
        checks.emit(dict(name=identifier, status="failed", error=str(error)))


def check_canny(checks, width, height, quick=False):
    inputs = patterns(width, height)
    selected_patterns = (
        ("blank", "slanted", "connected_weak") if quick else tuple(inputs)
    )
    selected_kernels = ("sobel3", "difference") if quick else tuple(KERNELS)
    available = hasattr(checks.processor, "canny_ex")
    for pattern_name in selected_patterns:
        for mask_name in selected_kernels:
            if not available and mask_name != "sobel3":
                continue
            for low, high in threshold_pairs(quick):
                check_canny_case(
                    checks,
                    inputs[pattern_name],
                    pattern_name,
                    mask_name,
                    low,
                    high,
                    not available,
                )
                if available and mask_name == "sobel3":
                    check_canny_case(
                        checks,
                        inputs[pattern_name],
                        pattern_name,
                        mask_name,
                        low,
                        high,
                        True,
                    )
    if not available:
        checks.emit(
            dict(
                name="canny_ex_missing",
                status="unverified",
                requirement="Expose the configurable 25-coefficient Canny kernel",
            )
        )
        return
    buffer = checks.image(inputs["connected_weak"])
    before_source = image_array(buffer)
    outputs, steps = [], []
    try:
        for name in ("sobel3", "sobel5", "difference", "sobel3"):
            actual, metadata = checks.capture(
                lambda name=name: checks.processor.canny_ex(
                    buffer, 200, 40, KERNELS[name]
                ),
                [buffer],
            )
            steps.append(
                dict(
                    mask=name,
                    status=checks.status(metadata, "passed"),
                    metadata=metadata,
                )
            )
            outputs.append((name, actual, metadata["sha256"]))
        repeated = difference(outputs[0][1], outputs[-1][1])["different_pixels"] == 0
        changed = len({row[2] for row in outputs}) > 1
        preserved = (
            difference(image_array(buffer), before_source)["different_pixels"] == 0
        )
        checks.emit(
            dict(
                name="canny_mask_switch_%dx%d" % (width, height),
                status=(
                    "failed"
                    if (
                        not (repeated and changed and preserved)
                        or any(step["status"] == "failed" for step in steps)
                    )
                    else (
                        "unverified"
                        if any(step["status"] == "unverified" for step in steps)
                        else "passed"
                    )
                ),
                scope="Repeatability, mask sensitivity and source preservation; independent pixel references are checked separately",
                repeated_first_mask=repeated,
                mask_changes_observed=changed,
                inputs_preserved=preserved,
                hashes=[dict(mask=name, sha256=digest) for name, _, digest in outputs],
                steps=steps,
            )
        )
        actual, metadata = checks.capture(
            lambda: checks.processor.canny(buffer, 200, 40), [buffer]
        )
        comparison = difference(actual, outputs[0][1])
        checks.emit(
            dict(
                name="canny_default_vs_ex_%dx%d" % (width, height),
                status=checks.status(
                    metadata,
                    "passed" if comparison["different_pixels"] == 0 else "failed",
                ),
                scope="Integration equality of the default helper and explicit default kernel",
                comparison=comparison,
                **metadata
            )
        )
    except Exception as error:
        checks.emit(
            dict(
                name="canny_mask_switch_%dx%d" % (width, height),
                status="failed",
                error=str(error),
                steps=steps,
            )
        )
    checks.emit(
        dict(
            name="canny_unresolved_conventions_%dx%d" % (width, height),
            status="unverified",
            scope="Boundary extension, NMS equality/angle bins and hysteresis connectivity require separate evidence; scoped OpenCV matches do not resolve these conventions",
        )
    )


def check_invalid(checks):
    source = np.zeros((240, 322), dtype=np.uint8)
    first, second = checks.image(source), checks.image(source[:, :320].copy())
    wrong_type = checks.image(source.astype(np.int16), "s16c1")
    for name in ("AND", "OR", "XOR", "ABS"):
        checks.reject(
            "%s_mismatched_dimensions" % name,
            lambda name=name: logic_call(checks, first, second, name),
        )
        checks.reject(
            "%s_wrong_dtype" % name,
            lambda name=name: logic_call(checks, first, wrong_type, name),
        )
    checks.reject(
        "canny_reversed_thresholds", lambda: checks.processor.canny(first, 10, 11)
    )
    checks.reject(
        "canny_wrong_dtype", lambda: checks.processor.canny(wrong_type, 100, 50)
    )
    for value in (-1, 65536):
        checks.reject(
            "canny_high_out_of_range_%d" % value,
            lambda value=value: checks.processor.canny(first, value, 0),
        )
        checks.reject(
            "canny_low_out_of_range_%d" % value,
            lambda value=value: checks.processor.canny(first, 65535, value),
        )
    if hasattr(checks.processor, "canny_ex"):
        for length in (24, 26):
            checks.reject(
                "canny_ex_mask_length_%d" % length,
                lambda length=length: checks.processor.canny_ex(
                    first, 100, 50, [0] * length
                ),
            )
        for value in (-129, 128):
            mask = [0] * 25
            mask[12] = value
            checks.reject(
                "canny_ex_coefficient_%d" % value,
                lambda mask=mask: checks.processor.canny_ex(first, 100, 50, mask),
            )


def self_check():
    """Validate the independent references and demonstrate fault detection.
    检查独立参考，并演示故障可被检测。
    """
    cv2.setNumThreads(1)
    left, right = pair_inputs(256, 256)
    pairs = set(zip(left.tobytes(), right.tobytes()))
    if len(pairs) != 65536:
        raise AssertionError("Pair grid is not exhaustive")
    for operation, actual in [
        ("AND", cv2.bitwise_and(left, right)),
        ("OR", cv2.bitwise_or(left, right)),
        ("XOR", cv2.bitwise_xor(left, right)),
        ("ABS", cv2.absdiff(left, right)),
    ]:
        if difference(actual, logic_reference(left, right, operation))[
            "different_pixels"
        ]:
            raise AssertionError("Scalar logic reference disagrees with OpenCV")
    source = patterns(128, 96)
    gradient_checks, canny_checks = [], []
    for mask_name, size in (("sobel3", 3), ("sobel5", 5)):
        for pattern_name in ("slanted", "corner", "connected_weak", "noise"):
            data = source[pattern_name]
            dx, dy = gradients(data, KERNELS[mask_name])
            sobel_x = cv2.Sobel(
                data, cv2.CV_16S, 1, 0, ksize=size, borderType=cv2.BORDER_REPLICATE
            )
            sobel_y = cv2.Sobel(
                data, cv2.CV_16S, 0, 1, ksize=size, borderType=cv2.BORDER_REPLICATE
            )
            comparisons = [difference(dx, sobel_x), difference(dy, sobel_y)]
            if any(row["different_pixels"] for row in comparisons):
                raise AssertionError(
                    "Explicit derivative coefficients disagree with Sobel"
                )
            gradient_checks.append(
                dict(mask=mask_name, pattern=pattern_name, comparisons=comparisons)
            )
            for l2 in (False, True):
                actual = cv2.Canny(dx, dy, 50, 100, L2gradient=l2)
                builtin = cv2.Canny(data, 50, 100, apertureSize=size, L2gradient=l2)
                comparison = difference(actual, builtin)
                if comparison["different_pixels"]:
                    raise AssertionError(
                        "Derivative Canny overload disagrees with the built-in Sobel path"
                    )
                canny_checks.append(
                    dict(
                        mask=mask_name,
                        pattern=pattern_name,
                        l2=l2,
                        comparison=comparison,
                    )
                )
    for pattern_name in ("slanted", "corner", "connected_weak", "noise"):
        data = source[pattern_name]
        dx, dy = gradients(data, DIFFERENCE)
        padded = cv2.copyMakeBorder(data, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(
            np.int16
        )
        expected_x = padded[1:-1, 2:] - padded[1:-1, :-2]
        expected_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
        comparisons = [difference(dx, expected_x), difference(dy, expected_y)]
        if any(row["different_pixels"] for row in comparisons):
            raise AssertionError(
                "Difference kernel disagrees with direct neighbor subtraction"
            )
        gradient_checks.append(
            dict(mask="difference", pattern=pattern_name, comparisons=comparisons)
        )
    data = source["step20"]
    dx, dy = gradients(data, SOBEL3)
    correct = cv2.Canny(dx, dy, 40, 79, L2gradient=False)
    wrong_threshold = cv2.Canny(dx, dy, 65535, 65535, L2gradient=False)
    wrong_derivative = cv2.Canny(
        np.zeros(dx.shape, dtype=np.int16), dy, 40, 79, L2gradient=False
    )
    threshold_difference, derivative_difference = difference(
        correct, wrong_threshold
    ), difference(correct, wrong_derivative)
    if (
        not threshold_difference["different_pixels"]
        or not derivative_difference["different_pixels"]
    ):
        raise AssertionError("Fault injection did not distinguish the incorrect result")
    candidates = shift_candidates(left, right)
    if difference(
        candidates["arithmetic_half_twos_complement"],
        candidates["logical_nine_bit_half"],
    )["different_pixels"]:
        raise AssertionError(
            "Equivalent SHIFT candidates should have identical byte encodings"
        )
    if not difference(
        candidates["biased_255_half_floor"], candidates["biased_256_half_floor"]
    )["different_pixels"]:
        raise AssertionError("SHIFT candidate comparison does not discriminate offsets")
    fixed_shift = shift_numeric_reference(left, right)
    fixed_shift_comparison = difference(
        fixed_shift, candidates["arithmetic_half_twos_complement"]
    )
    if fixed_shift_comparison["different_pixels"]:
        raise AssertionError(
            "Fixed SHIFT contract disagrees with arithmetic right shift"
        )
    shift_faults = {
        name: difference(fixed_shift, candidates[name])
        for name in (
            "truncate_half_twos_complement",
            "biased_255_half_floor",
            "biased_256_half_floor",
            "positive_saturated_half",
        )
    }
    if any(not comparison["different_pixels"] for comparison in shift_faults.values()):
        raise AssertionError(
            "Fixed SHIFT contract did not detect a wrong arithmetic model"
        )
    shift_alias = difference(
        shift_numeric_reference(left, left), np.zeros(left.shape, dtype=np.uint8)
    )
    if shift_alias["different_pixels"]:
        raise AssertionError("Fixed SHIFT same-input reference is not exactly zero")
    return dict(
        status="passed",
        host_only=True,
        opencv_version=cv2.__version__,
        exhaustive_pairs=len(pairs),
        gradient_checks=gradient_checks,
        derivative_canny_checks=canny_checks,
        shift_numeric_contract=dict(
            arithmetic_right_shift_comparison=fixed_shift_comparison,
            same_input_alias_comparison=shift_alias,
            fault_injection=shift_faults,
            scope="Software reference self-check; exploratory capture records remain unverified",
        ),
        fault_injection=dict(
            wrong_threshold_detected=True,
            wrong_threshold_comparison=threshold_difference,
            wrong_derivative_detected=True,
            wrong_derivative_comparison=derivative_difference,
        ),
        hardware_not_tested=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_edges_logic.json"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--logic-sizes", default="256x256,322x242")
    parser.add_argument("--edge-sizes", default="320x240,322x242")
    parser.add_argument("--groups", default="logic,canny,invalid")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--max-edge-artifacts", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.loops < 1 or args.max_edge_artifacts < 0:
        parser.error("Loops must be positive and artifact limit nonnegative")
    if args.self_check:
        result = self_check()
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return

    def sizes(value):
        result = [
            tuple(int(part) for part in item.split("x")) for item in value.split(",")
        ]
        if any(len(size) != 2 or min(size) < 64 or max(size) > 4096 for size in result):
            raise ValueError("Invalid dimensions")
        return result

    try:
        logic_sizes, edge_sizes = sizes(args.logic_sizes), sizes(args.edge_sizes)
    except ValueError:
        parser.error("Sizes must contain WxH values in [64,4096]")
    groups = args.groups.split(",")
    if any(group not in ("logic", "canny", "invalid") for group in groups):
        parser.error("Groups must be logic,canny,invalid")
    import visiong as vg

    cv2.setNumThreads(1)
    raw_dir = args.raw_dir or args.output.parent / (args.output.stem + "_raw")
    checks = Checks(vg, args.output, raw_dir, args.loops, args.max_edge_artifacts)
    if "logic" in groups:
        if (256, 256) not in logic_sizes:
            logic_sizes.insert(0, (256, 256))
        for width, height in logic_sizes:
            check_logic(checks, width, height)
    if "canny" in groups:
        for width, height in edge_sizes:
            check_canny(checks, width, height, args.quick)
    if "invalid" in groups:
        check_invalid(checks)
    counts = Counter(row["status"] for row in checks.rows)
    print(
        json.dumps(
            dict(
                summary=dict(counts),
                complete=not counts["failed"] and not counts["unverified"],
            )
        ),
        flush=True,
    )
    if counts["failed"] or (args.require_complete and counts["unverified"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
