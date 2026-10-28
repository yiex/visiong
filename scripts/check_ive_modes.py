# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check IVE modes, saturation, geometry and unresolved numeric conventions.
检查 IVE 模式、饱和值、几何关系及尚未确定的数值约定。

Run on the target with its regular VisionG library. Unknown conventions are
reported as unverified; a matching numeric hypothesis is not an acceptance.
在目标设备上使用常规 VisionG 库运行。未知约定记为未验证，吻合的数值假设不算验收通过。
Separate contract checks test explicit proposed rules without reclassifying the
earlier observations. 独立契约检查验证明确提出的规则，不改写此前观测的验证状态。
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time


def interrupt_count():
    """Read the hardware interrupt counter. / 读取硬件中断计数。"""
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def image(data, pixel_format="gray8"):
    return vg.ImageBuffer.from_numpy(data, pixel_format)


def array(value):
    return value.to_numpy().copy()


def tile_values(values, width, height, dtype=None):
    """Use every supplied value without a random dependency. / 平铺全部测试值，不依赖随机模块。"""
    if dtype is None:
        dtype = np.uint8
    row = [values[x % len(values)] for x in range(width)]
    return np.asarray([row] * height, dtype=dtype)


def difference(actual, expected, border=0):
    """Compare integers exactly, including U64 values. / 精确比较整数，包括 U64 数值。"""
    if list(actual.shape) != list(expected.shape):
        return dict(different=None, maximum=None, shape_matches=False)
    left = actual[border:-border, border:-border] if border else actual
    right = expected[border:-border, border:-border] if border else expected
    if (
        left.dtype.str == right.dtype.str
        and left.dtype.kind in "uib"
        and left.tobytes() == right.tobytes()
    ):
        return dict(different=0, maximum=0, shape_matches=True)
    count, peak = 0, 0
    for a, b in zip(left.ravel().tolist(), right.ravel().tolist()):
        delta = abs(int(a) - int(b))
        count += delta != 0
        peak = max(peak, delta)
    return dict(different=count, maximum=peak, shape_matches=True)


def crop_candidates(actual, references, border=0):
    return {
        name: difference(actual, expected, border)
        for name, expected in references.items()
    }


class Report:
    def __init__(self, path, loops):
        self.path = path
        self.loops = loops
        self.rows = []
        self.contract_groups = set()
        self.processor = vg.IVE()

    def emit(self, row):
        self.rows.append(row)
        document = dict(
            schema=1,
            counts=dict(Counter(item["status"] for item in self.rows)),
            results=self.rows,
        )
        self.path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    def missing(self, name, requirement):
        self.emit(dict(name=name, status="unverified", requirement=requirement))

    def run(self, name, function, hardware=True):
        before = interrupt_count()
        start = time.perf_counter()
        try:
            row = function()
        except Exception as error:
            row = dict(status="failed", error=str(error))
        after = interrupt_count()
        row.update(name=name, elapsed_ms=(time.perf_counter() - start) * 1000)
        row["interrupts"] = (
            after - before if before is not None and after is not None else None
        )
        if hardware and row["status"] == "passed":
            if row["interrupts"] is None:
                row.update(status="unverified", reason="Hardware counter unavailable")
            elif row["interrupts"] <= 0:
                row.update(status="failed", reason="No hardware interrupt observed")
        self.emit(row)

    def exact(
        self, name, call, expected, border=0, expected_dtype=None, reference=None
    ):
        def check():
            signatures, comparisons, timings = [], [], []
            for _ in range(self.loops):
                start = time.perf_counter()
                actual = array(call())
                timings.append((time.perf_counter() - start) * 1000)
                signatures.append(hashlib.sha256(actual.tobytes()).hexdigest())
                comparisons.append(difference(actual, expected, border))
            matches = all(row["different"] == 0 for row in comparisons)
            type_matches = expected_dtype is None or actual.dtype.str == expected_dtype
            result = dict(
                status=(
                    "passed"
                    if matches and len(set(signatures)) == 1 and type_matches
                    else "failed"
                ),
                dtype=actual.dtype.str,
                expected_dtype=expected_dtype,
                shape=list(actual.shape),
                comparisons=comparisons,
                unique_outputs=len(set(signatures)),
                excluded_border=border,
                min_ms=min(timings),
                mean_ms=sum(timings) / len(timings),
            )
            if reference is not None:
                result["reference"] = reference
            return result

        self.run(name, check)

    def rejected(self, name, call):
        """Invalid controls must be rejected before a hardware submission.
        无效控制参数必须在提交硬件前被拒绝。
        """

        def check():
            before = interrupt_count()
            try:
                call()
            except (ValueError, TypeError) as error:
                after = interrupt_count()
                return dict(
                    status=(
                        "unverified"
                        if before is None or after is None
                        else ("passed" if after == before else "failed")
                    ),
                    exception=type(error).__name__,
                    reason="Invalid input must not submit a hardware job",
                )
            return dict(status="failed", reason="Invalid input was accepted")

        self.run(name, check, hardware=False)


def threshold_reference(source, mode, low, high, min_val=0, mid_val=0, max_val=255):
    """Follow rk_ive.h, including its TRUNC definition. / 按 rk_ive.h 定义构造参考，包括 TRUNC。"""
    output = source.copy()
    for y, row in enumerate(source.tolist()):
        for x, value in enumerate(row):
            if value <= low:
                result = value if mode in (1, 4, 7) else min_val
            elif mode < 3 or value > high:
                result = value if mode in (2, 5, 7) else max_val
            else:
                result = value if mode == 6 else mid_val
            output[y, x] = result
    return output


def check_threshold(report, width, height):
    source = tile_values(list(range(256)), width, height)
    buffer = image(source)
    names = (
        "BINARY",
        "TRUNC",
        "TO_MINVAL",
        "MIN_MID_MAX",
        "ORI_MID_MAX",
        "MIN_MID_ORI",
        "MIN_ORI_MAX",
        "ORI_MID_ORI",
    )
    for index, name in enumerate(names):
        mode = getattr(vg.ThreshMode, name, None)
        if mode is None:
            report.missing("threshold_" + name, "Expose ThreshMode." + name)
            continue
        for low, high in [(0, 0), (1, 2), (63, 192), (254, 255), (255, 255)]:
            expected = threshold_reference(source, index, low, high)
            report.exact(
                "threshold_%s_%d_%d_%dx%d" % (name, low, high, width, height),
                lambda mode=mode, low=low, high=high: report.processor.threshold(
                    buffer, low, high, mode
                ),
                expected,
            )
    if hasattr(report.processor, "threshold_u8"):
        for index, name in enumerate(names):
            mode = getattr(vg.ThreshMode, name, None)
            if mode is None:
                continue
            report.exact(
                "threshold_u8_values_%s_%dx%d" % (name, width, height),
                lambda mode=mode: report.processor.threshold_u8(
                    buffer, 63, 192, mode, 17, 89, 231
                ),
                threshold_reference(source, index, 63, 192, 17, 89, 231),
            )
    else:
        report.missing(
            "threshold_configurable_values",
            "Expose independent U8 min/mid/max values",
        )
    typed_thresholds(report, width, height)


def typed_thresholds(report, width, height):
    """Check representable ORI values exactly. / 精确检查 ORI 模式中可表示的数值。"""
    signed_values = [
        -32768,
        -1024,
        -129,
        -128,
        -64,
        -34,
        -33,
        -32,
        -1,
        0,
        1,
        6,
        7,
        8,
        46,
        47,
        48,
        90,
        91,
        92,
        127,
        128,
        255,
        256,
        1024,
        32767,
    ]
    unsigned_values = [
        0,
        1,
        6,
        7,
        8,
        90,
        91,
        92,
        127,
        128,
        255,
        256,
        1022,
        1023,
        1024,
        32767,
        32768,
        32769,
        65535,
    ]
    for signed, values, dtype, format_name, method_name, enum_name, names in [
        (
            True,
            signed_values,
            np.int16,
            "s16c1",
            "threshold_s16",
            "ThreshS16Mode",
            (
                "S16_TO_S8_MIN_MID_MAX",
                "S16_TO_S8_MIN_ORI_MAX",
                "S16_TO_U8_MIN_MID_MAX",
                "S16_TO_U8_MIN_ORI_MAX",
            ),
        ),
        (
            False,
            unsigned_values,
            np.uint16,
            "u16c1",
            "threshold_u16",
            "ThreshU16Mode",
            ("U16_TO_U8_MIN_MID_MAX", "U16_TO_U8_MIN_ORI_MAX"),
        ),
    ]:
        method = getattr(report.processor, method_name, None)
        enum = getattr(vg, enum_name, None)
        if method is None or enum is None:
            report.missing(method_name, "Expose " + method_name + " and " + enum_name)
            continue
        source = tile_values(values, width, height, dtype)
        buffer = image(source, format_name)
        for name in names:
            mode = getattr(enum, name, None)
            if mode is None:
                report.missing(
                    method_name + "_" + name, "Expose " + enum_name + "." + name
                )
                continue
            signed_output = "_S8_" in name
            output_dtype = np.int8 if signed_output else np.uint8
            low, high = (-33, 47) if signed_output else (7, 91)
            minimum, middle, maximum = (-110, 7, 111) if signed_output else (3, 37, 251)
            original = "_ORI_" in name
            wanted = [
                (
                    minimum
                    if value <= low
                    else (maximum if value > high else (value if original else middle))
                )
                for value in values
            ]
            expected = tile_values(wanted, width, height, output_dtype)
            report.exact(
                "%s_%s_%dx%d" % (method_name, name, width, height),
                lambda method=method, mode=mode, low=low, high=high, minimum=minimum, middle=middle, maximum=maximum: method(
                    buffer, low, high, mode, minimum, middle, maximum
                ),
                expected,
                expected_dtype=expected.dtype.str,
            )
            if original:

                def wide_range(
                    method=method,
                    mode=mode,
                    signed_output=signed_output,
                    minimum=minimum,
                    middle=middle,
                    maximum=maximum,
                ):
                    low, high = (-1024, 1024) if signed else (0, 32768)
                    actual = array(
                        method(buffer, low, high, mode, minimum, middle, maximum)
                    )
                    output_values = actual[0, : len(values)].tolist()
                    clamp_low, clamp_high = (-128, 127) if signed_output else (0, 255)
                    raw = [
                        (
                            minimum
                            if value <= low
                            else (maximum if value > high else value)
                        )
                        for value in values
                    ]
                    clipped = [max(clamp_low, min(clamp_high, value)) for value in raw]
                    wrapped = [
                        ((value + 128) % 256) - 128 if signed_output else value % 256
                        for value in raw
                    ]
                    return dict(
                        status="unverified",
                        input_values=values,
                        output_values=output_values,
                        clipped_values=clipped,
                        wrapped_values=wrapped,
                        reason="ORI converts a 16-bit middle interval outside the 8-bit output range; establish clipping or wrapping explicitly",
                    )

                report.run(
                    "%s_%s_wide_ori_%dx%d" % (method_name, name, width, height),
                    wide_range,
                )
    check_word_ori_contract(report)


def raw_low_byte_threshold(values, low, high, minimum, maximum, signed_output):
    """Proposed ORI rule: retain the raw low byte, not a saturated value.
    待确立的 ORI 规则：保留原始低字节，而不是饱和值。
    """
    output = []
    for value in values:
        if value <= low:
            result = minimum
        elif value > high:
            result = maximum
        else:
            result = value & 255
            if signed_output and result >= 128:
                result -= 256
        output.append(result)
    return output


def check_word_ori_contract(report):
    """Test every word value independently from sparse convention observations.
    独立于稀疏约定观测，逐个验证全部字输入值。
    """
    if "word_ori_full_domain" in report.contract_groups:
        return
    report.contract_groups.add("word_ori_full_domain")
    modes = [
        (
            "threshold_s16",
            "ThreshS16Mode",
            "S16_TO_S8_MIN_ORI_MAX",
            True,
            True,
        ),
        (
            "threshold_s16",
            "ThreshS16Mode",
            "S16_TO_U8_MIN_ORI_MAX",
            True,
            False,
        ),
        (
            "threshold_u16",
            "ThreshU16Mode",
            "U16_TO_U8_MIN_ORI_MAX",
            False,
            False,
        ),
    ]
    for method_name, enum_name, mode_name, signed_input, signed_output in modes:
        method = getattr(report.processor, method_name, None)
        enum = getattr(vg, enum_name, None)
        mode = getattr(enum, mode_name, None)
        if method is None or mode is None:
            report.missing(
                "threshold_raw_low_byte_contract_" + mode_name,
                "Expose " + method_name + " and " + enum_name + "." + mode_name,
            )
            continue
        values = list(range(-32768, 32768) if signed_input else range(65536))
        dtype = np.int16 if signed_input else np.uint16
        source = np.asarray(
            [values[start : start + 256] for start in range(0, 65536, 256)],
            dtype=dtype,
        )
        buffer = image(source, "s16c1" if signed_input else "u16c1")
        minimum, middle, maximum = (-110, 7, 111) if signed_output else (3, 37, 251)
        intervals = (
            [
                (-32768, 32767),
                (-1024, 1024),
                (-129, 128),
                (-1, 0),
                (-32768, -32767),
                (32766, 32767),
            ]
            if signed_input
            else [
                (0, 65535),
                (0, 32768),
                (127, 128),
                (255, 256),
                (32767, 32768),
                (65534, 65535),
            ]
        )
        for low, high in intervals:
            wanted = raw_low_byte_threshold(
                values, low, high, minimum, maximum, signed_output
            )
            expected = np.asarray(
                [wanted[start : start + 256] for start in range(0, 65536, 256)],
                dtype=np.int8 if signed_output else np.uint8,
            )
            report.exact(
                "threshold_raw_low_byte_contract_%s_%d_%d_full65536"
                % (mode_name, low, high),
                lambda method=method, mode=mode, low=low, high=high, minimum=minimum, middle=middle, maximum=maximum, buffer=buffer: method(
                    buffer, low, high, mode, minimum, middle, maximum
                ),
                expected,
                expected_dtype=expected.dtype.str,
                reference=dict(
                    contract="proposed_raw_low_byte_v1",
                    rule="src<=low selects min; src>high selects max; otherwise retain src&255 and interpret as the output dtype",
                    distinct_input_values=65536,
                    input_range=[values[0], values[-1]],
                    input_shape=[256, 256],
                    threshold_interval=[low, high],
                    acceptance="This explicit rule is tested independently; earlier sparse ORI observations remain unverified",
                ),
            )


def check_cast(report, width, height):
    source = tile_values(list(range(256)), width, height)
    buffer = image(source)
    for bias, numerator, denominator in [
        (0, 1, 1),
        (0, 2, 1),
        (127, 1, 1),
        (-128, 1, 1),
        (0, 0, 1),
    ]:
        expected = tile_values(
            [max(0, min(255, value * numerator + bias)) for value in range(256)],
            width,
            height,
        )
        report.exact(
            "cast8_exact_%d_%d_%d_%dx%d"
            % (bias, numerator, denominator, width, height),
            lambda bias=bias, numerator=numerator, denominator=denominator: report.processor.cast_8bit_to_8bit(
                buffer, bias, numerator, denominator
            ),
            expected,
        )

    def fractional8():
        actual = array(report.processor.cast_8bit_to_8bit(buffer, 0, 1, 3))
        candidates = {
            "floor": tile_values([value // 3 for value in range(256)], width, height),
            "nearest": tile_values(
                [(value + 1) // 3 for value in range(256)], width, height
            ),
            "ceil": tile_values(
                [(value + 2) // 3 for value in range(256)], width, height
            ),
        }
        return dict(
            status="unverified",
            reason="Fractional rounding is not specified by the public API",
            candidates=crop_candidates(actual, candidates),
            samples=actual[0, :32].tolist(),
        )

    report.run("cast8_fractional_%dx%d" % (width, height), fractional8)
    for numerator, denominator, bias in [(1, 2, 3), (3, 2, -3), (5, 7, 9)]:
        expected = tile_values(
            [
                cast_reference(value, "U8_TO_U8", numerator, denominator, bias)
                for value in range(256)
            ],
            width,
            height,
        )
        report.exact(
            "cast8_q24_%d_%d_bias%d_%dx%d"
            % (numerator, denominator, bias, width, height),
            lambda numerator=numerator, denominator=denominator, bias=bias: report.processor.cast_8bit_to_8bit(
                buffer, bias, numerator, denominator
            ),
            expected,
        )
    signed_values = [
        -32768,
        -16384,
        -1024,
        -512,
        -508,
        -128,
        -4,
        0,
        4,
        128,
        508,
        512,
        1024,
        16384,
        32767,
    ]
    unsigned_values = [0, 4, 128, 508, 1020, 1024, 4096, 32768, 65532]
    signed = image(tile_values(signed_values, width, height, np.int16), "s16c1")
    unsigned = image(tile_values(unsigned_values, width, height, np.uint16), "u16c1")
    signed_s8_values = [
        -32768,
        -16384,
        -1024,
        -512,
        -128,
        0,
        128,
        512,
        1024,
        16384,
        32767,
    ]
    signed_s8 = image(tile_values(signed_s8_values, width, height, np.int16), "s16c1")
    cases = [
        (
            "S16_TO_S8",
            signed_s8,
            signed_s8_values,
            lambda value: max(-128, min(127, value // 128)),
            np.int8,
        ),
        (
            "S16_TO_U8_ABS",
            signed,
            signed_values,
            lambda value: min(255, abs(value) // 4),
            np.uint8,
        ),
        (
            "S16_TO_U8_BIAS",
            signed,
            signed_values,
            lambda value: max(0, min(255, (value + 508) // 4)),
            np.uint8,
        ),
        (
            "U16_TO_U8",
            unsigned,
            unsigned_values,
            lambda value: min(255, value // 4),
            np.uint8,
        ),
    ]
    for name, source_buffer, values, formula, dtype in cases:
        mode = getattr(vg.Cast16to8Mode, name)
        expected = tile_values(
            [formula(value) for value in values], width, height, dtype
        )
        report.exact(
            "cast16_exact_%s_%dx%d" % (name, width, height),
            lambda source_buffer=source_buffer, mode=mode: report.processor.cast_16bit_to_8bit(
                source_buffer, mode
            ),
            expected,
            expected_dtype=expected.dtype.str,
        )
    row_values = np.asarray(
        [
            [(17 * x + 43 * y + (x * y) % 31) % 256 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    row_source = (row_values.astype(np.uint16) * 4).astype(np.uint16)
    row_buffer = image(row_source, "u16c1")
    report.exact(
        "cast16_U16_row_varying_%dx%d" % (width, height),
        lambda: report.processor.cast_16bit_to_8bit(
            row_buffer, vg.Cast16to8Mode.U16_TO_U8
        ),
        row_values,
        expected_dtype="|u1",
    )
    fractional_values = [
        -513,
        -511,
        -510,
        -509,
        -507,
        -506,
        -505,
        -129,
        -127,
        -65,
        -64,
        -63,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        63,
        64,
        65,
        127,
        129,
        505,
        506,
        507,
        509,
        510,
        511,
        513,
    ]
    source = image(tile_values(fractional_values, width, height, np.int16), "s16c1")
    for name in ("S16_TO_S8", "S16_TO_U8_ABS", "S16_TO_U8_BIAS"):

        def fractional16(name=name):
            actual = array(
                report.processor.cast_16bit_to_8bit(
                    source, getattr(vg.Cast16to8Mode, name)
                )
            )
            return dict(
                status="unverified",
                reason="Record exact fractional results before selecting a rounding contract",
                input_values=fractional_values,
                output_values=actual[0, : len(fractional_values)].tolist(),
                dtype=actual.dtype.str,
            )

        report.run("cast16_fractional_%s_%dx%d" % (name, width, height), fractional16)
    check_legacy_cast_contract(report)
    check_cast_ex(report, width, height)
    check_cast16_boundary_contract(report)


def cast_reference(value, mode, numerator, denominator, bias):
    """Apply signed Q24 scaling, ties toward positive infinity, then bias.
    执行有符号 Q24 缩放，等距时向正无穷舍入，然后添加偏置。
    """
    factor = ((numerator << 24) + denominator // 2) // denominator
    source = abs(value) if mode.endswith("_ABS") else value
    scaled = (source * factor + (1 << 23)) // (1 << 24)
    low, high = (-128, 127) if mode.endswith("_TO_S8") else (0, 255)
    return max(low, min(high, scaled + bias))


def full_word_cast_source(unsigned):
    """Place every word value in a dedicated 256x256 image.
    使用专用 256x256 图像容纳全部字输入值。
    """
    values = list(range(65536) if unsigned else range(-32768, 32768))
    source = np.asarray(
        [values[start : start + 256] for start in range(0, 65536, 256)],
        dtype=np.uint16 if unsigned else np.int16,
    )
    return values, image(source, "u16c1" if unsigned else "s16c1")


def cast_contract_expected(values, mode, numerator, denominator, bias):
    output = [
        cast_reference(value, mode, numerator, denominator, bias) for value in values
    ]
    return np.asarray(
        [output[start : start + 256] for start in range(0, 65536, 256)],
        dtype=np.int8 if mode.endswith("_TO_S8") else np.uint8,
    )


def check_legacy_cast_contract(report):
    """Test fixed legacy controls separately from historical observations.
    单独严格验证旧接口的固定控制值，不改写历史观测。
    """
    if "legacy_cast_q24" in report.contract_groups:
        return
    report.contract_groups.add("legacy_cast_q24")
    legacy8 = getattr(report.processor, "cast_8bit_to_8bit", None)
    if legacy8 is None:
        report.missing("legacy_q24_contract_U8_1_3", "Expose cast_8bit_to_8bit")
    else:
        byte_values = list(range(256)) * 256
        source = np.asarray([list(range(256))] * 256, dtype=np.uint8)
        buffer = image(source)
        expected = cast_contract_expected(byte_values, "U8_TO_U8", 1, 3, 0)
        report.exact(
            "legacy_q24_contract_U8_TO_U8_1_3_bias0_full256",
            lambda: legacy8(buffer, 0, 1, 3),
            expected,
            expected_dtype="|u1",
            reference=dict(
                contract="legacy_q24_contract_v1",
                controls=dict(numerator=1, denominator=3, bias=0),
                distinct_input_values=256,
                input_shape=[256, 256],
                rule="Q24 ratio rounded to nearest; signed scaled value rounds half toward positive infinity, then bias and dtype saturation",
                historical_observations="retained as unverified",
            ),
        )
    legacy16 = getattr(report.processor, "cast_16bit_to_8bit", None)
    enum = getattr(vg, "Cast16to8Mode", None)
    if legacy16 is None or enum is None:
        report.missing("legacy_q24_contract_word_modes", "Expose legacy 16-bit casts")
        return
    signed_values, signed_buffer = full_word_cast_source(False)
    unsigned_values, unsigned_buffer = full_word_cast_source(True)
    for name, numerator, denominator, bias in [
        ("S16_TO_S8", 1, 128, 0),
        ("S16_TO_U8_ABS", 1, 4, 0),
        ("S16_TO_U8_BIAS", 1, 4, 127),
        ("U16_TO_U8", 1, 4, 0),
    ]:
        mode = getattr(enum, name, None)
        if mode is None:
            report.missing(
                "legacy_q24_contract_" + name, "Expose Cast16to8Mode." + name
            )
            continue
        unsigned = name.startswith("U")
        values = unsigned_values if unsigned else signed_values
        buffer = unsigned_buffer if unsigned else signed_buffer
        expected = cast_contract_expected(values, name, numerator, denominator, bias)
        report.exact(
            "legacy_q24_contract_%s_%d_%d_bias%d_full65536"
            % (name, numerator, denominator, bias),
            lambda buffer=buffer, mode=mode: legacy16(buffer, mode),
            expected,
            expected_dtype=expected.dtype.str,
            reference=dict(
                contract="legacy_q24_contract_v1",
                fixed_controls=dict(
                    numerator=numerator, denominator=denominator, bias=bias
                ),
                distinct_input_values=65536,
                input_range=[values[0], values[-1]],
                input_shape=[256, 256],
                rule="Apply the explicit Q24 reference to every input; ABS precedes scaling, bias follows signed half-to-positive rounding",
                historical_observations="retained as unverified",
            ),
        )


def check_cast16_boundary_contract(report):
    """Check numeric parameter boundaries with complete input domains.
    使用完整输入域检查数值控制参数边界。
    """
    if "cast16_numeric_boundaries" in report.contract_groups:
        return
    report.contract_groups.add("cast16_numeric_boundaries")
    method = getattr(report.processor, "cast_16bit_ex", None)
    enum = getattr(vg, "Cast16to8Mode", None)
    if method is None or enum is None:
        report.missing("cast16_q24_boundary_contract", "Expose cast_16bit_ex controls")
        return
    signed_values, signed_buffer = full_word_cast_source(False)
    unsigned_values, unsigned_buffer = full_word_cast_source(True)
    cases = [
        ("S16_TO_S8", 1, 65535, 0),
        ("U16_TO_U8", 255, 65535, -128),
        ("U16_TO_U8", 255, 1, -128),
        ("S16_TO_S8", 255, 1, -128),
    ] + [
        (name, 0, 1, 127)
        for name in ("S16_TO_S8", "S16_TO_U8_ABS", "S16_TO_U8_BIAS", "U16_TO_U8")
    ]
    for name, numerator, denominator, bias in cases:
        mode = getattr(enum, name, None)
        if mode is None:
            report.missing(
                "cast16_q24_boundary_" + name, "Expose Cast16to8Mode." + name
            )
            continue
        unsigned = name.startswith("U")
        values = unsigned_values if unsigned else signed_values
        buffer = unsigned_buffer if unsigned else signed_buffer
        expected = cast_contract_expected(values, name, numerator, denominator, bias)
        reference = dict(
            contract="cast_q24_numeric_boundaries_v1",
            controls=dict(numerator=numerator, denominator=denominator, bias=bias),
            distinct_input_values=65536,
            input_range=[values[0], values[-1]],
            input_shape=[256, 256],
        )
        if (name, numerator, denominator, bias) == ("S16_TO_S8", 1, 65535, 0):
            reference["distinguishing_fixture"] = dict(
                input=-32768, q24_result=0, exact_rational_result=-1
            )
        report.exact(
            "cast16_q24_boundary_%s_%d_%d_bias%d_full65536"
            % (name, numerator, denominator, bias),
            lambda buffer=buffer, mode=mode, numerator=numerator, denominator=denominator, bias=bias: method(
                buffer, mode, numerator, denominator, bias
            ),
            expected,
            expected_dtype=expected.dtype.str,
            reference=reference,
        )


def check_cast_ex(report, width, height):
    for bits, method_name, enum_name, names in [
        (
            8,
            "cast_8bit_ex",
            "Cast8to8Mode",
            ("S8_TO_S8", "S8_TO_U8_ABS", "S8_TO_U8_BIAS", "U8_TO_U8"),
        ),
        (
            16,
            "cast_16bit_ex",
            "Cast16to8Mode",
            ("S16_TO_S8", "S16_TO_U8_ABS", "S16_TO_U8_BIAS", "U16_TO_U8"),
        ),
    ]:
        method, enum = getattr(report.processor, method_name, None), getattr(
            vg, enum_name, None
        )
        if method is None or enum is None:
            report.missing(
                method_name,
                "Expose all cast modes and numerator/denominator/bias controls",
            )
            continue
        for name in names:
            mode = getattr(enum, name, None)
            if mode is None:
                report.missing(
                    method_name + "_" + name, "Expose " + enum_name + "." + name
                )
                continue
            unsigned = name.startswith("U")
            lower = 0 if unsigned else -(1 << (bits - 1))
            # Use a dedicated image when the requested geometry cannot hold the full domain.
            # 请求的几何尺寸容纳不下完整输入域时，使用专用图像。
            test_width, test_height = (
                (256, 256) if bits == 16 and width * height < 65536 else (width, height)
            )
            size_label = "%dx%d" % (width, height)
            if (test_width, test_height) != (width, height):
                size_label += "_full_domain256x256"
            values = list(range(lower, lower + (1 << bits)))
            dtype = (
                (np.uint8 if unsigned else np.int8)
                if bits == 8
                else (np.uint16 if unsigned else np.int16)
            )
            format_name = (
                ("gray8" if unsigned else "s8c1")
                if bits == 8
                else ("u16c1" if unsigned else "s16c1")
            )
            source = np.asarray(
                [
                    [
                        values[(y * test_width + x) % len(values)]
                        for x in range(test_width)
                    ]
                    for y in range(test_height)
                ],
                dtype=dtype,
            )
            buffer = image(source, format_name)
            for numerator, denominator, bias in [
                (1, 1, 0),
                (1, 2, 3),
                (3, 2, -3),
                (5, 7, 9),
            ]:
                output_dtype = np.int8 if name.endswith("_TO_S8") else np.uint8
                lut = [
                    cast_reference(value, name, numerator, denominator, bias)
                    for value in values
                ]
                expected = np.asarray(
                    [
                        [
                            lut[(y * test_width + x) % len(lut)]
                            for x in range(test_width)
                        ]
                        for y in range(test_height)
                    ],
                    dtype=output_dtype,
                )
                report.exact(
                    "%s_%s_%d_%d_bias%d_%s"
                    % (method_name, name, numerator, denominator, bias, size_label),
                    lambda method=method, buffer=buffer, mode=mode, numerator=numerator, denominator=denominator, bias=bias: method(
                        buffer, mode, numerator, denominator, bias
                    ),
                    expected,
                    expected_dtype=expected.dtype.str,
                    reference=dict(
                        contract="cast_q24_v1",
                        distinct_input_values=1 << bits,
                        requested_shape=[height, width],
                        input_shape=[test_height, test_width],
                    ),
                )
            report.rejected(
                "%s_%s_zero_denominator_%s" % (method_name, name, size_label),
                lambda method=method, buffer=buffer, mode=mode: method(
                    buffer, mode, 1, 0, 0
                ),
            )


def prefix_reference(source, squared=False):
    width = source.shape[1]
    total = [0] * width
    rows = []
    for row in source.tolist():
        running = 0
        result = []
        for x, value in enumerate(row):
            running += value * value if squared else value
            total[x] += running
            result.append(total[x])
        rows.append(result)
    return np.asarray(rows, dtype=np.uint64)


def combined_integral_reference(source):
    """Pack independently accumulated 28-bit sum and 36-bit square sum.
    将独立累加的 28 位积分和与 36 位平方积分和打包。
    """
    summed = prefix_reference(source).tolist()
    squared = prefix_reference(source, True).tolist()
    if summed[-1][-1] >= (1 << 28) or squared[-1][-1] >= (1 << 36):
        raise ValueError("Reference input exceeds the established COMBINE field ranges")
    return np.asarray(
        [
            [(square << 28) | value for value, square in zip(row, square_row)]
            for row, square_row in zip(summed, squared)
        ],
        dtype=np.uint64,
    )


def check_integral(report, width, height):
    source = np.asarray(
        [
            [(x * 17 + y * 29 + (x * y) % 71) % 256 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    buffer = image(source)
    summed, squared = prefix_reference(source), prefix_reference(source, True)
    for name, expected in [("SUM", summed), ("SQSUM", squared)]:
        report.exact(
            "integral_%s_%dx%d" % (name, width, height),
            lambda name=name: report.processor.integral(
                buffer, getattr(vg.IntegOutCtrl, name)
            ),
            expected,
            expected_dtype="<u8",
        )

    def combined():
        actual = array(report.processor.integral(buffer, vg.IntegOutCtrl.COMBINE))
        values = actual.ravel().tolist()
        sums, squares = summed.ravel().tolist(), squared.ravel().tolist()
        candidates = []
        for bits in (24, 28, 32, 36, 40):
            mask = (1 << bits) - 1
            for low, high, label in [
                (sums, squares, "sum_low"),
                (squares, sums, "square_low"),
            ]:
                mismatch = sum(
                    value != ((int(b) << bits) | (int(a) & mask))
                    for value, a, b in zip(values, low, high)
                )
                candidates.append(
                    dict(low_bits=bits, packing=label, differing_pixels=mismatch)
                )
        samples = [(0, 0), (1, 1), (5, 7), (height - 1, width - 1)]
        return dict(
            status="unverified",
            reason="The public COMBINE bit layout has not been documented",
            dtype=actual.dtype.str,
            candidates=candidates,
            samples=[
                dict(
                    x=x,
                    y=y,
                    packed=int(actual[y, x]),
                    sum=int(summed[y, x]),
                    square=int(squared[y, x]),
                )
                for y, x in samples
            ],
        )

    report.run("integral_COMBINE_%dx%d" % (width, height), combined)
    for name, holdout in [
        ("one", np.ones((height, width), dtype=np.uint8)),
        ("maximum", np.full((height, width), 255, dtype=np.uint8)),
        (
            "alternating",
            np.asarray(
                [
                    [(255 if (x + y) & 1 else 0) for x in range(width)]
                    for y in range(height)
                ],
                dtype=np.uint8,
            ),
        ),
    ]:
        holdout_buffer = image(holdout)
        report.exact(
            "integral_COMBINE_holdout_%s_%dx%d" % (name, width, height),
            lambda holdout_buffer=holdout_buffer: report.processor.integral(
                holdout_buffer, vg.IntegOutCtrl.COMBINE
            ),
            combined_integral_reference(holdout),
            expected_dtype="<u8",
        )


def check_dma(report, width, height):
    source = np.asarray(
        [
            [(17 * x + 31 * y + x * y % 13) % 256 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    buffer = image(source)
    report.exact(
        "dma_direct_%dx%d" % (width, height),
        lambda: report.processor.dma(buffer, vg.DmaMode.DIRECT_COPY),
        source,
    )
    for name in ("SET_3BYTE", "SET_8BYTE"):
        report.exact(
            "dma_%s_zero_%dx%d" % (name, width, height),
            lambda name=name: report.processor.dma(buffer, getattr(vg.DmaMode, name)),
            np.zeros((height, width), dtype=np.uint8),
        )
    transfer = getattr(report.processor, "dma_transfer", None)
    if transfer is None:
        report.missing(
            "dma_interval_configuration",
            "Expose hor_seg_size, elem_size, ver_seg_rows and derived output dimensions",
        )
        report.missing("dma_set_nonzero", "Expose the fill value and output dimensions")
        return
    for segment, element, rows in [
        (2, 1, 1),
        (3, 2, 3),
        (4, 1, 2),
        (8, 3, 3),
        (16, 7, 5),
    ]:
        if width % segment or height % rows:
            continue
        expected = np.asarray(
            [
                [
                    int(source[y, x + offset])
                    for x in range(0, width, segment)
                    for offset in range(element)
                ]
                for y in range(0, height, rows)
            ],
            dtype=np.uint8,
        )
        out_height, out_width = expected.shape
        report.exact(
            "dma_interval_%d_%d_%d_%dx%d" % (segment, element, rows, width, height),
            lambda segment=segment, element=element, rows=rows: transfer(
                buffer, vg.DmaMode.INTERVAL_COPY, 0, segment, element, rows
            ),
            expected,
        )
        report.exact(
            "dma_interval_explicit_%d_%d_%d_%dx%d"
            % (segment, element, rows, width, height),
            lambda segment=segment, element=element, rows=rows, out_width=out_width, out_height=out_height: transfer(
                buffer,
                vg.DmaMode.INTERVAL_COPY,
                0,
                segment,
                element,
                rows,
                out_width,
                out_height,
            ),
            expected,
        )
    # Include a destination width not divisible by sixteen. / 加入输出宽度非十六整倍的情形。
    compact_width = (width // 4) * 4 - 4
    compact_source = source[:, :compact_width].copy()
    compact_buffer = image(compact_source)
    expected = compact_source[:, ::4].copy()
    report.exact(
        "dma_interval_unaligned_output_%dx%d" % (width, height),
        lambda: transfer(compact_buffer, vg.DmaMode.INTERVAL_COPY, 0, 4, 1, 1),
        expected,
    )
    for name, byte_count, value in [
        ("SET_3BYTE", 3, 0x332211),
        ("SET_8BYTE", 8, 0x8877665544332211),
    ]:
        mode = getattr(vg.DmaMode, name)
        pattern = value.to_bytes(byte_count, "little")
        for output_width in (width, width + 1, width + 2):
            output_height = height - 1

            def fill(
                mode=mode,
                pattern=pattern,
                value=value,
                output_width=output_width,
                output_height=output_height,
            ):
                actual = array(
                    transfer(
                        buffer,
                        mode,
                        value,
                        output_width=output_width,
                        output_height=output_height,
                    )
                )
                before_after = difference(array(buffer), source)
                row_pattern = np.asarray(
                    [[pattern[x % len(pattern)] for x in range(output_width)]]
                    * output_height,
                    dtype=np.uint8,
                )
                flat_pattern = np.asarray(
                    [
                        [
                            pattern[(y * output_width + x) % len(pattern)]
                            for x in range(output_width)
                        ]
                        for y in range(output_height)
                    ],
                    dtype=np.uint8,
                )
                stride = ((output_width + 15) // 16) * 16
                pitch_pattern = np.asarray(
                    [
                        [
                            pattern[(y * stride + x) % len(pattern)]
                            for x in range(output_width)
                        ]
                        for y in range(output_height)
                    ],
                    dtype=np.uint8,
                )
                candidates = crop_candidates(
                    actual,
                    dict(
                        row_restart=row_pattern,
                        continuous_visible=flat_pattern,
                        continuous_aligned_pitch=pitch_pattern,
                    ),
                )
                dimensions_match = list(actual.shape) == [output_height, output_width]
                byte_order_matches = (
                    dimensions_match
                    and difference(actual[:1], row_pattern[:1])["different"] == 0
                )
                return dict(
                    status=(
                        "unverified"
                        if before_after["different"] == 0 and byte_order_matches
                        else "failed"
                    ),
                    shape=list(actual.shape),
                    expected_shape=[output_height, output_width],
                    pattern=list(pattern),
                    candidates=candidates,
                    first_rows=actual[:3, : min(output_width, 24)].tolist(),
                    source_comparison=before_after,
                    dimensions_match=dimensions_match,
                    little_endian_first_row=byte_order_matches,
                    reason="Little-endian byte order is specified; distinguish row restart from continuous pattern phase at odd widths",
                )

            report.run(
                "dma_%s_nonzero_width%d_%dx%d" % (name, output_width, width, height),
                fill,
            )
    check_dma_row_contract(report, buffer, source, width, height)


def dma_row_pattern(width, height, value, byte_count):
    """Proposed fill rule: restart the little-endian pattern at every row.
    待确立的填充规则：每行从小端字节模式的首字节重新开始。
    """
    pattern = value.to_bytes(byte_count, "little")
    row = [pattern[x % byte_count] for x in range(width)]
    return [list(row) for _ in range(height)]


def check_dma_row_contract(report, buffer, source, width, height):
    """Keep strict whole-image row checks separate from phase observations.
    将逐行全图严格检查与相位观测分开保存。
    """
    if "dma_row_pattern" in report.contract_groups:
        return
    report.contract_groups.add("dma_row_pattern")
    transfer = report.processor.dma_transfer
    dimensions = [
        (1, 1),
        (3, 3),
        (5, 7),
        (9, 5),
        (17, 33),
        (width if width & 1 else width - 1, height if height & 1 else height - 1),
    ]
    for mode_name, byte_count, value in [
        ("SET_3BYTE", 3, 0x332211),
        ("SET_8BYTE", 8, 0x8877665544332211),
    ]:
        mode = getattr(vg.DmaMode, mode_name)
        for output_width, output_height in dimensions:
            expected = np.asarray(
                dma_row_pattern(output_width, output_height, value, byte_count),
                dtype=np.uint8,
            )

            def fill_contract(
                mode=mode,
                value=value,
                output_width=output_width,
                output_height=output_height,
            ):
                result = transfer(
                    buffer,
                    mode,
                    value,
                    output_width=output_width,
                    output_height=output_height,
                )
                if difference(array(buffer), source)["different"] != 0:
                    raise AssertionError("DMA fill changed its source image")
                return result

            report.exact(
                "dma_row_pattern_contract_%s_%dx%d"
                % (mode_name, output_width, output_height),
                fill_contract,
                expected,
                expected_dtype="|u1",
                reference=dict(
                    contract="proposed_little_endian_row_restart_v1",
                    rule="Each row starts at byte zero of value.to_bytes(pattern_size,'little'); every output pixel is compared",
                    fill_value_hex=hex(value),
                    pattern_little_endian=list(value.to_bytes(byte_count, "little")),
                    output_shape=[output_height, output_width],
                    acceptance="This explicit rule is tested independently; earlier DMA phase observations remain unverified",
                ),
            )


def contract_reference_self_test():
    """Test the references without importing or calling a hardware library.
    不导入或调用硬件库，检查参考规则能否拒绝错误实现。
    """
    checks = 0
    for signed_input, signed_output in [(True, True), (True, False), (False, False)]:
        values = list(range(-32768, 32768) if signed_input else range(65536))
        low, high = (-1024, 1024) if signed_input else (0, 32768)
        minimum, maximum = (-110, 111) if signed_output else (3, 251)
        result = raw_low_byte_threshold(
            values, low, high, minimum, maximum, signed_output
        )
        output_min, output_max = (-128, 127) if signed_output else (0, 255)
        saturated = [
            (
                minimum
                if value <= low
                else (
                    maximum if value > high else max(output_min, min(output_max, value))
                )
            )
            for value in values
        ]
        if result == saturated:
            raise AssertionError("The raw-low-byte reference accepted saturation")
        if len(result) != 65536 or len(set(values)) != 65536:
            raise AssertionError("The word domain does not cover all 65536 inputs")
        boundaries = [low, low + 1, high, high + 1]
        boundary_result = raw_low_byte_threshold(
            boundaries, low, high, minimum, maximum, signed_output
        )
        middle = []
        for value in (low + 1, high):
            byte = value & 255
            middle.append(byte - 256 if signed_output and byte >= 128 else byte)
        if boundary_result != [minimum, middle[0], middle[1], maximum]:
            raise AssertionError("Threshold boundaries do not follow the raw word")
        checks += 3
    for byte_count, value in [(3, 0x332211), (8, 0x8877665544332211)]:
        width, height = 5, 3
        reference = dma_row_pattern(width, height, value, byte_count)
        literal_row = (
            [0x11, 0x22, 0x33, 0x11, 0x22]
            if byte_count == 3
            else [0x11, 0x22, 0x33, 0x44, 0x55]
        )
        if reference != [list(literal_row) for _ in range(height)]:
            raise AssertionError(
                "The row reference disagrees with the literal byte fixture"
            )
        pattern = value.to_bytes(byte_count, "little")
        continuous = [
            [pattern[(y * width + x) % byte_count] for x in range(width)]
            for y in range(height)
        ]
        reversed_order = dma_row_pattern(
            width, height, int.from_bytes(pattern, "big"), byte_count
        )
        if reference == continuous:
            raise AssertionError("The row reference accepted cross-row pattern phase")
        if reference == reversed_order:
            raise AssertionError("The row reference accepted big-endian byte order")
        if reference[0] != reference[1] or reference[1] != reference[2]:
            raise AssertionError("The row reference did not restart its byte pattern")
        checks += 4
    cast_fixtures = [
        (3, "U8_TO_U8", 1, 2, 3, 5),
        (3, "U8_TO_U8", 3, 2, -3, 2),
        (-64, "S16_TO_S8", 1, 128, 0, 0),
        (-65, "S16_TO_S8", 1, 128, 0, -1),
        (-2, "S16_TO_U8_BIAS", 1, 4, 127, 127),
        (-3, "S16_TO_U8_ABS", 1, 4, 0, 1),
        (127, "S16_TO_S8", 1, 1, -128, -1),
        (127, "S16_TO_U8_BIAS", 1, 1, -128, 0),
        (65535, "U16_TO_U8", 255, 65535, -128, 127),
        (1, "U16_TO_U8", 255, 1, -128, 127),
        (2, "U16_TO_U8", 255, 1, -128, 255),
        (-32768, "S16_TO_S8", 1, 65535, 0, 0),
    ] + [
        (
            65535 if name.startswith("U") else -32768,
            name,
            0,
            1,
            127,
            127,
        )
        for name in ("S16_TO_S8", "S16_TO_U8_ABS", "S16_TO_U8_BIAS", "U16_TO_U8")
    ]
    for value, mode, numerator, denominator, bias, expected in cast_fixtures:
        actual = cast_reference(value, mode, numerator, denominator, bias)
        if actual != expected:
            raise AssertionError(
                "Cast reference disagrees with its literal numeric fixture"
            )
        checks += 1
    q24_result = cast_reference(-32768, "S16_TO_S8", 1, 65535, 0)
    rational_result = (-32768 * 2 + 65535) // (2 * 65535)
    if (q24_result, rational_result) != (0, -1):
        raise AssertionError(
            "The cast reference does not distinguish Q24 from an exact rational ratio"
        )
    checks += 1
    signed_result = cast_reference(127, "S16_TO_S8", 1, 1, -128)
    unsigned_result = cast_reference(127, "S16_TO_U8_BIAS", 1, 1, -128)
    if (
        signed_result.to_bytes(1, "little", signed=True) != b"\xff"
        or unsigned_result.to_bytes(1, "little") != b"\x00"
    ):
        raise AssertionError(
            "Signed and unsigned cast references lost their distinct byte interpretation"
        )
    checks += 1
    checks += arithmetic_reference_self_test()
    checks += equalize_degenerate_reference_self_test()
    print(
        json.dumps(
            dict(
                status="passed",
                scope="reference_self_test_only",
                hardware_called=False,
                checks=checks,
                rejected_alternatives=[
                    "saturating_word_ORI",
                    "continuous_cross_row_fill_phase",
                    "big_endian_fill_order",
                    "exact_rational_instead_of_Q24",
                    "unsigned_saturation_for_signed_cast",
                    "bias_before_scaling",
                    "weighted_add_saturation",
                    "separately_truncated_weighted_products",
                    "equalization_ties_even",
                    "single_bin_equalization_identity",
                    "single_bin_equalization_step255",
                ],
            )
        )
    )


def sad_reference(first, second, block):
    rows = []
    a, b = first.tolist(), second.tolist()
    for top in range(0, first.shape[0] - block + 1, block):
        row = []
        for left in range(0, first.shape[1] - block + 1, block):
            row.append(
                sum(
                    abs(a[y][x] - b[y][x])
                    for y in range(top, top + block)
                    for x in range(left, left + block)
                )
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.uint16)


def check_sad(report, width, height):
    first = np.asarray(
        [[(x * 7 + y * 11) % 256 for x in range(width)] for y in range(height)],
        dtype=np.uint8,
    )
    second = np.asarray(
        [[(x * 19 + y * 3) % 256 for x in range(width)] for y in range(height)],
        dtype=np.uint8,
    )
    a, b = image(first), image(second)
    zero, maximum = image(np.zeros((height, width), dtype=np.uint8)), image(
        np.full((height, width), 255, dtype=np.uint8)
    )
    for block, name in [(4, "MB_4X4"), (8, "MB_8X8"), (16, "MB_16X16")]:
        if width % block or height % block:
            if hasattr(report.processor, "sad_config"):
                report.rejected(
                    "sad_partial_blocks_rejected_%s_%dx%d" % (name, width, height),
                    lambda name=name: report.processor.sad(
                        a, b, getattr(vg.SadMode, name), 0
                    ),
                )
            else:
                report.missing(
                    "sad_partial_blocks_%s_%dx%d" % (name, width, height),
                    "Define partial-block padding/cropping or reject non-divisible dimensions before allocating hardware output",
                )
            continue
        mode = getattr(vg.SadMode, name)
        expected = sad_reference(first, second, block)
        report.exact(
            "sad_values_%s_%dx%d" % (name, width, height),
            lambda mode=mode: report.processor.sad(a, b, mode, 1024, 17, 231)[0],
            expected,
            expected_dtype="<u2",
        )
        threshold = np.asarray(
            [
                [231 if value > 1024 else 17 for value in row]
                for row in expected.tolist()
            ],
            dtype=np.uint8,
        )
        report.exact(
            "sad_threshold_%s_%dx%d" % (name, width, height),
            lambda mode=mode: report.processor.sad(a, b, mode, 1024, 17, 231)[1],
            threshold,
        )
        peak = block * block * 255
        full = np.full((height // block, width // block), peak, dtype=np.uint16)
        for limit in (peak - 1, peak, min(65535, peak + 1)):
            report.exact(
                "sad_boundary_%s_%d_%dx%d" % (name, limit, width, height),
                lambda mode=mode, limit=limit: report.processor.sad(
                    zero, maximum, mode, limit, 17, 231
                )[1],
                np.full(full.shape, 231 if peak > limit else 17, dtype=np.uint8),
            )
        report.exact(
            "sad_maximum_%s_%dx%d" % (name, width, height),
            lambda mode=mode: report.processor.sad(zero, maximum, mode, 0)[0],
            full,
            expected_dtype="<u2",
        )
        method = getattr(report.processor, "sad_config", None)
        if method is not None:
            for output_name in ("SAD", "BOTH", "THRESH"):
                for bits_name in ("BITS8", "BITS16"):

                    def selected(
                        output_name=output_name, bits_name=bits_name, mode=mode
                    ):
                        result, threshold_result = method(
                            a,
                            b,
                            mode,
                            1024,
                            17,
                            231,
                            getattr(vg.SadOutMode, output_name),
                            getattr(vg.SadOutBits, bits_name),
                        )
                        comparisons = {}
                        valid = True
                        if output_name == "THRESH":
                            valid = valid and not result.is_valid()
                        else:
                            wanted = (
                                expected
                                if bits_name == "BITS16"
                                else np.asarray(
                                    [
                                        [value >> 8 for value in row]
                                        for row in expected.tolist()
                                    ],
                                    dtype=np.uint8,
                                )
                            )
                            actual = array(result)
                            comparisons["sad"] = difference(actual, wanted)
                            valid = (
                                valid
                                and comparisons["sad"]["different"] == 0
                                and actual.dtype.str == wanted.dtype.str
                            )
                        if output_name == "SAD":
                            valid = valid and not threshold_result.is_valid()
                        else:
                            comparisons["threshold"] = difference(
                                array(threshold_result), threshold
                            )
                            valid = valid and comparisons["threshold"]["different"] == 0
                        return dict(
                            status="passed" if valid else "failed",
                            comparisons=comparisons,
                            reference="8-bit SAD retains bits 8..15 of the full block sum; thresholding uses the full sum",
                        )

                    report.run(
                        "sad_config_%s_%s_%s_%dx%d"
                        % (name, output_name, bits_name, width, height),
                        selected,
                    )


NEIGHBORS = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]


def lbp_reference(source, absolute, threshold):
    """Compare neighbors independently, clockwise from top-left at bit seven.
    独立比较邻域，左上角为第七位，沿顺时针排列。
    """
    height, width = source.shape
    center = source[1:-1, 1:-1].astype(np.int16)
    code = np.zeros(center.shape, dtype=np.uint8)
    for index, (dx, dy) in enumerate(NEIGHBORS):
        neighbor = source[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx].astype(
            np.int16
        )
        delta = cv2.absdiff(neighbor, center) if absolute else neighbor - center
        selected = cv2.compare(delta, threshold, cv2.CMP_GE)
        code = cv2.bitwise_or(
            code,
            cv2.bitwise_and(
                selected, np.full(code.shape, 1 << (7 - index), dtype=np.uint8)
            ),
        )
    result = np.zeros(source.shape, dtype=np.uint8)
    result[1:-1, 1:-1] = code
    return result


def check_lbp(report, width, height):
    for threshold in (0, 1, 8, -8):
        flat = np.full((height, width), 80, dtype=np.uint8)
        for absolute in (False, True) if threshold >= 0 else (False,):
            expected = np.full(
                (height, width), 255 if 0 >= threshold else 0, dtype=np.uint8
            )
            report.exact(
                "lbp_flat_%s_%d_%dx%d"
                % ("abs" if absolute else "normal", threshold, width, height),
                lambda absolute=absolute, threshold=threshold: report.processor.lbp(
                    image(flat), absolute, threshold
                ),
                expected,
                border=2,
            )

    def one_hot():
        source = np.full((height, width), 80, dtype=np.uint8)
        centers = [(12 + index * 12, 16) for index in range(8)]
        for (cx, cy), (dx, dy) in zip(centers, NEIGHBORS):
            source[cy + dy, cx + dx] = 96
        actual = array(report.processor.lbp(image(source), False, 8))
        samples = [int(actual[cy, cx]) for cx, cy in centers]
        expected = [128, 64, 32, 16, 8, 4, 2, 1]
        return dict(
            status="passed" if samples == expected else "failed",
            reference="Clockwise neighbor order starting at the top-left, bits seven through zero",
            offsets=[list(offset) for offset in NEIGHBORS],
            bit_values=samples,
            expected=expected,
        )

    report.run("lbp_bit_order_%dx%d" % (width, height), one_hot)

    def comparison_boundary():
        results = []
        for absolute in (False, True):
            for delta in (7, 8, 9, -7, -8, -9) if absolute else (7, 8, 9):
                source = np.full((height, width), 80, dtype=np.uint8)
                source[20:23, 20:23] = 80 + delta
                source[21, 21] = 80
                actual = array(report.processor.lbp(image(source), absolute, 8))
                expected = 255 if (abs(delta) if absolute else delta) >= 8 else 0
                results.append(
                    dict(
                        abs_mode=absolute,
                        delta=delta,
                        actual=int(actual[21, 21]),
                        expected=expected,
                    )
                )
        return dict(
            status=(
                "passed"
                if all(row["actual"] == row["expected"] for row in results)
                else "failed"
            ),
            samples=results,
        )

    report.run("lbp_comparison_boundary_%dx%d" % (width, height), comparison_boundary)
    generator = random.Random(581)
    source = np.frombuffer(
        bytes(generator.randrange(256) for _ in range(width * height)), dtype=np.uint8
    ).reshape(height, width)
    buffer = image(source)
    for absolute, thresholds in [
        (False, (-128, -8, 0, 1, 8, 127)),
        (True, (0, 1, 8, 127, 128, 255)),
    ]:
        for threshold in thresholds:
            report.exact(
                "lbp_holdout_%s_%d_%dx%d"
                % ("abs" if absolute else "normal", threshold, width, height),
                lambda absolute=absolute, threshold=threshold: report.processor.lbp(
                    buffer, absolute, threshold
                ),
                lbp_reference(source, absolute, threshold),
                border=2,
            )


GRADIENT_MASK = [
    [-1, -2, 0, 2, 1],
    [-4, -8, 0, 8, 4],
    [-6, -12, 0, 12, 6],
    [-4, -8, 0, 8, 4],
    [-1, -2, 0, 2, 1],
]


def convolution_reference(source, mask, norm, low, high, dtype):
    """Use OpenCV convolution and explicit signed arithmetic shift.
    使用 OpenCV 卷积及显式有符号算术右移。
    """
    filtered = cv2.filter2D(source, cv2.CV_32F, mask)
    return np.asarray(
        [
            [max(low, min(high, int(value) >> norm)) for value in row]
            for row in filtered.tolist()
        ],
        dtype=dtype,
    )


def check_sobel_extended(report, source):
    method = getattr(report.processor, "sobel_ex", None)
    if method is None:
        report.missing(
            "sobel_ex_controls",
            "Expose explicit mask, normalization and four output types",
        )
        return
    height, width = source.shape
    buffer = image(source)
    mask = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 0, 1, 0],
            [0, -2, 0, 2, 0],
            [0, -1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    values = mask.astype(np.int8).ravel().tolist()
    report.rejected(
        "sobel_ex_shift16_rejected_%dx%d" % (width, height),
        lambda: method(buffer, values, 16, vg.SobelOutCtrl.BOTH, vg.ImageTypeIVE.S16C1),
    )
    for type_name, dtype, low, high in [
        ("U8C1", np.uint8, 0, 255),
        ("S8C1", np.int8, -128, 127),
        ("U16C1", np.uint16, 0, 65535),
        ("S16C1", np.int16, -32768, 32767),
    ]:
        output_format = getattr(vg.ImageTypeIVE, type_name, None)
        if output_format is None:
            report.missing("sobel_ex_" + type_name, "Expose ImageTypeIVE." + type_name)
            continue
        expected_x = convolution_reference(source, mask, 0, low, high, dtype)
        expected_y = convolution_reference(source, mask.T, 0, low, high, dtype)
        for name, index, expected in [
            ("BOTH_HOR", 0, expected_x),
            ("BOTH_VER", 1, expected_y),
            ("HOR", 0, expected_x),
            ("VER", 1, expected_y),
        ]:
            mode = getattr(vg.SobelOutCtrl, "BOTH" if name.startswith("BOTH") else name)
            report.exact(
                "sobel_ex_%s_%s_%dx%d" % (type_name, name, width, height),
                lambda method=method, mode=mode, index=index, output_format=output_format: method(
                    buffer, values, 0, mode, output_format
                )[
                    index
                ],
                expected,
                border=2,
                expected_dtype=expected.dtype.str,
            )


def check_magnitude_holdout(report, source):
    height, width = source.shape
    horizontal = cv2.filter2D(source, cv2.CV_32F, GRADIENT_MASK)
    vertical = cv2.filter2D(source, cv2.CV_32F, GRADIENT_MASK.T)
    output_dtype = (
        np.uint16 if hasattr(report.processor, "mag_and_ang_ex") else np.int16
    )
    expected = np.asarray(
        [
            [abs(int(x)) + abs(int(y)) for x, y in zip(xrow, yrow)]
            for xrow, yrow in zip(horizontal.tolist(), vertical.tolist())
        ],
        dtype=output_dtype,
    )
    buffer = image(source)
    report.exact(
        "magnitude_L1_holdout_%dx%d" % (width, height),
        lambda: report.processor.mag_and_ang(buffer, 0, True),
        expected,
        border=2,
        expected_dtype=expected.dtype.str,
    )
    method = getattr(report.processor, "mag_and_ang_ex", None)
    if method is not None:
        maximum_input = image(np.full((height, width), 255, dtype=np.uint8))
        for coefficient in (127, -128):
            mask = [0] * 25
            mask[12] = coefficient
            report.exact(
                "magnitude_ex_unsigned_limit_%d_%dx%d" % (coefficient, width, height),
                lambda mask=mask: method(maximum_input, 0, mask)[0],
                np.full((height, width), 2 * abs(coefficient) * 255, dtype=np.uint16),
                border=2,
                expected_dtype="<u2",
            )
            report.exact(
                "angle_ex_canonical_%d_%dx%d" % (coefficient, width, height),
                lambda mask=mask: method(maximum_input, 0, mask)[1],
                np.full((height, width), 1 if coefficient > 0 else 5, dtype=np.uint8),
                border=2,
            )

    def angular_bins():
        actual = array(report.processor.mag_and_ang(buffer, 0, False))
        nearest = np.asarray(
            [
                [
                    (
                        int(
                            math.floor(
                                (math.degrees(math.atan2(y, x)) % 360 + 22.5) / 45
                            )
                        )
                        % 8
                        if x or y
                        else 0
                    )
                    for x, y in zip(xrow, yrow)
                ]
                for xrow, yrow in zip(horizontal.tolist(), vertical.tolist())
            ],
            dtype=np.uint8,
        )
        return dict(
            status="unverified",
            nearest_eight_bin_comparison=difference(actual, nearest, 2),
            observed_values=sorted(set(actual.ravel().tolist())),
            reason="Canonical bin centers are established; verify the boundaries on independent textured gradients before specifying angular quantization",
        )

    report.run("angle_eight_bin_boundary_holdout_%dx%d" % (width, height), angular_bins)


def check_normalized_controls(report, width, height):
    method = getattr(report.processor, "norm_grad_config", None)
    enum = getattr(vg, "NormGradOutCtrl", None)
    if method is None or enum is None:
        report.missing(
            "norm_grad_config",
            "Expose normalized gradient modes and mask/shift controls",
        )
        return
    source = np.asarray(
        [
            [128 if ((x // 5 + y // 7) & 1) else 0 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    buffer = image(source)
    mask = GRADIENT_MASK.astype(np.int8).ravel().tolist()
    report.rejected(
        "normgrad_shift16_rejected_%dx%d" % (width, height),
        lambda: method(buffer, enum.HOR_AND_VER, 16, mask),
    )
    for norm in (0, 3, 7):
        expected_x = convolution_reference(
            source, GRADIENT_MASK, norm, -128, 127, np.int8
        )
        expected_y = convolution_reference(
            source, GRADIENT_MASK.T, norm, -128, 127, np.int8
        )
        for name, index, expected in [
            ("HOR_AND_VER", 0, expected_x),
            ("HOR_AND_VER", 1, expected_y),
            ("HOR", 0, expected_x),
            ("VER", 1, expected_y),
        ]:
            mode = getattr(enum, name, None)
            if mode is None:
                report.missing(
                    "norm_grad_config_" + name, "Expose NormGradOutCtrl." + name
                )
                continue
            report.exact(
                "normgrad_config_%s_%d_shift%d_%dx%d"
                % (name, index, norm, width, height),
                lambda mode=mode, norm=norm, index=index: method(
                    buffer, mode, norm, mask
                )[index],
                expected,
                border=2,
                expected_dtype="|i1",
            )
        combined_mode = getattr(enum, "COMBINE", None)
        if combined_mode is not None:
            packed = np.asarray(
                [
                    [((int(y) & 255) << 8) | (int(x) & 255) for x, y in zip(xrow, yrow)]
                    for xrow, yrow in zip(expected_x.tolist(), expected_y.tolist())
                ],
                dtype=np.uint16,
            )
            report.exact(
                "normgrad_config_COMBINE_shift%d_%dx%d" % (norm, width, height),
                lambda norm=norm: method(buffer, combined_mode, norm, mask)[2],
                packed,
                border=2,
                expected_dtype="<u2",
            )


def check_gradient(report, width, height):
    generator = random.Random(634)
    sobel_source = np.frombuffer(
        bytes(generator.randrange(256) for _ in range(width * height)), dtype=np.uint8
    ).reshape(height, width)
    sobel_buffer = image(sobel_source)
    reference_x = cv2.Sobel(sobel_source, cv2.CV_16S, 1, 0, ksize=3)
    reference_y = cv2.Sobel(sobel_source, cv2.CV_16S, 0, 1, ksize=3)
    for name, index, expected in [
        ("BOTH_HOR", 0, reference_x),
        ("BOTH_VER", 1, reference_y),
        ("HOR", 0, reference_x),
        ("VER", 0, reference_y),
    ]:
        mode = getattr(vg.SobelOutCtrl, "BOTH" if name.startswith("BOTH") else name)
        report.exact(
            "sobel_%s_s16_%dx%d" % (name, width, height),
            lambda mode=mode, index=index: report.processor.sobel(
                sobel_buffer, mode, vg.ImageTypeIVE.S16C1
            )[index],
            expected,
            border=2,
            expected_dtype="<i2",
        )
    for name, expected in [("HOR", reference_x), ("VER", reference_y)]:
        mode = getattr(vg.SobelOutCtrl, name)
        unsigned = np.asarray(
            [[max(0, min(255, value)) for value in row] for row in expected.tolist()],
            dtype=np.uint8,
        )
        report.exact(
            "sobel_%s_u8_saturation_%dx%d" % (name, width, height),
            lambda mode=mode: report.processor.sobel(
                sobel_buffer, mode, vg.ImageTypeIVE.U8C1
            )[0],
            unsigned,
            border=2,
        )
    check_sobel_extended(report, sobel_source)
    check_magnitude_holdout(report, sobel_source)
    check_normalized_controls(report, width, height)
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1)]:
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
        roi = (
            slice(height // 2 - 8, height // 2 + 8),
            slice(width // 2 - 8, width // 2 + 8),
        )
        gx, gy = 128 * dx, 128 * dy

        def normalized():
            horizontal, vertical = report.processor.norm_grad(buffer)
            actual_x, actual_y = array(horizontal), array(vertical)
            expected_x, expected_y = np.full((16, 16), dx, dtype=np.int16), np.full(
                (16, 16), dy, dtype=np.int16
            )
            comparisons = [
                difference(actual_x[roi], expected_x),
                difference(actual_y[roi], expected_y),
            ]
            return dict(
                status=(
                    "passed"
                    if all(item["different"] == 0 for item in comparisons)
                    else "failed"
                ),
                comparisons=comparisons,
                expected=[dx, dy],
                dtypes=[actual_x.dtype.str, actual_y.dtype.str],
            )

        report.run("normgrad_linear_%d_%d_%dx%d" % (dx, dy, width, height), normalized)

        def magnitude():
            actual = array(report.processor.mag_and_ang(buffer, 0, True))
            values = sorted(set(int(value) for value in actual[roi].ravel().tolist()))
            expected = abs(gx) + abs(gy)
            return dict(
                status="passed" if values == [expected] else "failed",
                values=values,
                gradient=[gx, gy],
                euclidean=math.hypot(gx, gy),
                manhattan=abs(gx) + abs(gy),
                maximum_component=max(abs(gx), abs(gy)),
                reference="L1 norm of independently specified horizontal and vertical integer gradients",
                dtype=actual.dtype.str,
            )

        report.run("magnitude_linear_%d_%d_%dx%d" % (dx, dy, width, height), magnitude)

        def angle():
            actual = array(report.processor.mag_and_ang(buffer, 0, False))
            angle_degrees = math.degrees(math.atan2(gy, gx)) % 360 if gx or gy else None
            observed = sorted(set(int(value) for value in actual[roi].ravel().tolist()))
            expected = int(angle_degrees / 45) if angle_degrees is not None else None
            return dict(
                status=(
                    "unverified"
                    if expected is None
                    else ("passed" if observed == [expected] else "failed")
                ),
                gradient=[gx, gy],
                values=observed,
                expected_bin=expected,
                degrees=angle_degrees,
                reference="Canonical centers of eight directional bins; zero-gradient and bin boundaries are verified separately",
            )

        report.run("angle_linear_%d_%d_%dx%d" % (dx, dy, width, height), angle)

    def saturation():
        source = np.zeros((height, width), dtype=np.uint8)
        source[:, width // 2 :] = 255
        buffer = image(source)
        actual_x, actual_y = (
            array(value) for value in report.processor.norm_grad(buffer)
        )
        filtered_x = cv2.filter2D(source, cv2.CV_32F, GRADIENT_MASK)
        filtered_y = cv2.filter2D(source, cv2.CV_32F, GRADIENT_MASK.T)
        results = {}
        for label, actual, filtered in [
            ("horizontal", actual_x, filtered_x),
            ("vertical", actual_y, filtered_y),
        ]:
            candidates = {}
            for rule in ("floor", "nearest_positive_tie", "truncate"):
                rows = []
                for row in filtered.tolist():
                    values = []
                    for value in row:
                        scaled = value / 128
                        value = (
                            math.floor(scaled)
                            if rule == "floor"
                            else (
                                math.floor(scaled + 0.5)
                                if rule == "nearest_positive_tie"
                                else math.trunc(scaled)
                            )
                        )
                        values.append(max(-128, min(127, value)))
                    rows.append(values)
                candidates[rule] = np.asarray(rows, dtype=np.int16)
            results[label] = dict(
                minimum=int(actual.min()),
                maximum=int(actual.max()),
                candidates=crop_candidates(actual, candidates, 2),
            )
        return dict(
            status="unverified",
            results=results,
            reason="Do not accept a one-unit tolerance as a substitute for a rounding specification",
        )

    report.run("normgrad_saturation_rounding_%dx%d" % (width, height), saturation)

    def magnitude_threshold():
        source = np.asarray(
            [
                [max(0, min(255, 128 + x - width // 2)) for x in range(width)]
                for _ in range(height)
            ],
            dtype=np.uint8,
        )
        method = getattr(report.processor, "mag_and_ang_ex", None)
        if method is None:
            results = []
            for threshold in (0, 127, 128, 129, 65535):
                actual = array(
                    report.processor.mag_and_ang(image(source), threshold, True)
                )
                results.append(
                    dict(
                        threshold=threshold,
                        magnitude=int(actual[height // 2, width // 2]),
                    )
                )
            return dict(
                status="unverified",
                samples=results,
                reason="The runtime exposes no verified magnitude-threshold contract",
            )

        # A vertical ramp has magnitude 128 and a nonzero canonical angle. / 竖向斜坡的幅值为 128，方向编码非零。
        vertical = np.asarray(
            [
                [max(0, min(255, 128 + y - height // 2)) for _ in range(width)]
                for y in range(height)
            ],
            dtype=np.uint8,
        )
        vertical_buffer = image(vertical)
        roi = (
            slice(height // 2 - 8, height // 2 + 8),
            slice(width // 2 - 8, width // 2 + 8),
        )
        cases, passed = [], True
        for threshold in (0, 127, 128, 129, 65535):
            expected_magnitude = np.full(
                (16, 16), 128 if threshold <= 128 else 0, dtype=np.uint16
            )
            expected_angle = np.full(
                (16, 16), 2 if threshold <= 128 else 0, dtype=np.uint8
            )
            comparisons = []
            for _ in range(report.loops):
                magnitude, angle = method(vertical_buffer, threshold, [])
                actual_magnitude, actual_angle = array(magnitude), array(angle)
                comparisons.append(
                    dict(
                        magnitude=difference(actual_magnitude[roi], expected_magnitude),
                        angle=difference(actual_angle[roi], expected_angle),
                    )
                )
            valid = all(
                item["magnitude"]["different"] == 0 and item["angle"]["different"] == 0
                for item in comparisons
            )
            valid = (
                valid
                and actual_magnitude.dtype.str == "<u2"
                and actual_angle.dtype.str == "|u1"
            )
            passed = passed and valid
            cases.append(
                dict(
                    input="vertical_ramp",
                    threshold=threshold,
                    expected_magnitude=128 if threshold <= 128 else 0,
                    expected_angle=2 if threshold <= 128 else 0,
                    comparisons=comparisons,
                    passed=valid,
                )
            )

        texture = np.asarray(
            [
                [(17 * x + 31 * y + (x * y) % 19) % 256 for x in range(width)]
                for y in range(height)
            ],
            dtype=np.uint8,
        )
        texture_buffer = image(texture)
        baseline_magnitude, baseline_angle = method(texture_buffer, 0, [])
        baseline_magnitude, baseline_angle = array(baseline_magnitude), array(
            baseline_angle
        )
        magnitude_rows, angle_rows = (
            baseline_magnitude.tolist(),
            baseline_angle.tolist(),
        )
        values = sorted(set(baseline_magnitude.ravel().tolist()))
        middle = int(values[len(values) // 2])
        thresholds = list(
            dict.fromkeys(
                (0, 127, 128, 129, middle, min(65535, int(values[-1]) + 1), 65535)
            )
        )
        mixed_case = False
        for threshold in thresholds:
            expected_magnitude = np.asarray(
                [
                    [value if value >= threshold else 0 for value in row]
                    for row in magnitude_rows
                ],
                dtype=np.uint16,
            )
            expected_angle = np.asarray(
                [
                    [
                        angle if magnitude >= threshold else 0
                        for magnitude, angle in zip(mrow, arow)
                    ]
                    for mrow, arow in zip(magnitude_rows, angle_rows)
                ],
                dtype=np.uint8,
            )
            suppressed = sum(
                value < threshold for row in magnitude_rows for value in row
            )
            kept = width * height - suppressed
            mixed_case = mixed_case or bool(suppressed and kept)
            comparisons = []
            for _ in range(report.loops):
                actual_magnitude, actual_angle = method(texture_buffer, threshold, [])
                comparisons.append(
                    dict(
                        magnitude=difference(
                            array(actual_magnitude), expected_magnitude
                        ),
                        angle=difference(array(actual_angle), expected_angle),
                    )
                )
            valid = all(
                item["magnitude"]["different"] == 0 and item["angle"]["different"] == 0
                for item in comparisons
            )
            passed = passed and valid
            cases.append(
                dict(
                    input="texture",
                    threshold=threshold,
                    suppressed_pixels=suppressed,
                    retained_pixels=kept,
                    comparisons=comparisons,
                    passed=valid,
                )
            )
        return dict(
            status=(
                "failed" if not passed else ("passed" if mixed_case else "unverified")
            ),
            cases=cases,
            mixed_suppression_case=mixed_case,
            reference="Public threshold contract: retain magnitude >= threshold; zero both magnitude and angle below threshold",
            texture_reference="Compare postprocessing with the threshold-zero hardware baseline over the full image; this does not establish the original gradient or angle-bin formula",
        )

    report.run("magnitude_threshold_%dx%d" % (width, height), magnitude_threshold)


def gradient_quantization_reference(gx, gy):
    """Apply the observed signed integer angle intervals without floating point.
    不使用浮点数，按已观测的有符号整数区间计算方向。
    """
    if gx == 0 and gy == 0:
        return 0
    offset = 0
    if gx < 0:
        gx, gy, offset = -gx, -gy, 4
    scaled_y = gy * 8192
    if scaled_y >= gx * 19777:
        angle = 2
    elif scaled_y >= gx * 3393:
        angle = 1
    elif scaled_y >= -gx * 3393:
        angle = 0
    elif scaled_y >= -gx * 19777:
        angle = 7
    else:
        angle = 6
    return (angle + offset) % 8


def gradient_quantization_large_points():
    """Generate signed cutoff witnesses and deterministic held-out samples.
    生成有符号阈值见证点及确定性留出样本。
    """
    constants = set()

    def rational(numerator, denominator):
        divisor = math.gcd(numerator, denominator)
        return numerator // divisor, denominator // divisor

    for bits in (8, 12, 14, 16):
        denominator = 1 << bits
        lower = math.isqrt(2 * denominator * denominator) - denominator
        nearest = lower + (
            (2 * (lower + denominator) + 1) ** 2 < 8 * denominator * denominator
        )
        for numerator in (lower, nearest, lower + 1):
            constants.add(rational(numerator, denominator))
    constants.update(
        rational(numerator, denominator)
        for numerator, denominator in (
            (2071, 5000),
            (41, 99),
            (70, 169),
            (169, 408),
            (408, 985),
            (985, 2378),
            (2378, 5741),
            (5741, 13860),
        )
    )
    cutoffs = constants | {
        rational(denominator, 2 * denominator + numerator)
        for numerator, denominator in constants
    }
    denominators = (
        1,
        2,
        3,
        7,
        12,
        16,
        29,
        31,
        32,
        63,
        64,
        65,
        70,
        99,
        127,
        128,
        129,
        169,
        227,
        239,
        255,
        256,
        257,
        268,
        309,
        408,
        437,
        512,
        548,
        577,
        647,
        688,
        915,
        985,
        1024,
        1055,
        1393,
        1789,
        2378,
        3363,
        4096,
        4249,
        4493,
        5000,
        5169,
        5741,
        8119,
        8192,
        10258,
        10847,
        12479,
        13860,
        16384,
        19601,
        27145,
        30000,
        32639,
        32640,
    )
    positive = set()
    for x in denominators:
        centers = {math.isqrt(2 * x * x) - x}
        centers.update(
            numerator * x // denominator for numerator, denominator in cutoffs
        )
        for center in centers:
            for delta in (-2, -1, 0, 1, 2):
                if 0 <= center + delta <= 32640:
                    positive.add((x, center + delta))
    positive.update(
        (169 * scale, 70 * scale) for scale in (1, 2, 3, 7, 31, 127, 191, 193)
    )
    positive.update(((0, 0), (32640, 0), (32640, 32640), (32640, 32639)))
    designed = {
        (sx * a, sy * b)
        for x, y in positive
        for a, b in ((x, y), (y, x))
        for sx in (-1, 1)
        for sy in (-1, 1)
    }
    points = [(x, y, "large_boundary", False) for x, y in sorted(designed)]
    occupied = set(designed)
    generator = random.Random(283141)
    for purpose in ("heldout_boundary", "heldout_random"):
        count = 0
        while count < 512:
            x = generator.randrange(1, 32641)
            if purpose == "heldout_boundary":
                y = math.isqrt(2 * x * x) - x + generator.randrange(-3, 4)
                if generator.randrange(2):
                    x, y = y, x
            else:
                y = generator.randrange(32641)
            x *= -1 if generator.randrange(2) else 1
            y *= -1 if generator.randrange(2) else 1
            if (
                not -32640 <= x <= 32640
                or not -32640 <= y <= 32640
                or (x, y) in occupied
            ):
                continue
            occupied.add((x, y))
            points.append((x, y, purpose, False))
            count += 1
    for index in range(0, len(points), max(1, len(points) // 48)):
        gx, gy = points[index][:2]
        if any(127 <= abs(value) <= 32513 for value in (gx, gy)):
            points.append((gx, gy, "encoding_holdout", True))
    return points


def gradient_quantization_points():
    """Stream the full small domain and bounded large/tie fixtures.
    流式生成完整小梯度域与有界大梯度、等号边界用例。
    """
    for gx in range(-255, 256):
        for gy in range(-255, 256):
            yield gx, gy, "small_full_domain", False
    yield from gradient_quantization_large_points()
    targeted = {(0, 0)}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for cutoff in (3393, 19777):
                for delta in (-1, 0, 1):
                    targeted.add((sx * 8192, sy * cutoff + delta))
            for delta in (-1, 0, 1):
                targeted.add((sx * 16384, sy * 6786 + delta))
            targeted.add((sx * 32640, sy * 32640))
    for sign in (-1, 1):
        for value in (1, 32640):
            targeted.add((sign * value, 0))
            targeted.add((0, sign * value))
    for gx, gy in sorted(targeted):
        yield gx, gy, "exact_tie_neighbours", False
    for gx, gy in sorted(targeted):
        if any(127 <= abs(value) <= 32513 for value in (gx, gy)):
            yield gx, gy, "exact_tie_encoding", True


def gradient_quantization_frame(points, size=320):
    """Build disjoint cross tiles and verify each derivative by scalar convolution.
    构造互不重叠的十字像素块，并用标量卷积核对每个导数。
    """
    if len(points) > (size // 5) ** 2:
        raise ValueError("Gradient fixture exceeds tile capacity")
    data = bytearray(size * size)
    expected = []
    for index, (gx, gy, purpose, alternate) in enumerate(points):
        y, x = index // (size // 5) * 5 + 2, index % (size // 5) * 5 + 2
        if abs(gx) > 32640 or abs(gy) > 32640:
            raise ValueError("Gradient fixture exceeds its validated component range")
        for value, dy, dx in ((gx, 0, 1), (gy, 1, 0)):
            quotient = min(abs(value) // 127, 255)
            remainder = abs(value) - quotient * 127
            if alternate and quotient > 0 and remainder <= 128:
                quotient, remainder = quotient - 1, remainder + 127
            if not 0 <= quotient <= 255 or not 0 <= remainder <= 255:
                raise ValueError("Gradient cannot be encoded as unsigned pixels")
            sign = -1 if value < 0 else 1
            data[(y + sign * dy) * size + x + sign * dx] = quotient
            data[(y + 2 * sign * dy) * size + x + 2 * sign * dx] = remainder
        expected.append((abs(gx) + abs(gy), gradient_quantization_reference(gx, gy)))
    for index, (gx, gy, purpose, alternate) in enumerate(points):
        y, x = index // (size // 5) * 5 + 2, index % (size // 5) * 5 + 2
        for wanted, dy, dx in ((gx, 0, 1), (gy, 1, 0)):
            actual = sum(
                coefficient * data[(y + delta * dy) * size + x + delta * dx]
                for delta, coefficient in ((-2, -1), (-1, -127), (1, 127), (2, 1))
            )
            if actual != wanted:
                raise ValueError("Scalar convolution differs from fixture gradient")
    return data, expected


def gradient_quantization_batch(report, points, index, requested, retained):
    """Compare both sampled outputs exactly and audit every hardware call.
    精确比较两个采样输出，并审计每次硬件调用。
    """
    raw, expected = gradient_quantization_frame(points)
    source_hash = hashlib.sha256(raw).hexdigest()
    source = image(np.frombuffer(raw, dtype=np.uint8).reshape(320, 320))
    mask = [0] * 10 + [-1, -127, 0, 127, 1] + [0] * 10

    def check():
        comparisons, signatures, irqs, timings = [], [], [], []
        preserved, retained_ok, previous_ok = True, True, True
        dtype_matches, shape_matches = True, True
        first_outputs, first_hashes = None, None
        for _ in range(report.loops):
            before, tick = interrupt_count(), time.perf_counter()
            outputs = report.processor.mag_and_ang_ex(source, 0, mask)
            timings.append((time.perf_counter() - tick) * 1000)
            after = interrupt_count()
            irqs.append(
                after - before if before is not None and after is not None else None
            )
            if len(outputs) != 2:
                raise ValueError("Magnitude and angle outputs are both required")
            values = [array(value) for value in outputs]
            shapes = [list(value.shape) for value in values]
            dtypes = [value.dtype.str for value in values]
            shape_matches = shape_matches and shapes == [[320, 320], [320, 320]]
            dtype_matches = dtype_matches and dtypes == ["<u2", "|u1"]
            current_hashes = [
                hashlib.sha256(value.tobytes()).hexdigest() for value in values
            ]
            signatures.append(current_hashes)
            errors, peak, examples = [0, 0], [0, 0], []
            if shapes == [[320, 320], [320, 320]]:
                for point_index, wanted in enumerate(expected):
                    y, x = point_index // 64 * 5 + 2, point_index % 64 * 5 + 2
                    actual = (int(values[0][y, x]), int(values[1][y, x]))
                    for component in (0, 1):
                        delta = abs(actual[component] - wanted[component])
                        errors[component] += delta != 0
                        peak[component] = max(peak[component], delta)
                    if actual != wanted and len(examples) < 8:
                        examples.append(
                            dict(
                                index=point_index,
                                gradient=list(points[point_index][:2]),
                                actual=list(actual),
                                expected=list(wanted),
                            )
                        )
            else:
                errors = [None, None]
            comparisons.append(
                dict(
                    magnitude_different=errors[0],
                    angle_different=errors[1],
                    magnitude_maximum=peak[0],
                    angle_maximum=peak[1],
                    first_mismatches=examples,
                )
            )
            preserved = (
                preserved
                and hashlib.sha256(array(source).tobytes()).hexdigest() == source_hash
                and hashlib.sha256(raw).hexdigest() == source_hash
            )
            retained_ok = retained_ok and all(
                hashlib.sha256(array(value).tobytes()).hexdigest() == checksum
                for value, checksum in zip(outputs, current_hashes)
            )
            if first_outputs is None:
                first_outputs, first_hashes = outputs, current_hashes
            else:
                retained_ok = retained_ok and all(
                    hashlib.sha256(array(value).tobytes()).hexdigest() == checksum
                    for value, checksum in zip(first_outputs, first_hashes)
                )
            if retained:
                previous_ok = previous_ok and all(
                    hashlib.sha256(array(value).tobytes()).hexdigest() == checksum
                    for value, checksum in zip(retained["outputs"], retained["hashes"])
                )
        passed = (
            all(
                item["magnitude_different"] == item["angle_different"] == 0
                for item in comparisons
            )
            and dtype_matches
            and shape_matches
            and preserved
            and retained_ok
            and previous_ok
            and len({tuple(value) for value in signatures}) == 1
        )
        status = "passed" if passed else "failed"
        reason = None
        if any(value is not None and value <= 0 for value in irqs):
            status, reason = "failed", "No hardware activity for a call"
        elif status == "passed" and any(value is None for value in irqs):
            status, reason = "unverified", "Hardware counter unavailable"
        elif status == "passed" and report.loops < 2:
            status, reason = (
                "unverified",
                "Repeat and retained-output checks require at least two loops",
            )
        previous_batch = retained.get("index")
        retained.update(outputs=first_outputs, hashes=first_hashes, index=index)
        return dict(
            status=status,
            reason=reason,
            reference="Independent integer convolution; U16 abs(gx)+abs(gy), U8 signed half-open intervals with denominator8192 and cutoffs3393/19777",
            reference_scope="Constructed tile centers, components within[-32640,32640], norm0 and threshold0; no claim about internal arithmetic",
            requested_shape=list(requested),
            fixture_shape=[320, 320],
            fixture_policy="Fixed bounded frames; generated full domain runs once per report",
            sample_count=len(points),
            sample_counts=dict(Counter(point[2] for point in points)),
            compared_scope="Every constructed tile center; full outputs checked for dtype, shape and stability",
            excluded_border=None,
            scalar_convolution_centers=len(points),
            source_sha256=source_hash,
            comparisons=comparisons,
            dtypes=dtypes,
            expected_dtypes=["<u2", "|u1"],
            shapes=shapes,
            dtype_matches=dtype_matches,
            shape_matches=shape_matches,
            inputs_preserved=preserved,
            outputs_preserved=retained_ok,
            previous_batch=previous_batch,
            previous_batch_outputs_preserved=(
                previous_ok if previous_batch is not None else None
            ),
            signatures=signatures,
            interrupts_per_call=irqs,
            repeat_checked=report.loops >= 2,
            min_ms=min(timings),
        )

    report.run("gradient_quantization_batch_%03d" % index, check)


def check_gradient_quantization(report, width, height):
    """Check an explicit numeric contract without changing old observations.
    验证独立数值契约，不改写旧观测。
    """
    key = "gradient_quantization"
    if key in report.contract_groups:
        return
    report.contract_groups.add(key)
    if not hasattr(report.processor, "mag_and_ang_ex"):
        report.missing(key, "mag_and_ang_ex")
        return
    points, retained, index = [], {}, 0
    for point in gradient_quantization_points():
        points.append(point)
        if len(points) == 4096:
            gradient_quantization_batch(
                report, points, index, [height, width], retained
            )
            points, index = [], index + 1
    if points:
        gradient_quantization_batch(report, points, index, [height, width], retained)


def check_ccl(report, width, height):
    source = np.zeros((height, width), dtype=np.uint8)
    source[0:7, 0:9] = 255
    source[30:40, 40:50] = 255
    source[60:75, 80:100] = 255
    source[65:70, 85:90] = 0
    source[height - 6 : height, width - 8 : width] = 255
    for index in range(8):
        source[100 + index, 130 + index] = 255
    source[150, 180] = 255
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        source, connectivity=8
    )
    expected = [
        dict(
            x=int(statistics[index, cv2.CC_STAT_LEFT]),
            y=int(statistics[index, cv2.CC_STAT_TOP]),
            w=int(statistics[index, cv2.CC_STAT_WIDTH]),
            h=int(statistics[index, cv2.CC_STAT_HEIGHT]),
            pixels=int(statistics[index, cv2.CC_STAT_AREA]),
        )
        for index in range(1, count)
    ]
    buffer = image(source)
    for minimum in (0, 1, 7, 8, 48, 63, 99, 100, 275):

        def geometry(minimum=minimum):
            wanted = sorted(
                [row for row in expected if row["pixels"] >= minimum],
                key=lambda row: (row["y"], row["x"]),
            )
            actual = report.processor.ccl(buffer, minimum)
            observed = sorted(
                [
                    dict(x=blob.x, y=blob.y, w=blob.w, h=blob.h, pixels=blob.pixels)
                    for blob in actual
                ],
                key=lambda row: (row["y"], row["x"]),
            )
            return dict(
                status="passed" if wanted == observed else "failed",
                minimum_area=minimum,
                expected=wanted,
                actual=observed,
                reported_centers=[[blob.cx, blob.cy] for blob in actual],
                reference_connectivity=8,
                threshold_contract="keep area >= min_area",
            )

        report.run("ccl_geometry_min_%d_%dx%d" % (minimum, width, height), geometry)

    def sparse_labels():
        merged = np.zeros((height, width), dtype=np.uint8)
        for index in range(20):
            left = 8 + index * 7
            merged[24:70, left : left + 3] = 255
        for group in range(5):
            left = 8 + group * 28
            merged[66:72, left : left + 24] = 255
        merged[120:132, width - 40 : width - 22] = 255
        count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
            merged, connectivity=8
        )
        expected = sorted(
            [
                [
                    int(statistics[index, cv2.CC_STAT_LEFT]),
                    int(statistics[index, cv2.CC_STAT_TOP]),
                    int(statistics[index, cv2.CC_STAT_WIDTH]),
                    int(statistics[index, cv2.CC_STAT_HEIGHT]),
                    int(statistics[index, cv2.CC_STAT_AREA]),
                ]
                for index in range(1, count)
            ]
        )
        actual = report.processor.ccl(image(merged), 1)
        observed = sorted(
            [[blob.x, blob.y, blob.w, blob.h, blob.pixels] for blob in actual]
        )
        return dict(
            status="passed" if expected == observed else "failed",
            expected=expected,
            actual=observed,
            reason="Twenty early labels merge into five regions before a late isolated region; valid SDK slots may be sparse",
        )

    report.run("ccl_sparse_merged_labels_%dx%d" % (width, height), sparse_labels)

    def histogram_empty_switch():
        populated = np.zeros((height, width), dtype=np.uint8)
        populated[20:26, 30:38] = 255
        populated_buffer = image(populated)
        empty_buffer = image(np.zeros((height, width), dtype=np.uint8))
        histogram_buffer = image(np.ones((height, width), dtype=np.uint8))
        expected = [[30, 20, 8, 6, 48]]
        steps, valid = [], True
        for iteration in range(max(2, report.loops)):
            blobs = report.processor.ccl(populated_buffer, 1)
            observed = sorted(
                [[blob.x, blob.y, blob.w, blob.h, blob.pixels] for blob in blobs]
            )
            histogram = list(report.processor.hist(histogram_buffer))
            before = interrupt_count()
            empty = report.processor.ccl(empty_buffer, 1)
            after = interrupt_count()
            delta = after - before if before is not None and after is not None else None
            correct_histogram = (
                len(histogram) == 256
                and histogram[1] == width * height
                and sum(histogram) == width * height
            )
            correct = (
                observed == expected
                and correct_histogram
                and len(empty) == 0
                and delta is not None
                and delta > 0
            )
            valid = valid and correct
            steps.append(
                dict(
                    iteration=iteration,
                    populated=observed,
                    histogram_valid=correct_histogram,
                    empty_regions=len(empty),
                    empty_interrupts=delta,
                    passed=correct,
                )
            )
        return dict(
            status="passed" if valid else "failed",
            steps=steps,
            reference="A nonempty CCL result and a histogram with a large bin-one count must not leak regions into the next empty CCL call",
        )

    report.run(
        "ccl_histogram_to_empty_switch_%dx%d" % (width, height), histogram_empty_switch
    )


def edge_observation(actual, references, radius):
    values = {}
    mask = np.ones(actual.shape, dtype=np.uint8)
    mask[radius:-radius, radius:-radius] = 0
    for name, reference in references.items():
        mismatch = cv2.compare(actual, reference, cv2.CMP_NE)
        values[name] = int(cv2.countNonZero(cv2.bitwise_and(mismatch, mask)))
    return values


def check_rank_morph(report, width, height):
    generator = random.Random(1106)
    source = (
        np.frombuffer(
            bytes(generator.randrange(256) for _ in range(width * height)),
            dtype=np.uint8,
        )
        .reshape(height, width)
        .copy()
    )
    source[:3, :] = 255
    source[:, :3] = 0
    source[-3:, :] = 0
    source[:, -3:] = 255
    buffer = image(source)
    for name in ("MIN", "MEDIAN", "MAX"):
        mode = getattr(vg.OrdStatFilterMode, name)
        if name == "MEDIAN":
            expected = cv2.medianBlur(source, 3)
        elif name == "MIN":
            expected = cv2.erode(source, np.ones((3, 3), dtype=np.uint8))
        else:
            expected = cv2.dilate(source, np.ones((3, 3), dtype=np.uint8))
        report.exact(
            "rank_%s_interior_%dx%d" % (name, width, height),
            lambda mode=mode: report.processor.ordered_stat_filter(buffer, mode),
            expected,
            border=2,
        )

        def boundary(name=name, mode=mode):
            actual = array(report.processor.ordered_stat_filter(buffer, mode))
            references = {}
            for label, kind in [
                ("constant_zero", cv2.BORDER_CONSTANT),
                ("replicate", cv2.BORDER_REPLICATE),
                ("reflect", cv2.BORDER_REFLECT),
                ("reflect101", cv2.BORDER_REFLECT_101),
            ]:
                if name == "MEDIAN":
                    padded = cv2.copyMakeBorder(source, 2, 2, 2, 2, kind, value=0)
                    references[label] = cv2.medianBlur(padded, 3)[2:-2, 2:-2]
                else:
                    operation = cv2.erode if name == "MIN" else cv2.dilate
                    references[label] = operation(
                        source,
                        np.ones((3, 3), dtype=np.uint8),
                        borderType=kind,
                        borderValue=0,
                    )
            return dict(
                status="unverified",
                edge_different_pixels=edge_observation(actual, references, 2),
                first_row=actual[0, :32].tolist(),
                reason="The public rank-filter border convention is unspecified",
            )

        report.run("rank_%s_boundary_%dx%d" % (name, width, height), boundary)
    binary = cv2.threshold(source, 127, 255, cv2.THRESH_BINARY)[1]
    binary_buffer = image(binary)
    for size in (3, 5):
        kernel = np.ones((size, size), dtype=np.uint8)
        for name in ("dilate", "erode"):
            operation = getattr(cv2, name)
            method = getattr(report.processor, name)
            expected = operation(binary, kernel)
            report.exact(
                "morph_%s_%d_interior_%dx%d" % (name, size, width, height),
                lambda size=size, method=method: method(binary_buffer, size),
                expected,
                border=2,
            )

            def boundary(size=size, name=name, operation=operation, method=method):
                actual = array(method(binary_buffer, size))
                references = {}
                for label, kind, value in [
                    ("constant_zero", cv2.BORDER_CONSTANT, 0),
                    ("constant_255", cv2.BORDER_CONSTANT, 255),
                    ("replicate", cv2.BORDER_REPLICATE, 0),
                    ("reflect101", cv2.BORDER_REFLECT_101, 0),
                ]:
                    references[label] = operation(
                        binary,
                        np.ones((size, size), dtype=np.uint8),
                        borderType=kind,
                        borderValue=value,
                    )
                return dict(
                    status="unverified",
                    edge_different_pixels=edge_observation(actual, references, 2),
                    first_row=actual[0, :32].tolist(),
                    reason="The public morphology border convention is unspecified",
                )

            report.run(
                "morph_%s_%d_boundary_%dx%d" % (name, size, width, height), boundary
            )
    method = getattr(report.processor, "morphology", None)
    if method is None:
        report.missing(
            "morphology_masks",
            "Expose binary 25-element morphology masks with an explicit border contract",
        )
        return
    asymmetric = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    cross = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    hollow = np.ones((5, 5), dtype=np.uint8)
    hollow[1:4, 1:4] = 0
    identity = np.zeros((5, 5), dtype=np.uint8)
    identity[2, 2] = 1
    # Revisit the first mask after switching state. / 切换状态后再次使用第一个掩码。
    for label, kernel in [
        ("asymmetric_first", asymmetric),
        ("cross", cross),
        ("hollow", hollow),
        ("identity", identity),
        ("asymmetric_again", asymmetric),
    ]:
        values = kernel.ravel().tolist()
        for dilate in (True, False):
            expected = (cv2.dilate if dilate else cv2.erode)(
                binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            report.exact(
                "morphology_%s_%s_full_frame_%dx%d"
                % ("dilate" if dilate else "erode", label, width, height),
                lambda values=values, dilate=dilate: method(
                    binary_buffer, values, dilate
                ),
                expected,
            )
    report.rejected(
        "morphology_nonbinary_rejected_%dx%d" % (width, height),
        lambda: method(buffer, cross.ravel().tolist(), True),
    )


def border_convolution_reference(source, mask, norm, low, high, dtype):
    """Apply the existing integer-shift reference to a replicated two-pixel halo.
    为输入补两像素复制边界，再使用已有整数移位参考。
    """
    padded = cv2.copyMakeBorder(source, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    return convolution_reference(padded, mask, norm, low, high, dtype)[2:-2, 2:-2]


def border_contract_exact(report, name, call, inputs, expected, requested, reference):
    """Check full outputs, each hardware submission and retained buffer contents.
    检查完整输出、每次硬件提交及保留缓冲区的内容。
    """

    def check():
        snapshots = [array(value) for value in inputs]
        signatures, comparisons, irqs, timings = [], [], [], []
        inputs_preserved, outputs_preserved = True, True
        first_outputs, first_values = None, None
        dtype_matches, shape_matches = True, True
        for _ in range(report.loops):
            before, start = interrupt_count(), time.perf_counter()
            outputs = call()
            actual = {key: array(value) for key, value in outputs.items()}
            timings.append((time.perf_counter() - start) * 1000)
            after = interrupt_count()
            irqs.append(
                after - before if before is not None and after is not None else None
            )
            if set(actual) != set(expected):
                raise ValueError("Output components differ from the reference")
            comparison = {}
            for key, expected_array in expected.items():
                comparison[key] = difference(actual[key], expected_array)
                dtype_matches = (
                    dtype_matches and actual[key].dtype.str == expected_array.dtype.str
                )
                shape_matches = shape_matches and list(actual[key].shape) == list(
                    expected_array.shape
                )
            comparisons.append(comparison)
            signatures.append(
                {
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in actual.items()
                }
            )
            inputs_preserved = inputs_preserved and all(
                difference(array(value), snapshot)["different"] == 0
                for value, snapshot in zip(inputs, snapshots)
            )
            outputs_preserved = outputs_preserved and all(
                difference(array(outputs[key]), actual[key])["different"] == 0
                for key in actual
            )
            if first_outputs is None:
                first_outputs, first_values = outputs, actual
            else:
                outputs_preserved = outputs_preserved and all(
                    difference(array(first_outputs[key]), first_values[key])[
                        "different"
                    ]
                    == 0
                    for key in first_values
                )
        unique = {
            key: len({signature[key] for signature in signatures}) for key in expected
        }
        passed = (
            all(
                value["different"] == 0
                for comparison in comparisons
                for value in comparison.values()
            )
            and dtype_matches
            and shape_matches
            and inputs_preserved
            and outputs_preserved
            and all(count == 1 for count in unique.values())
        )
        status = "passed" if passed else "failed"
        if any(value is not None and value <= 0 for value in irqs):
            status = "failed"
        elif status == "passed" and any(value is None for value in irqs):
            status = "unverified"
        return dict(
            status=status,
            reference=reference,
            requested_shape=list(requested),
            fixture_shape=list(snapshots[0].shape),
            fixture_policy="Bounded deterministic fixture; every pixel including the boundary is compared",
            excluded_border=0,
            comparisons=comparisons,
            dtypes={key: value.dtype.str for key, value in actual.items()},
            expected_dtypes={key: value.dtype.str for key, value in expected.items()},
            shapes={key: list(value.shape) for key, value in actual.items()},
            dtype_matches=dtype_matches,
            shape_matches=shape_matches,
            inputs_preserved=inputs_preserved,
            outputs_preserved=outputs_preserved,
            unique_outputs=unique,
            signatures=signatures,
            interrupts_per_call=irqs,
            min_ms=min(timings),
        )

    report.run(name, check)


def check_border_contract(report, width, height):
    """Test explicit full-frame contracts without reclassifying earlier observations.
    验证明确的全图契约，不更改此前观测的验证状态。
    """
    fixture_width, fixture_height = 66, 66
    requested = [height, width]
    generator = random.Random(281107)
    edge = np.frombuffer(
        bytes(
            generator.randrange(1, 255) for _ in range(fixture_width * fixture_height)
        ),
        dtype=np.uint8,
    ).reshape(fixture_height, fixture_width)
    saturation = np.asarray(
        [
            [
                255 if (x + y) % 2 else (x * 17 + y * 29) % 3
                for x in range(fixture_width)
            ]
            for y in range(fixture_height)
        ],
        dtype=np.uint8,
    )
    tap, signed_difference = [0] * 25, [0] * 25
    tap[5] = 1
    signed_difference[5], signed_difference[12] = -1, 1
    checker = [127 if (x + y) % 2 else -127 for y in range(5) for x in range(5)]
    checker[12] = 3
    fixtures = {
        "edge": (edge, image(edge)),
        "saturation": (saturation, image(saturation)),
    }
    formats = [
        ("U8C1", np.uint8, 0, 255),
        ("S8C1", np.int8, -128, 127),
        ("U16C1", np.uint16, 0, 65535),
        ("S16C1", np.int16, -32768, 32767),
    ]
    contracts = [("tap_boundary", "edge", tap, 0, "U16C1")]
    for type_name, _, _, _ in formats:
        for norm in (1, 3):
            contracts.append(
                ("small_signed", "edge", signed_difference, norm, type_name)
            )
        contracts.append(("saturation", "saturation", checker, 0, type_name))
    for type_name in ("U16C1", "S16C1"):
        contracts.append(("wide_signed", "saturation", checker, 7, type_name))
    type_enum = getattr(vg, "ImageTypeIVE", None)
    for method_name in ("filter_ex", "sobel_ex"):
        method = getattr(report.processor, method_name, None)
        sobel_ctrl = getattr(getattr(vg, "SobelOutCtrl", None), "BOTH", None)
        if (
            method is None
            or type_enum is None
            or (method_name == "sobel_ex" and sobel_ctrl is None)
        ):
            report.missing(
                "border_contract_" + method_name, "Expose typed filter/Sobel controls"
            )
            continue
        for label, source_name, values, norm, type_name in contracts:
            output_format = getattr(type_enum, type_name, None)
            if output_format is None:
                report.missing(
                    "border_contract_" + type_name, "Expose ImageTypeIVE." + type_name
                )
                continue
            source, buffer = fixtures[source_name]
            _, dtype, low, high = next(item for item in formats if item[0] == type_name)
            mask = np.asarray(values, dtype=np.float32).reshape(5, 5)
            expected_h = border_convolution_reference(
                source, mask, norm, low, high, dtype
            )
            if method_name == "filter_ex":
                expected = dict(value=expected_h)

                def call(
                    method=method,
                    buffer=buffer,
                    values=values,
                    norm=norm,
                    output_format=output_format,
                ):
                    return dict(value=method(buffer, values, norm, output_format))

            else:
                expected = dict(
                    horizontal=expected_h,
                    vertical=border_convolution_reference(
                        source, mask.T, norm, low, high, dtype
                    ),
                )

                def call(
                    method=method,
                    buffer=buffer,
                    values=values,
                    norm=norm,
                    output_format=output_format,
                    sobel_ctrl=sobel_ctrl,
                ):
                    horizontal, vertical = method(
                        buffer, values, norm, sobel_ctrl, output_format
                    )
                    return dict(horizontal=horizontal, vertical=vertical)

            border_contract_exact(
                report,
                "border_contract_%s_%s_%s_shift%d_requested%dx%d"
                % (method_name, label, type_name, norm, width, height),
                call,
                [buffer],
                expected,
                requested,
                "Full-frame observable contract: replicate edge coordinates, integer arithmetic right shift, then destination-type saturation",
            )
    normalized = getattr(report.processor, "norm_grad_config", None)
    norm_enum = getattr(vg, "NormGradOutCtrl", None)
    if normalized is None or norm_enum is None:
        report.missing(
            "border_contract_normgrad", "Expose normalized gradient modes and masks"
        )
    else:
        normalized_cases = [
            ("small_signed", "edge", signed_difference, 1, "HOR_AND_VER"),
            ("small_signed", "edge", signed_difference, 3, "HOR_AND_VER"),
            ("small_signed", "edge", signed_difference, 1, "HOR"),
            ("small_signed", "edge", signed_difference, 1, "VER"),
            ("small_signed", "edge", signed_difference, 1, "COMBINE"),
            ("saturation", "saturation", checker, 0, "HOR_AND_VER"),
            ("wide_signed", "saturation", checker, 15, "HOR_AND_VER"),
        ]
        for label, source_name, values, norm, mode_name in normalized_cases:
            mode = getattr(norm_enum, mode_name, None)
            if mode is None:
                report.missing(
                    "border_contract_normgrad_" + mode_name,
                    "Expose NormGradOutCtrl." + mode_name,
                )
                continue
            source, buffer = fixtures[source_name]
            mask = np.asarray(values, dtype=np.float32).reshape(5, 5)
            expected_h = border_convolution_reference(
                source, mask, norm, -128, 127, np.int8
            )
            expected_v = border_convolution_reference(
                source, mask.T, norm, -128, 127, np.int8
            )
            if mode_name == "COMBINE":
                expected = dict(
                    combined=np.asarray(
                        [
                            [
                                (int(h) & 255) | ((int(v) & 255) << 8)
                                for h, v in zip(hrow, vrow)
                            ]
                            for hrow, vrow in zip(
                                expected_h.tolist(), expected_v.tolist()
                            )
                        ],
                        dtype=np.uint16,
                    )
                )
            else:
                expected = {}
                if mode_name != "VER":
                    expected["horizontal"] = expected_h
                if mode_name != "HOR":
                    expected["vertical"] = expected_v

            def call(
                buffer=buffer, mode=mode, norm=norm, values=values, mode_name=mode_name
            ):
                horizontal, vertical, combined = normalized(buffer, mode, norm, values)
                if mode_name == "COMBINE":
                    return dict(combined=combined)
                result = {}
                if mode_name != "VER":
                    result["horizontal"] = horizontal
                if mode_name != "HOR":
                    result["vertical"] = vertical
                return result

            border_contract_exact(
                report,
                "border_contract_normgrad_%s_%s_shift%d_requested%dx%d"
                % (label, mode_name, norm, width, height),
                call,
                [buffer],
                expected,
                requested,
                "Full-frame replicate/floor/signed-byte saturation; COMBINE stores horizontal in the low byte and vertical in the high byte",
            )
    rank = getattr(report.processor, "ordered_stat_filter", None)
    rank_enum = getattr(vg, "OrdStatFilterMode", None)
    edge_buffer = fixtures["edge"][1]
    if rank is None or rank_enum is None:
        report.missing("border_contract_rank", "Expose the three order-statistic modes")
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
        for mode_name in ("MIN", "MEDIAN", "MAX"):
            mode = getattr(rank_enum, mode_name, None)
            if mode is None:
                report.missing(
                    "border_contract_rank_" + mode_name,
                    "Expose OrdStatFilterMode." + mode_name,
                )
                continue
            expected = (
                cv2.medianBlur(edge, 3)
                if mode_name == "MEDIAN"
                else (cv2.erode if mode_name == "MIN" else cv2.dilate)(
                    edge, kernel, borderType=cv2.BORDER_REPLICATE
                )
            )
            border_contract_exact(
                report,
                "border_contract_rank_%s_requested%dx%d" % (mode_name, width, height),
                lambda mode=mode: dict(value=rank(edge_buffer, mode)),
                [edge_buffer],
                dict(value=expected),
                requested,
                "Observable 3x3 rank output with clamped outside coordinates; mathematically equivalent boundary extensions do not identify internal padding",
            )
    morphology = getattr(report.processor, "morphology", None)
    if morphology is None:
        report.missing(
            "border_contract_morphology", "Expose explicit binary morphology masks"
        )
    else:
        binary = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)[1]
        binary_buffer = image(binary)
        mask_values = [1 if index in (0, 8, 12, 21) else 0 for index in range(25)]
        kernel = np.asarray(mask_values, dtype=np.uint8).reshape(5, 5)
        for dilate in (False, True):
            expected = (cv2.dilate if dilate else cv2.erode)(
                binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            border_contract_exact(
                report,
                "border_contract_morphology_asymmetric_%s_requested%dx%d"
                % ("dilate" if dilate else "erode", width, height),
                lambda dilate=dilate: dict(
                    value=morphology(binary_buffer, mask_values, dilate)
                ),
                [binary_buffer],
                dict(value=expected),
                requested,
                "Full-frame asymmetric binary morphology with an explicit constant-zero outside-image contract",
            )


ARITHMETIC_WEIGHTS = (
    (0, 0),
    (0, 65535),
    (65535, 0),
    (32768, 32768),
    (16384, 49152),
    (1, 65535),
    (65535, 1),
    (12345, 43210),
    (21845, 43691),
    (65535, 65535),
    (32767, 32769),
    (1, 1),
    (32768, 0),
    (65535, 32768),
    (257, 65279),
)


def arithmetic_pair_bytes(pixels):
    """Cover every U8 pair before repeating padding. / 覆盖全部 U8 像素对后重复填充。"""
    if pixels < 65536:
        raise ValueError("A complete U8 pair fixture needs at least 65536 pixels")
    return (
        bytes((index % 65536) // 256 for index in range(pixels)),
        bytes(index % 256 for index in range(pixels)),
    )


def arithmetic_add_reference(first, second, x, y):
    """Sum products, truncate Q0.16, then wrap to U8. / 合并乘积、截断 Q0.16 后回绕到 U8。"""
    return bytes(((a * x + b * y) // 65536) % 256 for a, b in zip(first, second))


def arithmetic_histograms(pixels):
    """Use nondegenerate distributions with exact rounding boundaries.
    使用包含精确舍入边界的非退化分布。
    """
    if pixels < 65536:
        raise ValueError("Arithmetic histogram fixtures need at least 65536 pixels")

    def histogram(items):
        result = [0] * 256
        for value, count in items:
            result[value] += count
        if (
            sum(result) != pixels
            or min(result) < 0
            or sum(value > 0 for value in result) < 2
        ):
            raise ValueError("Invalid nondegenerate histogram fixture")
        return result

    ramp = Counter(index % 256 for index in range(pixels))
    result = dict(ramp=[ramp[value] for value in range(256)])
    result["two_peaks"] = histogram([(0, pixels // 2), (255, pixels - pixels // 2)])
    result["sparse"] = histogram(
        [
            (17, 1),
            (20, pixels // 7),
            (129, pixels // 3),
            (240, pixels - 1 - pixels // 7 - pixels // 3),
        ]
    )
    for denominator in (50999, 51000, 51001):
        result["half_boundary_d%d" % denominator] = histogram(
            [
                (17, pixels - denominator),
                (18, 100),
                (19, 200),
                (20, 200),
                (239, denominator - 500),
            ]
        )
    result["last_singleton"] = histogram([(31, pixels - 2), (127, 1), (255, 1)])
    result["first_singleton"] = histogram([(0, 1), (127, pixels - 2), (255, 1)])
    for denominator, cumulative in ((26447, 26084), (27841, 24511)):
        result["precision_boundary_d%d" % denominator] = histogram(
            [
                (17, pixels - denominator),
                (18, cumulative),
                (239, denominator - cumulative),
            ]
        )
    result["unequal_two_peaks"] = histogram(
        [(17, pixels // 3), (239, pixels - pixels // 3)]
    )
    result["three_peaks"] = histogram(
        [(5, 1), (6, pixels // 2), (254, pixels - 1 - pixels // 2)]
    )
    result["occupied_half"] = histogram(
        [
            (value, pixels // 128 + (value - 64 < pixels % 128))
            for value in range(64, 192)
        ]
    )
    return result


def arithmetic_equalize_table(histogram):
    """CDF-min with exact rational half-up rounding. / CDF-min 精确有理数半入舍入。"""
    minimum = next((count for count in histogram if count), 0)
    denominator = sum(histogram) - minimum
    if denominator <= 0:
        raise ValueError("Single-bin equalization needs its own verified contract")
    running, table = 0, []
    for count in histogram:
        running += count
        table.append(
            max(
                0,
                min(
                    255,
                    (255 * max(0, running - minimum) + denominator // 2) // denominator,
                ),
            )
        )
    return bytes(table)


def arithmetic_reference_self_test():
    """Check literal arithmetic boundaries without hardware or array libraries.
    不使用硬件或数组库，检查明确的算术边界。
    """
    first, second = arithmetic_pair_bytes(65536)
    if any(
        (a, b) != (index // 256, index % 256)
        for index, (a, b) in enumerate(zip(first, second))
    ):
        raise AssertionError("Weighted-add fixture lost a U8 pair")
    checks = 1
    for a, b, x, y, expected in (
        (128, 129, 65535, 65535, 0),
        (255, 255, 65535, 65535, 253),
        (1, 2, 32768, 32768, 1),
        (1, 1, 32768, 32768, 1),
        (255, 0, 65535, 0, 254),
        (0, 255, 0, 65535, 254),
    ):
        if arithmetic_add_reference(bytes([a]), bytes([b]), x, y) != bytes([expected]):
            raise AssertionError(
                "Weighted-add reference disagrees with a literal fixture"
            )
        checks += 1
    fixtures = arithmetic_histograms(76800)
    if len(fixtures) != 13:
        raise AssertionError(
            "Equalization coverage must contain13 nondegenerate histograms"
        )
    for histogram in fixtures.values():
        table = arithmetic_equalize_table(histogram)
        if len(table) != 256 or table[-1] != 255 or sum(histogram) != 76800:
            raise AssertionError("Invalid equalization map or histogram total")
        checks += 1
    if list(arithmetic_equalize_table(fixtures["half_boundary_d51000"])[18:21]) != [
        1,
        2,
        3,
    ]:
        raise AssertionError("Equalization accepted ties-even instead of half-up")
    if (
        arithmetic_equalize_table(fixtures["precision_boundary_d26447"])[18] != 251
        or arithmetic_equalize_table(fixtures["precision_boundary_d27841"])[18] != 225
    ):
        raise AssertionError("Equalization lost its exact rational precision boundary")
    checks += 2
    return checks


def arithmetic_contract_exact(
    report,
    name,
    call,
    inputs,
    expected,
    expected_counts,
    requested,
    reference,
    fixture_policy="Every U8 pair for Add; explicitly nondegenerate equalization histograms",
):
    """Check arithmetic outputs, per-call IRQ and buffer preservation.
    检查算术输出、逐次硬件中断和缓冲区完整性。
    """

    def check():
        snapshots = [array(value) for value in inputs]
        signatures, comparisons, irqs, timings = [], [], [], []
        inputs_preserved, outputs_preserved, dtype_matches, shape_matches = (
            True,
            True,
            True,
            True,
        )
        counts_match, image_validity = True, True
        first_output, first_bytes = None, None
        actual_dtype, actual_shape = None, None
        for _ in range(report.loops):
            before, started = interrupt_count(), time.perf_counter()
            result, counts = call()
            after = interrupt_count()
            timings.append((time.perf_counter() - started) * 1000)
            irqs.append(
                after - before if before is not None and after is not None else None
            )
            counts = list(counts)
            counts_match = counts_match and counts == expected_counts
            valid = result.is_valid()
            image_validity = image_validity and valid == (expected is not None)
            if expected is None:
                comparisons.append(
                    dict(
                        different=(
                            0
                            if counts == expected_counts
                            else sum(a != b for a, b in zip(counts, expected_counts))
                            + abs(len(counts) - len(expected_counts))
                        ),
                        maximum=0 if counts == expected_counts else None,
                        shape_matches=len(counts) == len(expected_counts),
                    )
                )
                signatures.append(
                    hashlib.sha256(
                        json.dumps(counts, separators=(",", ":")).encode("ascii")
                    ).hexdigest()
                )
            elif valid:
                actual = array(result)
                actual_dtype, actual_shape = actual.dtype.str, list(actual.shape)
                dtype_matches = (
                    dtype_matches
                    and actual.dtype.str == expected.dtype.str
                    and str(result.format).lower() == "gray8"
                )
                shape_matches = (
                    shape_matches
                    and list(actual.shape) == list(expected.shape)
                    and result.width == expected.shape[1]
                    and result.height == expected.shape[0]
                )
                comparisons.append(difference(actual, expected))
                raw = actual.tobytes()
                signatures.append(hashlib.sha256(raw).hexdigest())
                if first_output is None:
                    first_output, first_bytes = result, raw
                outputs_preserved = (
                    outputs_preserved and array(first_output).tobytes() == first_bytes
                )
            else:
                comparisons.append(
                    dict(different=None, maximum=None, shape_matches=False)
                )
            inputs_preserved = inputs_preserved and all(
                difference(array(value), snapshot)["different"] == 0
                for value, snapshot in zip(inputs, snapshots)
            )
        passed = (
            all(row["different"] == 0 for row in comparisons)
            and len(set(signatures)) == 1
            and counts_match
            and image_validity
            and inputs_preserved
            and outputs_preserved
            and dtype_matches
            and shape_matches
        )
        status = "passed" if passed else "failed"
        if any(count is not None and count != 1 for count in irqs):
            status = "failed"
        elif status == "passed" and any(count is None for count in irqs):
            status = "unverified"
        return dict(
            status=status,
            reference=reference,
            requested_shape=list(requested),
            fixture_shape=list(snapshots[0].shape),
            fixture_policy=fixture_policy,
            comparisons=comparisons,
            dtype=actual_dtype,
            expected_dtype=None if expected is None else expected.dtype.str,
            shape=actual_shape,
            dtype_matches=dtype_matches,
            shape_matches=shape_matches,
            image_validity_matches=image_validity,
            counts_match=counts_match,
            inputs_preserved=inputs_preserved,
            outputs_preserved=outputs_preserved,
            input_sha256=[
                hashlib.sha256(snapshot.tobytes()).hexdigest() for snapshot in snapshots
            ],
            signatures=signatures,
            unique_outputs=len(set(signatures)),
            interrupts_per_call=irqs,
            min_ms=min(timings),
        )

    report.run(name, check)


def check_arithmetic(report, width, height):
    """Test measured arithmetic rules separately from earlier observations.
    独立验证实测算术规则，保留此前观测的状态。
    """
    fixture_width, fixture_height = (
        (width, height) if width * height >= 65536 else (256, 256)
    )
    pixels = fixture_width * fixture_height
    requested = [height, width]
    first, second = arithmetic_pair_bytes(pixels)

    def data_image(raw):
        return image(
            np.frombuffer(raw, dtype=np.uint8).reshape(fixture_height, fixture_width)
        )

    first_buffer, second_buffer = data_image(first), data_image(second)
    add = getattr(report.processor, "add_weighted", None)
    if add is None:
        report.missing(
            "arithmetic_add_weighted", "Expose independent Q0.16 image weights"
        )
    else:
        for x, y in ARITHMETIC_WEIGHTS:
            expected = np.frombuffer(
                arithmetic_add_reference(first, second, x, y), dtype=np.uint8
            ).reshape(fixture_height, fixture_width)
            arithmetic_contract_exact(
                report,
                "arithmetic_add_%d_%d_requested%dx%d" % (x, y, width, height),
                lambda x=x, y=y: (add(first_buffer, second_buffer, x, y), []),
                [first_buffer, second_buffer],
                expected,
                [],
                requested,
                "Every U8 pair: sum Q0.16 products, floor once, then U8 modulo256; no saturation",
            )
    equalize, enum = getattr(report.processor, "equalize_hist_ex", None), getattr(
        vg, "EqualizeMode", None
    )
    if equalize is None or enum is None:
        report.missing(
            "arithmetic_equalize_hist",
            "Expose histogram-only, internal and external histogram modes",
        )
        return
    ramp = bytes(index % 256 for index in range(pixels))
    ramp_buffer = data_image(ramp)
    for label, histogram in arithmetic_histograms(pixels).items():
        raw = b"".join(bytes([value]) * count for value, count in enumerate(histogram))
        buffer = data_image(raw)
        table = arithmetic_equalize_table(histogram)
        for mode_name, mode, counts, source_buffer, source_raw in (
            ("HIST", enum.HIST, [], buffer, raw),
            ("EQHIST", enum.EQHIST, [], buffer, raw),
            ("EXT_SAME", enum.EQHIST_WITH_EXT_HIST, histogram, buffer, raw),
            ("EXT_DIFFERENT", enum.EQHIST_WITH_EXT_HIST, histogram, ramp_buffer, ramp),
        ):
            if mode_name == "EXT_DIFFERENT" and label == "ramp":
                continue
            expected = (
                None
                if mode_name == "HIST"
                else np.frombuffer(source_raw.translate(table), dtype=np.uint8).reshape(
                    fixture_height, fixture_width
                )
            )
            arithmetic_contract_exact(
                report,
                "arithmetic_equalize_%s_%s_requested%dx%d"
                % (label, mode_name, width, height),
                lambda source_buffer=source_buffer, mode=mode, counts=counts: equalize(
                    source_buffer, mode, counts
                ),
                [source_buffer],
                expected,
                histogram if mode_name == "HIST" else [],
                requested,
                (
                    "Exact256bin histogram"
                    if mode_name == "HIST"
                    else "Nondegenerate CDF-min, exact integer normalization and half-up rounding; external modes use the supplied distribution"
                ),
            )
    if add is not None:
        for name, x, y in (
            ("weight1_negative", -1, 0),
            ("weight1_overflow", 65536, 0),
            ("weight2_negative", 0, -1),
            ("weight2_overflow", 0, 65536),
        ):
            report.rejected(
                "arithmetic_" + name,
                lambda x=x, y=y: add(first_buffer, second_buffer, x, y),
            )
    word_buffer = image(
        np.full((fixture_height, fixture_width), 257, dtype=np.uint16), "u16c1"
    )
    if add is not None:
        smaller = image(np.zeros((fixture_height, fixture_width - 2), dtype=np.uint8))
        report.rejected(
            "arithmetic_add_wrong_type",
            lambda: add(word_buffer, second_buffer),
        )
        report.rejected(
            "arithmetic_add_wrong_shape",
            lambda: add(first_buffer, smaller),
        )
    report.rejected("arithmetic_equalize_wrong_type", lambda: equalize(word_buffer))
    histogram = arithmetic_histograms(pixels)["ramp"]
    for name, mode, counts in (
        ("external_missing", enum.EQHIST_WITH_EXT_HIST, []),
        ("external_short", enum.EQHIST_WITH_EXT_HIST, histogram[:-1]),
        ("external_long", enum.EQHIST_WITH_EXT_HIST, histogram + [0]),
        ("external_zero_total", enum.EQHIST_WITH_EXT_HIST, [0] * 256),
        (
            "external_total_plus_one",
            enum.EQHIST_WITH_EXT_HIST,
            [histogram[0] + 1] + histogram[1:],
        ),
        (
            "external_total_minus_one",
            enum.EQHIST_WITH_EXT_HIST,
            [histogram[0] - 1] + histogram[1:],
        ),
        ("external_negative_bin", enum.EQHIST_WITH_EXT_HIST, [-1] + histogram[1:]),
        ("external_overflow_bin", enum.EQHIST_WITH_EXT_HIST, [2**32] + histogram[1:]),
        ("hist_unexpected_bins", enum.HIST, histogram),
        ("equalize_unexpected_bins", enum.EQHIST, histogram),
    ):
        report.rejected(
            "arithmetic_" + name,
            lambda mode=mode, counts=counts: equalize(ramp_buffer, mode, counts),
        )


def equalize_single_bin_table(value):
    """Map below the occupied bin to zero, then to that bin's value.
    低于唯一非零 bin 的输入映射为零，其余映射到该 bin 的值。
    """
    if not 0 <= value <= 255:
        raise ValueError("The occupied histogram bin must be in [0,255]")
    return bytes(value) + bytes([value]) * (256 - value)


def equalize_degenerate_reference_self_test():
    """Check all maps without captured lookup tables. / 不使用采集表，检查全部映射。"""
    identity_differences, step_differences = 0, 0
    for value in range(256):
        table = equalize_single_bin_table(value)
        if len(table) != 256 or any(
            table[index] != (value if index >= value else 0) for index in range(256)
        ):
            raise AssertionError(
                "Single-bin map disagrees with the independent reference"
            )
        if table[value] != value:
            raise AssertionError(
                "Internal constant equalization did not preserve its value"
            )
        identity_differences += sum(table[index] != index for index in range(256))
        step_differences += sum(
            table[index] != (255 if index >= value else 0) for index in range(256)
        )
    if (identity_differences, step_differences) != (65025, 32895):
        raise AssertionError(
            "Single-bin references did not reject the alternative maps"
        )
    return 258


def check_equalize_degenerate(report, width, height):
    """Check all single-bin maps once per run at the measured 256-square size.
    使用已实测的 256 方形尺寸，每轮运行只检查一次全部单 bin 映射。
    """
    key = "equalize_single_bin_full_domain"
    if key in report.contract_groups:
        return
    report.contract_groups.add(key)
    equalize = getattr(report.processor, "equalize_hist_ex", None)
    enum = getattr(vg, "EqualizeMode", None)
    if equalize is None or enum is None:
        report.missing(
            "equalize_degenerate", "Expose internal and external histogram equalization"
        )
        return
    requested = [height, width]
    ramp_raw = bytes(range(256)) * 256
    ramp_buffer = image(np.frombuffer(ramp_raw, dtype=np.uint8).reshape(256, 256))
    policy = "Fixed256x256, N65536; all256 occupied bins and every external input value; once per complete run regardless of requested sizes"
    for value in range(256):
        constant_raw = bytes([value]) * 65536
        constant = np.frombuffer(constant_raw, dtype=np.uint8).reshape(256, 256)
        buffer = image(constant)
        histogram = [0] * 256
        histogram[value] = 65536
        external = np.frombuffer(
            equalize_single_bin_table(value) * 256, dtype=np.uint8
        ).reshape(256, 256)
        for name, mode, supplied, source, expected in (
            ("internal", enum.EQHIST, [], buffer, constant),
            ("external", enum.EQHIST_WITH_EXT_HIST, histogram, ramp_buffer, external),
        ):
            arithmetic_contract_exact(
                report,
                "equalize_degenerate_%s_bin%d_fixed256_requested%dx%d"
                % (name, value, width, height),
                lambda source=source, mode=mode, supplied=supplied: equalize(
                    source, mode, supplied
                ),
                [source],
                expected,
                [],
                requested,
                dict(
                    contract="single_bin_equalization_256square",
                    occupied_bin=value,
                    histogram_total=65536,
                    formula="L(v)=0 when v<k,otherwise k; internal constant k is preserved",
                    requested_size_policy="Metadata only; fixed256x256 coverage is deduplicated across requested sizes",
                ),
                fixture_policy=policy,
            )


def check_extended(report, width, height):
    source = np.asarray(
        [
            [(x * 17 + y * 31 + x * y % 19) % 256 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    other = np.asarray(
        [
            [(x * 3 + y * 29 + x * y % 23) % 256 for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint8,
    )
    buffer, second = image(source), image(other)
    if hasattr(report.processor, "filter_ex"):
        identity = np.zeros((5, 5), dtype=np.float32)
        identity[2, 2] = 1
        negative = identity * -127
        positive = np.full((5, 5), 127, dtype=np.float32)
        report.rejected(
            "filter_ex_shift16_rejected_%dx%d" % (width, height),
            lambda: report.processor.filter_ex(
                buffer,
                identity.astype(np.int8).ravel().tolist(),
                16,
                vg.ImageTypeIVE.U8C1,
            ),
        )
        for label, mask in [
            ("identity", identity),
            ("negative", negative),
            ("positive_saturation", positive),
        ]:
            for name, dtype, low, high in [
                ("U8C1", np.uint8, 0, 255),
                ("S8C1", np.int8, -128, 127),
                ("U16C1", np.uint16, 0, 65535),
                ("S16C1", np.int16, -32768, 32767),
            ]:
                output_format = getattr(vg.ImageTypeIVE, name, None)
                if output_format is None:
                    report.missing("filter_ex_" + name, "Expose ImageTypeIVE." + name)
                    continue
                mask_values = mask.astype(np.int8).ravel().tolist()
                expected = convolution_reference(source, mask, 0, low, high, dtype)
                report.exact(
                    "filter_ex_%s_%s_%dx%d" % (label, name, width, height),
                    lambda mask_values=mask_values, output_format=output_format: report.processor.filter_ex(
                        buffer, mask_values, 0, output_format
                    ),
                    expected,
                    border=2,
                    expected_dtype=expected.dtype.str,
                )
    else:
        report.missing("filter_ex", "Expose explicit mask/shift and four output types")
    map_method, map_enum = getattr(report.processor, "map_typed", None), getattr(
        vg, "MapMode", None
    )
    if map_method is not None and map_enum is not None:
        for name, dtype, table in [
            ("U8", np.uint8, [value ^ 173 for value in range(256)]),
            ("S16", np.int16, [(value - 128) * 255 for value in range(256)]),
            ("U16", np.uint16, [value * 257 for value in range(256)]),
        ]:
            mode = getattr(map_enum, name)
            expected = np.asarray(
                [[table[value] for value in row] for row in source.tolist()],
                dtype=dtype,
            )
            report.exact(
                "map_typed_%s_%dx%d" % (name, width, height),
                lambda table=table, mode=mode: map_method(buffer, table, mode),
                expected,
                expected_dtype=expected.dtype.str,
            )
        report.rejected(
            "map_typed_short_table_%dx%d" % (width, height),
            lambda: map_method(buffer, [0] * 255, map_enum.U8),
        )
    else:
        report.missing("map_typed", "Expose U8, S16 and U16 lookup tables")
    if hasattr(report.processor, "ncc_raw"):

        def ncc_raw():
            actual = list(report.processor.ncc_raw(buffer, second))
            first_values, second_values = (
                source.ravel().tolist(),
                other.ravel().tolist(),
            )
            expected = [
                sum(a * b for a, b in zip(first_values, second_values)),
                sum(value * value for value in first_values),
                sum(value * value for value in second_values),
            ]
            return dict(
                status="passed" if actual == expected else "failed",
                actual=actual,
                expected=expected,
            )

        report.run("ncc_raw_%dx%d" % (width, height), ncc_raw)
    else:
        report.missing(
            "ncc_raw", "Expose unnormalized 64-bit cross product and square sums"
        )
    if hasattr(report.processor, "add_weighted"):
        first_constant = image(np.full((height, width), 64, dtype=np.uint8))
        second_constant = image(np.full((height, width), 192, dtype=np.uint8))
        for a, b, value in [(0, 0, 0), (32768, 32768, 128), (16384, 49152, 160)]:
            report.exact(
                "add_weighted_exact_%d_%d_%dx%d" % (a, b, width, height),
                lambda a=a, b=b: report.processor.add_weighted(
                    first_constant, second_constant, a, b
                ),
                np.full((height, width), value, dtype=np.uint8),
            )

        def fractional_add():
            actual = array(report.processor.add_weighted(buffer, second, 32768, 32768))
            floor = np.asarray(
                [
                    [(a + b) // 2 for a, b in zip(row_a, row_b)]
                    for row_a, row_b in zip(source.tolist(), other.tolist())
                ],
                dtype=np.uint8,
            )
            nearest = np.asarray(
                [
                    [(a + b + 1) // 2 for a, b in zip(row_a, row_b)]
                    for row_a, row_b in zip(source.tolist(), other.tolist())
                ],
                dtype=np.uint8,
            )
            return dict(
                status="unverified",
                candidates=crop_candidates(
                    actual, dict(floor=floor, nearest_positive_tie=nearest)
                ),
                reason="Establish weighted-add rounding independently of the cast unit",
            )

        report.run("add_weighted_fractional_%dx%d" % (width, height), fractional_add)
    else:
        report.missing("add_weighted", "Expose independent Q0.16 image weights")
    equalize, enum = getattr(report.processor, "equalize_hist_ex", None), getattr(
        vg, "EqualizeMode", None
    )
    if equalize is None or enum is None:
        report.missing(
            "equalize_hist_ex", "Expose histogram-only and external-histogram modes"
        )
        return
    counts = Counter(source.tobytes())
    histogram = [counts[value] for value in range(256)]

    def histogram_only():
        result, actual = equalize(buffer, enum.HIST)
        return dict(
            status=(
                "passed"
                if not result.is_valid() and list(actual) == histogram
                else "failed"
            ),
            image_valid=result.is_valid(),
            actual=list(actual),
            expected=histogram,
        )

    report.run("equalize_hist_HIST_%dx%d" % (width, height), histogram_only)
    for name in ("EQHIST", "EQHIST_WITH_EXT_HIST"):
        mode = getattr(enum, name, None)
        if mode is None:
            report.missing("equalize_hist_" + name, "Expose EqualizeMode." + name)
            continue

        def equalized(name=name, mode=mode):
            result, returned_counts = equalize(
                buffer, mode, histogram if name == "EQHIST_WITH_EXT_HIST" else []
            )
            actual = array(result)
            running, cumulative = 0, []
            for count in histogram:
                running += count
                cumulative.append(running)
            first = next(value for value in cumulative if value)
            candidates = {}
            for label, subtract in [("cdf", 0), ("cdf_min", first)]:
                denominator = width * height - subtract
                for rule in ("floor", "nearest"):
                    table = [
                        (
                            0
                            if denominator == 0
                            else max(
                                0,
                                min(
                                    255,
                                    (
                                        (count - subtract) * 255
                                        + (denominator // 2 if rule == "nearest" else 0)
                                    )
                                    // denominator,
                                ),
                            )
                        )
                        for count in cumulative
                    ]
                    candidates[label + "_" + rule] = np.asarray(
                        [[table[value] for value in row] for row in source.tolist()],
                        dtype=np.uint8,
                    )
            return dict(
                status="unverified" if not returned_counts else "failed",
                candidates=crop_candidates(actual, candidates),
                returned_counts=list(returned_counts),
                reason="Determine CDF normalization and rounding; OpenCV subtracts the first nonzero CDF",
            )

        report.run("equalize_hist_%s_%dx%d" % (name, width, height), equalized)
    report.rejected(
        "equalize_hist_bad_external_total_%dx%d" % (width, height),
        lambda: equalize(buffer, enum.EQHIST_WITH_EXT_HIST, [0] * 256),
    )


def main():
    groups = dict(
        threshold=check_threshold,
        cast=check_cast,
        integral=check_integral,
        dma=check_dma,
        sad=check_sad,
        lbp=check_lbp,
        gradient=check_gradient,
        gradient_quantization=check_gradient_quantization,
        ccl=check_ccl,
        rank=check_rank_morph,
        extended=check_extended,
        border=check_border_contract,
        arithmetic=check_arithmetic,
        equalize_degenerate=check_equalize_degenerate,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_modes.json"))
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--sizes", default="320x240,322x242,640x480")
    parser.add_argument("--groups", default=",".join(groups))
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check contract references on the host without OpenCV, NumPy or VisionG",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Also fail when any convention or mode remains unverified",
    )
    args = parser.parse_args()
    if args.self_test:
        contract_reference_self_test()
        return
    if args.loops < 1:
        parser.error("--loops must be positive")
    try:
        sizes = [
            tuple(int(part) for part in value.split("x"))
            for value in args.sizes.split(",")
        ]
        if any(len(size) != 2 or min(size) < 192 for size in sizes):
            raise ValueError
    except ValueError:
        parser.error("--sizes must contain WxH values with both dimensions >=192")
    selected = args.groups.split(",")
    if any(name not in groups for name in selected):
        parser.error("Unknown group; choose from " + ",".join(groups))
    global cv2, np, vg, GRADIENT_MASK
    import cv2
    import numpy as np
    import visiong as vg

    GRADIENT_MASK = np.asarray(GRADIENT_MASK, dtype=np.float32)
    cv2.setNumThreads(1)
    report = Report(args.output, args.loops)
    for width, height in sizes:
        for name in selected:
            groups[name](report, width, height)
    counts = Counter(row["status"] for row in report.rows)
    print(
        json.dumps(
            dict(
                summary=dict(counts),
                complete=counts["failed"] == 0 and counts["unverified"] == 0,
            )
        ),
        flush=True,
    )
    if counts["failed"] or (args.require_complete and counts["unverified"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
