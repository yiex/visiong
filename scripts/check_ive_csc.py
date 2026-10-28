# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check all IVE CSC modes, compact layouts and range controls.
检查全部 IVE CSC 模式、紧凑数据布局及色域控制。

Independent numeric references and integration checks are reported separately.
Conventional floating-point comparisons remain historical observations.
独立数值参考与集成检查分别记录，常规浮点比较保留为历史观察。
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from ive_hsv_reference import rgb_pixel_to_ive_hsv


COLOR_LAYOUTS = ("U8C3_PACKAGE", "U8C3_PLANAR")
YUV_LAYOUTS = ("YUV420SP", "YUV422SP")
MODES = tuple(
    family + "_" + standard + "_" + output_range
    for family in ("YUV2RGB", "YUV2HSV", "RGB2YUV", "RGB2HSV")
    for standard in ("BT601", "BT709")
    for output_range in ("LIMITED", "FULL")
)
FULL_Q8 = {
    "BT601": ((298, 0, 409), (298, -100, -208), (298, 516, 0)),
    "BT709": ((298, 0, 459), (298, -55, -136), (298, 541, 0)),
}
LIMITED_Q8 = {
    "BT601": ((256, 0, 351), (256, -86, -179), (256, 444, 0)),
    "BT709": ((256, 0, 394), (256, -47, -117), (256, 464, 0)),
}
RGB_FULL_Q8 = {
    "BT601": ((77, 150, 29), (-43, -85, 128), (128, -107, -21)),
    "BT709": ((54, 183, 18), (-29, -99, 128), (128, -116, -12)),
}
RGB_LIMITED_Q8 = {
    "BT601": ((66, 129, 25), (-38, -75, 112), (112, -94, -18)),
    "BT709": ((47, 157, 16), (-26, -87, 112), (112, -102, -10)),
}


def interrupts():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def compact_size(layout, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    pixels = width * height
    if layout == "YUV420SP":
        if width & 1 or height & 1:
            raise ValueError("420 requires even width and height")
        return pixels * 3 // 2
    if layout == "YUV422SP":
        if width & 1:
            raise ValueError("422 requires even width")
        return pixels * 2
    if layout in COLOR_LAYOUTS:
        return pixels * 3
    raise ValueError("Unsupported compact layout")


def encode_color(rgb, width, height, layout, reverse=False):
    """Pack channel bytes independently of hardware row strides.
    按通道打包字节，不依赖硬件行步长。
    """
    if len(rgb) != width * height * 3 or layout not in COLOR_LAYOUTS:
        raise ValueError("Invalid compact color data")
    channels = (
        (rgb[2::3], rgb[1::3], rgb[0::3])
        if reverse
        else (rgb[0::3], rgb[1::3], rgb[2::3])
    )
    if layout == "U8C3_PLANAR":
        return b"".join(channels)
    result = bytearray(len(rgb))
    result[0::3], result[1::3], result[2::3] = channels
    return bytes(result)


def decode_color(data, width, height, layout, reverse=False):
    if len(data) != width * height * 3 or layout not in COLOR_LAYOUTS:
        raise ValueError("Invalid compact color data")
    pixels = width * height
    if layout == "U8C3_PLANAR":
        channels = (data[:pixels], data[pixels : pixels * 2], data[pixels * 2 :])
    else:
        channels = (data[0::3], data[1::3], data[2::3])
    if reverse:
        channels = channels[::-1]
    result = bytearray(len(data))
    result[0::3], result[1::3], result[2::3] = channels
    return bytes(result)


def yuv_order(data, width, height, layout, vu=False):
    """Swap only interleaved chroma pairs; this operation is its own inverse.
    仅交换交织色度对，此操作也是其逆变换。
    """
    if len(data) != compact_size(layout, width, height):
        raise ValueError("Invalid compact YUV data")
    if not vu:
        return bytes(data)
    pixels = width * height
    result = bytearray(data)
    result[pixels::2], result[pixels + 1 :: 2] = data[pixels + 1 :: 2], data[pixels::2]
    return bytes(result)


def repeat_color_rows(data, width, height):
    stride = width * 3
    return b"".join(data[y * stride : (y + 1) * stride] * 2 for y in range(height))


def select_color_rows(data, width, height):
    stride = width * 3
    if len(data) != height * 2 * stride:
        raise ValueError("Expected a color image with duplicated rows")
    return b"".join(data[2 * y * stride : (2 * y + 1) * stride] for y in range(height))


def expand_422_to_420(data, width, height):
    """Duplicate luma rows while retaining every original 422 chroma row.
    重复亮度行，同时保留 422 的每一行原始色度。
    """
    if len(data) != compact_size("YUV422SP", width, height):
        raise ValueError("Invalid 422 image")
    pixels = width * height
    luma = b"".join(data[y * width : (y + 1) * width] * 2 for y in range(height))
    return luma + data[pixels:]


def extract_422_from_420(data, width, height):
    if len(data) != compact_size("YUV420SP", width, height * 2):
        raise ValueError("Expected a 420 image with twice the requested height")
    luma = b"".join(data[2 * y * width : (2 * y + 1) * width] for y in range(height))
    return luma + data[width * height * 2 :]


def rgb_pattern(width, height):
    return bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (
            (17 * x + 29 * y + x * y % 31) % 256,
            (43 * x + 7 * y + 53) % 256,
            (11 * x + 61 * y + 197) % 256,
        )
    )


def yuv_pattern(width, height, layout):
    luma = bytes(
        (17 * x + 29 * y + x * y % 19) % 256
        for y in range(height)
        for x in range(width)
    )
    rows = height // 2 if layout == "YUV420SP" else height
    chroma = bytes(
        channel
        for y in range(rows)
        for pair in range(width // 2)
        for channel in (
            (37 * pair + 53 * y + 13) % 256,
            (19 * pair + 97 * y + 211) % 256,
        )
    )
    return luma + chroma


def yuv_pixel_q8(y, u, v, standard, input_full_range=False, output_limited=False):
    """Use sampled-capture Q8 matrices and explicit input-range mapping.
    使用已采集样本验证的 Q8 矩阵及显式输入色域映射。
    """
    if input_full_range:
        y = 16 + (219 * y + 127) // 255
        u = 16 + (224 * u + 127) // 255
        v = 16 + (224 * v + 127) // 255
    source = (y if output_limited else y - 16, u - 128, v - 128)
    coefficients = LIMITED_Q8 if output_limited else FULL_Q8
    minimum, maximum = (16, 235) if output_limited else (0, 255)
    return tuple(
        max(minimum, min(maximum, (sum(a * b for a, b in zip(row, source)) + 128) >> 8))
        for row in coefficients[standard]
    )


def yuv_reference(
    data,
    width,
    height,
    layout,
    standard,
    hsv=False,
    input_full_range=False,
    output_limited=False,
):
    pixels = width * height
    result = bytearray(pixels * 3)
    for y in range(height):
        chroma_row = y // 2 if layout == "YUV420SP" else y
        for x in range(width):
            chroma = pixels + chroma_row * width + (x // 2) * 2
            rgb = yuv_pixel_q8(
                data[y * width + x],
                data[chroma],
                data[chroma + 1],
                standard,
                input_full_range,
                output_limited,
            )
            channels = rgb_pixel_to_ive_hsv(*rgb) if hsv else rgb
            offset = (y * width + x) * 3
            result[offset : offset + 3] = bytes(channels)
    return bytes(result)


def hsv_reference(rgb):
    return bytes(
        channel
        for offset in range(0, len(rgb), 3)
        for channel in rgb_pixel_to_ive_hsv(*rgb[offset : offset + 3])
    )


def rgb_pixel_yuv_q8(red, green, blue, standard, output_limited=False):
    """Express captured RGB-to-YUV behavior as an integer compatibility model.
    以整数兼容模型表达已采集的 RGB 转 YUV 行为。

    The Q8 representation does not establish the hardware's internal precision.
    Q8 表达形式不表示硬件内部精度已被确定。
    """
    coefficients = RGB_LIMITED_Q8 if output_limited else RGB_FULL_Q8
    minimum = (16, 16, 16) if output_limited else (0, 0, 0)
    maximum = (235, 240, 240) if output_limited else (255, 255, 255)
    bias = (16 if output_limited else 0, 128, 128)
    source = (red, green, blue)
    return tuple(
        max(
            low,
            min(high, ((sum(a * b for a, b in zip(row, source)) + 128) >> 8) + offset),
        )
        for row, offset, low, high in zip(
            coefficients[standard], bias, minimum, maximum
        )
    )


def rgb_yuv_reference(rgb, width, height, layout, standard, output_limited=False):
    """Keep pixel Y and top-left chroma: [2y,2x] for 420, [y,2x] for 422.
    保留逐像素 Y；420 色度取 [2y,2x]，422 色度取 [y,2x]。
    """
    if layout not in YUV_LAYOUTS or len(rgb) != width * height * 3:
        raise ValueError("Invalid compact RGB-to-YUV reference input")
    compact_size(layout, width, height)
    pixels = width * height
    vertical = 2 if layout == "YUV420SP" else 1
    luma, chroma = bytearray(pixels), bytearray(pixels // vertical)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 3
            values = rgb_pixel_yuv_q8(
                *rgb[offset : offset + 3], standard, output_limited
            )
            luma[y * width + x] = values[0]
            if y % vertical == 0 and x % 2 == 0:
                position = (y // vertical) * width + x
                chroma[position : position + 2] = bytes(values[1:])
    return bytes(luma) + bytes(chroma)


def nominal_rgb_to_yuv(rgb, width, height, layout, standard, limited):
    """Provide a conventional floating-point observation, never an acceptance.
    提供常规浮点模型作观察，不作为验收真值。
    """
    kr, kb = (0.299, 0.114) if standard == "BT601" else (0.2126, 0.0722)
    values = []
    for offset in range(0, len(rgb), 3):
        red, green, blue = rgb[offset : offset + 3]
        y = kr * red + (1 - kr - kb) * green + kb * blue
        u, v = (blue - y) / (2 * (1 - kb)), (red - y) / (2 * (1 - kr))
        values.append(
            (
                16 + y * 219 / 255 if limited else y,
                128 + u * (224 / 255 if limited else 1),
                128 + v * (224 / 255 if limited else 1),
            )
        )
    clamp = lambda value: max(0, min(255, int(math.floor(value + 0.5))))
    luma = bytes(clamp(row[0]) for row in values)
    vertical = 2 if layout == "YUV420SP" else 1
    chroma = bytearray()
    for top in range(0, height, vertical):
        for left in range(0, width, 2):
            indexes = [
                y * width + x
                for y in range(top, top + vertical)
                for x in range(left, left + 2)
            ]
            chroma.extend(
                clamp(sum(values[index][channel] for index in indexes) / len(indexes))
                for channel in (1, 2)
            )
    return luma + bytes(chroma)


def byte_difference(actual, expected):
    if actual == expected:
        return dict(
            actual_bytes=len(actual),
            expected_bytes=len(expected),
            different_bytes=0,
            maximum_difference=0,
            examples=[],
        )
    examples = []
    maximum, count = 0, 0
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            count += 1
            maximum = max(maximum, abs(a - b))
            if len(examples) < 8:
                examples.append(dict(offset=index, actual=a, expected=b))
    return dict(
        actual_bytes=len(actual),
        expected_bytes=len(expected),
        different_bytes=count + abs(len(actual) - len(expected)),
        maximum_difference=maximum,
        examples=examples,
    )


def result_counts(rows):
    """Keep historical comparisons separate from unresolved capability checks.
    区分历史比较记录与尚未确定的功能检查。
    """
    return (
        Counter(
            row["status"]
            for row in rows
            if row.get("evidence_role") != "historical_comparison"
        ),
        Counter(
            row["status"]
            for row in rows
            if row.get("evidence_role") == "historical_comparison"
        ),
    )


class Checks:
    def __init__(self, vg, output, loops):
        self.vg, self.processor = vg, vg.IVE()
        self.output, self.loops = output, loops
        self.results = []

    def emit(self, row):
        self.results.append(row)
        capabilities, historical = result_counts(self.results)
        document = dict(
            schema=1,
            counts=dict(Counter(item["status"] for item in self.results)),
            capability_counts=dict(capabilities),
            historical_comparison_counts=dict(historical),
            results=self.results,
        )
        self.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    def raw(
        self,
        data,
        width,
        height,
        source,
        destination,
        mode,
        full=False,
        bgr=False,
        vu=False,
    ):
        return bytes(
            self.processor.csc_bytes(
                data,
                width,
                height,
                getattr(self.vg.ImageTypeIVE, source),
                getattr(self.vg.ImageTypeIVE, destination),
                getattr(self.vg.CscMode, mode),
                full,
                bgr,
                vu,
            )
        )

    def check(self, name, category, call, expected, metadata=None, observational=False):
        row = dict(name=name, category=category, **(metadata or {}))
        signatures, durations, irqs = [], [], []
        comparisons = []
        try:
            for _ in range(self.loops):
                before = interrupts()
                start = time.perf_counter()
                actual = bytes(call())
                durations.append((time.perf_counter() - start) * 1000)
                after = interrupts()
                irqs.append(
                    after - before if after is not None and before is not None else None
                )
                signatures.append(hashlib.sha256(actual).hexdigest())
                comparisons.append(byte_difference(actual, expected))
            exact = all(item["different_bytes"] == 0 for item in comparisons)
            stable = len(set(signatures)) == 1
            valid_size = all(
                item["actual_bytes"] == item["expected_bytes"] for item in comparisons
            )
            status = (
                "unverified" if observational else ("passed" if exact else "failed")
            )
            if (
                not stable
                or not valid_size
                or any(value is not None and value <= 0 for value in irqs)
            ):
                status = "failed"
            elif any(value is None for value in irqs) and status == "passed":
                status = "unverified"
            row.update(
                status=status,
                comparisons=comparisons,
                unique_outputs=len(set(signatures)),
                sha256=signatures[0],
                interrupts_per_call=irqs,
                min_ms=min(durations),
            )
        except Exception as error:
            row.update(status="failed", error=str(error))
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
        row.update(name=name, category="rejection")
        self.emit(row)


def check_size(checks, width, height, quick=False, integration=True):
    rgb = rgb_pattern(width, height)
    numeric_references = {}
    for mode in MODES:
        family, standard, output_range = mode.split("_")
        yuv_input, yuv_output = family.startswith("YUV"), family.endswith("YUV")
        sources = YUV_LAYOUTS if yuv_input else COLOR_LAYOUTS
        destinations = YUV_LAYOUTS if yuv_output else COLOR_LAYOUTS
        for source in sources:
            for destination in destinations:
                try:
                    compact_size(source, width, height)
                    compact_size(destination, width, height)
                except ValueError:
                    continue
                canonical = yuv_pattern(width, height, source) if yuv_input else rgb
                for bgr in (False,) if quick or family == "YUV2HSV" else (False, True):
                    for vu in (
                        (False,)
                        if quick or not (yuv_input or yuv_output)
                        else (False, True)
                    ):
                        for full in (
                            (False, True) if yuv_input and not quick else (False,)
                        ):
                            source_bytes = (
                                yuv_order(canonical, width, height, source, vu)
                                if yuv_input
                                else encode_color(rgb, width, height, source, bgr)
                            )
                            if family == "RGB2HSV":
                                key = ("rgb_hsv",)
                                if key not in numeric_references:
                                    numeric_references[key] = hsv_reference(rgb)
                                reference = numeric_references[key]
                                expected = encode_color(
                                    reference, width, height, destination
                                )
                                category = "independent_math"
                                note = "Exact integer RGB-to-HSV model, including signed hue numerator and reciprocal quantization"
                            elif yuv_input:
                                limited = output_range == "LIMITED"
                                key = (
                                    "yuv_q8",
                                    source,
                                    standard,
                                    family,
                                    full,
                                    limited,
                                )
                                if key not in numeric_references:
                                    numeric_references[key] = yuv_reference(
                                        canonical,
                                        width,
                                        height,
                                        source,
                                        standard,
                                        family == "YUV2HSV",
                                        full,
                                        limited,
                                    )
                                reference = numeric_references[key]
                                expected = encode_color(
                                    reference,
                                    width,
                                    height,
                                    destination,
                                    bgr and family == "YUV2RGB",
                                )
                                category = "independent_math"
                                note = "Q8 YUV-to-RGB and direct YUV-to-HSV models match all 16,777,216 input triples per mode in eight independently hash-verified capture streams; RGB-to-HSV uses its exhaustive integer model"
                            else:
                                key = (
                                    "rgb_yuv_q8",
                                    destination,
                                    standard,
                                    output_range,
                                )
                                if key not in numeric_references:
                                    numeric_references[key] = rgb_yuv_reference(
                                        rgb,
                                        width,
                                        height,
                                        destination,
                                        standard,
                                        output_range == "LIMITED",
                                    )
                                reference = numeric_references[key]
                                expected = yuv_order(
                                    reference, width, height, destination, vu
                                )
                                category = "independent_math"
                                note = "Integer RGB-to-YUV compatibility model matches all 16,777,216 RGB colors per mode in independently hash-verified tiled capture streams; eight spatial patterns match top-left chroma sampling. Q8 notation does not establish internal hardware precision"
                            metadata = dict(
                                mode=mode,
                                size=[width, height],
                                source_layout=source,
                                destination_layout=destination,
                                input_full_range=full,
                                bgr_order=bgr,
                                vu_order=vu,
                                reference_note=note,
                            )
                            name = "%s_%s_to_%s_%dx%d_bgr%d_vu%d_inputfull%d" % (
                                mode,
                                source,
                                destination,
                                width,
                                height,
                                bgr,
                                vu,
                                full,
                            )
                            if family == "RGB2YUV":
                                name += "_q8_contract"
                            checks.check(
                                name,
                                category,
                                lambda source_bytes=source_bytes, source=source, destination=destination, mode=mode, full=full, bgr=bgr, vu=vu: checks.raw(
                                    source_bytes,
                                    width,
                                    height,
                                    source,
                                    destination,
                                    mode,
                                    full,
                                    bgr,
                                    vu,
                                ),
                                expected,
                                metadata,
                            )
                if family == "RGB2YUV" and source == COLOR_LAYOUTS[0]:
                    expected = nominal_rgb_to_yuv(
                        rgb,
                        width,
                        height,
                        destination,
                        standard,
                        output_range == "LIMITED",
                    )
                    checks.check(
                        "%s_%s_matrix_observation_%dx%d"
                        % (mode, destination, width, height),
                        "observation",
                        lambda destination=destination, mode=mode: checks.raw(
                            rgb, width, height, COLOR_LAYOUTS[0], destination, mode
                        ),
                        expected,
                        dict(
                            evidence_role="historical_comparison",
                            reference_note="Historical conventional floating-point matrix and box-averaged chroma comparison; retained as an observation, separately from the verified integer and top-left sampling contract",
                        ),
                        observational=True,
                    )
        if integration and not (width & 1 or height & 1):
            check_legacy_integration(checks, rgb, width, height, mode)


def check_legacy_integration(checks, rgb, width, height, mode):
    yuv_input, yuv_output = mode.startswith("YUV"), mode.startswith("RGB2YUV")
    source = yuv_pattern(width, height, "YUV420SP") if yuv_input else rgb
    try:
        frame = checks.vg.ImageBuffer.from_bytes(
            source, width, height, "nv12" if yuv_input else "rgb888"
        )
        result = checks.processor.csc(frame, getattr(checks.vg.CscMode, mode))
        expected = bytes(result.data)
    except Exception as error:
        checks.emit(
            dict(
                name="legacy_setup_" + mode,
                category="integration_check",
                status="failed",
                error=str(error),
            )
        )
        return
    checks.check(
        "legacy_integration_%s_%dx%d" % (mode, width, height),
        "integration_check",
        lambda: checks.raw(
            source,
            width,
            height,
            "YUV420SP" if yuv_input else "U8C3_PACKAGE",
            "YUV420SP" if yuv_output else "U8C3_PACKAGE",
            mode,
        ),
        expected,
        dict(
            reference_note="Equality to ImageBuffer.csc checks legacy integration; independent numeric acceptance is recorded separately"
        ),
    )


def check_invalid(checks):
    width, height = 322, 242
    mode = "YUV2RGB_BT601_FULL"
    valid = yuv_pattern(width, height, "YUV420SP")
    for suffix, data in [("short", valid[:-1]), ("long", valid + b"\0")]:
        checks.reject(
            "byte_count_" + suffix,
            lambda data=data: checks.raw(
                data, width, height, "YUV420SP", "U8C3_PACKAGE", mode
            ),
        )
    for w, h in [
        (0, 242),
        (-1, 242),
        (322, 0),
        (322, -1),
        (4097, 242),
        (322, 4097),
        (321, 242),
        (322, 241),
    ]:
        checks.reject(
            "invalid_420_dimensions_%d_%d" % (w, h),
            lambda w=w, h=h: checks.raw(b"", w, h, "YUV420SP", "U8C3_PACKAGE", mode),
        )
    checks.reject(
        "invalid_422_odd_width",
        lambda: checks.raw(b"", 321, 241, "YUV422SP", "U8C3_PACKAGE", mode),
    )
    checks.reject(
        "invalid_422_height_range",
        lambda: checks.raw(
            bytes(322 * 4097 * 2), 322, 4097, "YUV422SP", "U8C3_PACKAGE", mode
        ),
    )
    checks.reject(
        "invalid_source_layout",
        lambda: checks.raw(valid, width, height, "U8C1", "U8C3_PACKAGE", mode),
    )
    checks.reject(
        "invalid_direction",
        lambda: checks.raw(valid, width, height, "YUV420SP", "YUV420SP", mode),
    )
    color = rgb_pattern(width, height)
    checks.reject(
        "invalid_input_range_on_rgb",
        lambda: checks.raw(
            color,
            width,
            height,
            "U8C3_PACKAGE",
            "U8C3_PACKAGE",
            "RGB2HSV_BT601_FULL",
            True,
        ),
    )
    checks.reject(
        "invalid_vu_on_rgb_hsv",
        lambda: checks.raw(
            color,
            width,
            height,
            "U8C3_PACKAGE",
            "U8C3_PACKAGE",
            "RGB2HSV_BT601_FULL",
            False,
            False,
            True,
        ),
    )
    checks.reject(
        "invalid_bgr_on_yuv_hsv",
        lambda: checks.raw(
            valid,
            width,
            height,
            "YUV420SP",
            "U8C3_PACKAGE",
            "YUV2HSV_BT601_FULL",
            False,
            True,
        ),
    )


def self_check(capture_root=None):
    rgb = bytes((11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 122, 133))
    planar = bytes((11, 44, 77, 111, 22, 55, 88, 122, 33, 66, 99, 133))
    assert encode_color(rgb, 2, 2, "U8C3_PLANAR") == planar
    assert decode_color(planar, 2, 2, "U8C3_PLANAR") == rgb
    assert encode_color(rgb, 2, 2, "U8C3_PACKAGE", True) == bytes(
        (33, 22, 11, 66, 55, 44, 99, 88, 77, 133, 122, 111)
    )
    for layout in COLOR_LAYOUTS:
        for reverse in (False, True):
            assert (
                decode_color(
                    encode_color(rgb, 2, 2, layout, reverse), 2, 2, layout, reverse
                )
                == rgb
            )
    yuv = bytes((1, 2, 3, 4, 5, 6))
    assert yuv_order(yuv, 2, 2, "YUV420SP", True) == bytes((1, 2, 3, 4, 6, 5))
    for width, height in [(4, 3), (322, 33), (322, 241)]:
        source = yuv_pattern(width, height, "YUV422SP")
        assert (
            extract_422_from_420(
                expand_422_to_420(source, width, height), width, height
            )
            == source
        )
        assert (
            yuv_order(
                yuv_order(source, width, height, "YUV422SP", True),
                width,
                height,
                "YUV422SP",
                True,
            )
            == source
        )
        pixels = width * height
        assert (
            source[pixels : pixels + width]
            != source[pixels + width : pixels + width * 2]
        )
        colors = rgb_pattern(width, height)
        assert (
            select_color_rows(repeat_color_rows(colors, width, height), width, height)
            == colors
        )
        for standard in FULL_Q8:
            for limited in (False, True):
                independent = yuv_reference(
                    source,
                    width,
                    height,
                    "YUV422SP",
                    standard,
                    output_limited=limited,
                )
                expanded = yuv_reference(
                    expand_422_to_420(source, width, height),
                    width,
                    height * 2,
                    "YUV420SP",
                    standard,
                    output_limited=limited,
                )
                assert select_color_rows(expanded, width, height) == independent
    for standard in FULL_Q8:
        assert yuv_pixel_q8(16, 128, 128, standard) == (0, 0, 0)
        assert yuv_pixel_q8(235, 128, 128, standard) == (255, 255, 255)
        assert yuv_pixel_q8(0, 128, 128, standard, True) == (0, 0, 0)
        for y in (0, 16, 128, 235, 255):
            assert (
                yuv_pixel_q8(y, 128, 128, standard, output_limited=True)
                == (max(16, min(235, y)),) * 3
            )
    assert yuv_pixel_q8(128, 128, 192, "BT601", output_limited=True) == (216, 83, 128)
    assert yuv_pixel_q8(128, 128, 192, "BT709", output_limited=True) == (227, 99, 128)
    assert rgb_pixel_to_ive_hsv(255, 0, 0) == (0, 255, 255)
    assert rgb_pixel_to_ive_hsv(0, 255, 0) == (86, 255, 255)
    assert rgb_pixel_to_ive_hsv(0, 0, 255) == (171, 255, 255)
    primary = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))
    assert rgb_yuv_reference(primary, 2, 2, "YUV420SP", "BT601", True) == bytes(
        (82, 144, 41, 235, 90, 240)
    )
    assert rgb_yuv_reference(primary, 2, 2, "YUV420SP", "BT709") == bytes(
        (54, 182, 18, 254, 99, 255)
    )
    odd_rows = primary + bytes((255, 255, 0, 0, 0, 0))
    assert rgb_yuv_reference(odd_rows, 2, 3, "YUV422SP", "BT601", True) == bytes(
        (82, 144, 41, 235, 210, 16, 90, 240, 240, 110, 16, 146)
    )
    assert rgb_pixel_yuv_q8(0, 51, 204, "BT601", True) == (62, 202, 95)
    assert rgb_pixel_yuv_q8(255, 255, 255, "BT601", True) == (235, 127, 128)
    assert rgb_pixel_yuv_q8(255, 255, 255, "BT709") == (254, 128, 128)
    for width, height in ((4, 3), (6, 5)):
        colors = rgb_pattern(width, height)
        for standard in RGB_FULL_Q8:
            for limited in (False, True):
                expected = rgb_yuv_reference(
                    colors, width, height, "YUV422SP", standard, limited
                )
                doubled = rgb_yuv_reference(
                    repeat_color_rows(colors, width, height),
                    width,
                    height * 2,
                    "YUV420SP",
                    standard,
                    limited,
                )
                assert extract_422_from_420(doubled, width, height) == expected
    results = dict(
        status="passed",
        reference_checks=[
            "hand-specified packed/planar/BGR bytes",
            "UV/VU inverse",
            "odd-height 422 row preservation",
            "422 versus independently expanded 420 Q8 results",
            "Q8 neutral endpoints",
            "integer HSV primary colors",
            "integer RGB-to-YUV coefficients, rounding and limited clipping",
            "420 top-left and per-row 422 chroma with odd heights",
        ],
    )
    if capture_root is not None:
        results["captures"] = check_captures(capture_root)
    return results


def check_captures(root):
    metadata_path = root / "capture.json"
    if not metadata_path.exists():
        metadata_path = root / "yuv_capture.json"
    metadata = json.loads(metadata_path.read_text())
    results = []
    for standard, limited in (
        (standard, limited) for standard in FULL_Q8 for limited in (False, True)
    ):
        name = "YUV2RGB_" + standard + ("_LIMITED.bin" if limited else "_FULL.bin")
        captured = (root / name).read_bytes()
        expected_size = len(metadata["v_layers"]) * 256 * 256 * 3
        if len(captured) != expected_size:
            raise ValueError("Unexpected capture size for " + name)
        digest = hashlib.sha256(captured).hexdigest()
        if metadata["files"][name] != digest:
            raise ValueError("Capture hash disagrees with metadata for " + name)
        mismatches, maximum = 0, 0
        for layer, v in enumerate(metadata["v_layers"]):
            for y in range(256):
                for u in range(256):
                    offset = ((layer * 256 + y) * 256 + u) * 3
                    predicted = yuv_pixel_q8(y, u, v, standard, output_limited=limited)
                    for actual, expected in zip(
                        captured[offset : offset + 3], predicted
                    ):
                        mismatches += actual != expected
                        maximum = max(maximum, abs(actual - expected))
        results.append(
            dict(
                file=name,
                pixels=expected_size // 3,
                different_bytes=mismatches,
                maximum_difference=maximum,
                sha256=digest,
            )
        )
        if mismatches:
            raise AssertionError("Q8 reference disagrees with " + name)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_csc.json"))
    parser.add_argument("--sizes", default="320x240,322x242,322x241,322x33,322x3")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use canonical channel/range flags while retaining every valid mode/layout pair",
    )
    parser.add_argument("--skip-integration", action="store_true")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run software-only reference checks without importing VisionG",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        help="Optionally verify full/limited Q8 references against captured RGB files",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail for unresolved capabilities; historical conventional comparisons are excluded",
    )
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be positive")
    if args.self_check:
        result = self_check(args.capture_root)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return
    try:
        sizes = [
            tuple(int(part) for part in size.split("x"))
            for size in args.sizes.split(",")
        ]
        if any(
            len(size) != 2 or not 2 <= size[0] <= 4096 or not 2 <= size[1] <= 4096
            for size in sizes
        ):
            raise ValueError
    except ValueError:
        parser.error("--sizes requires width and height in [2,4096]")
    import visiong as vg

    if not hasattr(vg.IVE, "csc_bytes"):
        raise SystemExit("VisionG does not expose csc_bytes")
    checks = Checks(vg, args.output, args.loops)
    for width, height in sizes:
        check_size(checks, width, height, args.quick, not args.skip_integration)
    check_invalid(checks)
    counts = Counter(row["status"] for row in checks.results)
    capability_counts, historical_counts = result_counts(checks.results)
    print(
        json.dumps(
            dict(
                summary=dict(counts),
                capability_summary=dict(capability_counts),
                historical_comparison_summary=dict(historical_counts),
                capabilities_complete=not capability_counts["failed"]
                and not capability_counts["unverified"],
                complete=not counts["failed"] and not counts["unverified"],
            )
        ),
        flush=True,
    )
    if counts["failed"] or (args.require_complete and capability_counts["unverified"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
