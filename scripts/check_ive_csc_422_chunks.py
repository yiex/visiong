# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check tall 422 CSC images against independent integer references.
使用独立整数参考检查高图像的 422 CSC 分块转换。

The scalar Q8 and HSV contracts express observed compatibility, not circuit precision.
单像素 Q8 与 HSV 契约表达已观察到的兼容行为，不代表电路内部精度。
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time


HEIGHTS = (2047, 2048, 2049, 4095, 4096)
ALLOWED_HEIGHTS = HEIGHTS + (511, 1023, 1024, 1025)
BLOCK_ROWS = 1023
WIDTHS = (64, 66)
LAYOUTS = ("U8C3_PACKAGE", "U8C3_PLANAR")
MODES = tuple(
    family + "_" + standard + "_" + output_range
    for family in ("YUV2RGB", "YUV2HSV", "RGB2YUV")
    for standard in ("BT601", "BT709")
    for output_range in ("LIMITED", "FULL")
)
YUV_FULL_Q8 = {
    "BT601": ((298, 0, 409), (298, -100, -208), (298, 516, 0)),
    "BT709": ((298, 0, 459), (298, -55, -136), (298, 541, 0)),
}
YUV_LIMITED_Q8 = {
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
RGB_PALETTE = (
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 51, 204),
    (1, 2, 3),
    (254, 253, 252),
    (255, 1, 127),
    (17, 235, 16),
    (77, 150, 29),
    (127, 128, 129),
    (9, 250, 247),
    (247, 9, 250),
    (250, 247, 9),
    (31, 63, 191),
)
Y_PALETTE = (0, 1, 15, 16, 17, 31, 63, 95, 127, 128, 129, 191, 234, 235, 240, 255)
UV_PALETTE = (
    (0, 255),
    (255, 0),
    (16, 240),
    (240, 16),
    (127, 128),
    (128, 127),
    (128, 128),
    (129, 129),
    (31, 211),
    (211, 31),
    (53, 197),
    (197, 53),
    (1, 254),
    (254, 1),
    (95, 161),
    (161, 95),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hasher.update(block)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=hasher.hexdigest())


def interrupts():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def hsv_pixel(rgb):
    red, green, blue = rgb
    value, minimum = max(rgb), min(rgb)
    delta = value - minimum
    if not delta:
        return (0, 0, value)
    numerator = (
        green - blue
        if value == red
        else (blue - red + 2 * delta if value == green else red - green + 4 * delta)
    )
    saturation_reciprocal = ((255 << 8) + value // 2) // value
    hue_reciprocal = ((256 << 8) + 3 * delta) // (6 * delta)
    return (
        ((numerator * hue_reciprocal + 128) >> 8) & 255,
        (delta * saturation_reciprocal + 128) >> 8,
        value,
    )


def yuv_pixel(y, uv, standard, full_input, limited, hsv):
    u, v = uv
    if full_input:
        y = 16 + (219 * y + 127) // 255
        u, v = (16 + (224 * value + 127) // 255 for value in (u, v))
    values = (y if limited else y - 16, u - 128, v - 128)
    rows = (YUV_LIMITED_Q8 if limited else YUV_FULL_Q8)[standard]
    low, high = (16, 235) if limited else (0, 255)
    rgb = tuple(
        max(low, min(high, (sum(a * b for a, b in zip(row, values)) + 128) >> 8))
        for row in rows
    )
    return hsv_pixel(rgb) if hsv else rgb


def rgb_pixel(rgb, standard, limited):
    rows = (RGB_LIMITED_Q8 if limited else RGB_FULL_Q8)[standard]
    bias = (16 if limited else 0, 128, 128)
    lows, highs = (
        ((16, 16, 16), (235, 240, 240)) if limited else ((0, 0, 0), (255, 255, 255))
    )
    return tuple(
        max(
            low, min(high, ((sum(a * b for a, b in zip(row, rgb)) + 128) >> 8) + offset)
        )
        for row, offset, low, high in zip(rows, bias, lows, highs)
    )


def color_order(data, width, height, layout, bgr=False):
    if len(data) != width * height * 3 or layout not in LAYOUTS:
        raise ValueError("Invalid compact color layout")
    channels = (
        (data[2::3], data[1::3], data[0::3])
        if bgr
        else (data[0::3], data[1::3], data[2::3])
    )
    if layout == "U8C3_PLANAR":
        return b"".join(channels)
    output = bytearray(len(data))
    output[0::3], output[1::3], output[2::3] = channels
    return bytes(output)


def chroma_order(data, width, height, vu):
    if len(data) != width * height * 2:
        raise ValueError("Invalid compact 422 layout")
    if not vu:
        return data
    pixels = width * height
    output = bytearray(data)
    output[pixels::2], output[pixels + 1 :: 2] = data[pixels + 1 :: 2], data[pixels::2]
    return bytes(output)


def row_bases(height):
    # Non-power-of-two counters expose an incorrect row restart between blocks.
    # 非二次幂计数用于暴露块间发生的错误行相位重置。
    return (
        tuple((11 * y + y // 251) % 16 for y in range(height)),
        tuple((5 * y + y // 257) % 16 for y in range(height)),
    )


def patterns(width, height):
    color_bases, uv_bases = row_bases(height)
    colors = tuple(
        bytes(
            channel
            for x in range(width)
            for channel in RGB_PALETTE[(7 * x + base) % 16]
        )
        for base in range(16)
    )
    luma = tuple(
        bytes(Y_PALETTE[(7 * x + base) % 16] for x in range(width))
        for base in range(16)
    )
    chroma = tuple(
        bytes(
            channel
            for x in range(width // 2)
            for channel in UV_PALETTE[(3 * x + base) % 16]
        )
        for base in range(16)
    )
    rgb = b"".join(colors[base] for base in color_bases)
    yuv = b"".join(luma[base] for base in color_bases) + b"".join(
        chroma[base] for base in uv_bases
    )
    return rgb, yuv


def reference(width, height, mode, full_input):
    family, standard, output_range = mode.split("_")
    limited = output_range == "LIMITED"
    color_bases, uv_bases = row_bases(height)
    if family == "RGB2YUV":
        table = tuple(rgb_pixel(rgb, standard, limited) for rgb in RGB_PALETTE)
        luma = tuple(
            bytes(table[(7 * x + base) % 16][0] for x in range(width))
            for base in range(16)
        )
        chroma = tuple(
            bytes(
                channel
                for x in range(0, width, 2)
                for channel in table[(7 * x + base) % 16][1:]
            )
            for base in range(16)
        )
        return b"".join(luma[base] for base in color_bases) + b"".join(
            chroma[base] for base in color_bases
        )
    table = tuple(
        tuple(
            yuv_pixel(y, uv, standard, full_input, limited, family == "YUV2HSV")
            for uv in UV_PALETTE
        )
        for y in Y_PALETTE
    )
    rows = {}
    for color_base, uv_base in set(zip(color_bases, uv_bases)):
        rows[(color_base, uv_base)] = bytes(
            channel
            for x in range(width)
            for channel in table[(7 * x + color_base) % 16][
                (3 * (x // 2) + uv_base) % 16
            ]
        )
    return b"".join(rows[bases] for bases in zip(color_bases, uv_bases))


def variants(mode, seed, exhaustive):
    family = mode.split("_")[0]
    if exhaustive:
        return itertools.product(
            (False, True) if family != "YUV2HSV" else (False,),
            (False, True),
            (False, True) if family.startswith("YUV") else (False,),
        )
    return (
        (
            bool(seed & 1) and family != "YUV2HSV",
            bool(seed & 2),
            bool(seed & 4) and family.startswith("YUV"),
        ),
    )


def comparisons(actual, expected, width, height, layout):
    exact = actual == expected
    rows = sorted(
        {0, height - 1}
        | {
            y
            for boundary in range(BLOCK_ROWS, height, BLOCK_ROWS)
            for y in (boundary - 1, boundary, boundary + 1)
            if y < height
        }
    )
    pixels = width * height
    spans = []
    if layout == "YUV422SP":
        spans = [
            (plane * pixels + y * width, width) for plane in range(2) for y in rows
        ]
    elif layout == "U8C3_PLANAR":
        spans = [
            (channel * pixels + y * width, width) for channel in range(3) for y in rows
        ]
    else:
        spans = [(y * width * 3, width * 3) for y in rows]
    row = dict(
        exact=exact,
        actual_bytes=len(actual),
        expected_bytes=len(expected),
        boundary_rows=rows,
        boundary_exact=all(
            actual[start : start + length] == expected[start : start + length]
            for start, length in spans
        ),
    )
    if not exact:
        row["first_differences"] = [
            dict(offset=offset, actual=left, expected=right)
            for offset, (left, right) in itertools.islice(
                (
                    (i, pair)
                    for i, pair in enumerate(zip(actual, expected))
                    if pair[0] != pair[1]
                ),
                16,
            )
        ]
    return row


def run_case(
    processor,
    vg,
    data,
    expected,
    width,
    height,
    source,
    destination,
    mode,
    bgr,
    vu,
    full_input,
    loops,
):
    row = dict(
        size=[width, height],
        mode=mode,
        source=source,
        destination=destination,
        bgr_order=bgr,
        vu_order=vu,
        input_full_range=full_input,
        expected_calls=(height + BLOCK_ROWS - 1) // BLOCK_ROWS,
    )
    source_hash = digest(data)
    irqs, signatures, durations = [], [], []
    exact, boundary = True, True
    try:
        for _ in range(loops):
            before, started = interrupts(), time.monotonic()
            actual = bytes(
                processor.csc_bytes(
                    data,
                    width,
                    height,
                    getattr(vg.ImageTypeIVE, source),
                    getattr(vg.ImageTypeIVE, destination),
                    getattr(vg.CscMode, mode),
                    full_input,
                    bgr,
                    vu,
                )
            )
            durations.append((time.monotonic() - started) * 1000)
            after = interrupts()
            irqs.append(None if before is None or after is None else after - before)
            difference = comparisons(actual, expected, width, height, destination)
            exact &= difference["exact"]
            boundary &= difference["boundary_exact"]
            if not difference["exact"] and "difference" not in row:
                row["difference"] = difference
            signatures.append(digest(actual))
            del actual
        preserved, stable = digest(data) == source_hash, len(set(signatures)) == 1
        irq_valid = all(
            value is None or value == row["expected_calls"] for value in irqs
        )
        passed = exact and boundary and preserved and stable and irq_valid
        row.update(
            status=(
                "failed" if not passed else "unverified" if None in irqs else "passed"
            ),
            exact=bool(exact),
            boundary_exact=bool(boundary),
            source_preserved=preserved,
            stable=stable,
            irq_delta=irqs,
            min_ms=min(durations),
            source_sha256=source_hash,
            expected_sha256=digest(expected),
            output_sha256=signatures[0],
        )
    except Exception as error:
        row.update(status="failed", exception=type(error).__name__, error=str(error))
    return row


def self_check():
    assert hsv_pixel((255, 0, 0)) == (0, 255, 255)
    assert hsv_pixel((0, 255, 0)) == (86, 255, 255)
    assert hsv_pixel((0, 0, 255)) == (171, 255, 255)
    assert rgb_pixel((0, 51, 204), "BT601", True) == (62, 202, 95)
    assert rgb_pixel((255, 255, 255), "BT709", False) == (254, 128, 128)
    assert yuv_pixel(128, (128, 192), "BT601", False, True, False) == (216, 83, 128)
    assert yuv_pixel(128, (128, 192), "BT709", False, True, False) == (227, 99, 128)
    rgb, yuv = patterns(66, 2049)
    assert rgb[BLOCK_ROWS * 66 * 3 : (BLOCK_ROWS + 1) * 66 * 3] != rgb[: 66 * 3]
    assert rgb[2048 * 66 * 3 : 2049 * 66 * 3] != rgb[: 66 * 3]
    pixels = 66 * 2049
    assert (
        yuv[pixels + BLOCK_ROWS * 66 : pixels + (BLOCK_ROWS + 1) * 66]
        != yuv[pixels : pixels + 66]
    )
    assert yuv[pixels + 2048 * 66 : pixels + 2049 * 66] != yuv[pixels : pixels + 66]
    assert chroma_order(chroma_order(yuv, 66, 2049, True), 66, 2049, True) == yuv
    assert (
        reference(66, 2049, "RGB2YUV_BT601_FULL", False)[
            pixels + 2048 * 66 : pixels + 2049 * 66
        ]
        != reference(66, 2049, "RGB2YUV_BT601_FULL", False)[pixels : pixels + 66]
    )
    return dict(self_check="passed", native_calls=0, dependencies="standard_library")


def selected(text, allowed, integer_values=False):
    values = tuple(int(value) if integer_values else value for value in text.split(","))
    if (
        not values
        or len(values) != len(set(values))
        or any(value not in allowed for value in values)
    ):
        raise argparse.ArgumentTypeError(
            "Select unique values from " + ",".join(map(str, allowed))
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--widths", default="64,66")
    parser.add_argument("--heights", default="2047,2048,2049,4095,4096")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--loops", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-seconds", type=int, choices=range(1, 3601), default=600)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
        return 0
    if args.output is None:
        parser.error("--output is required for hardware checks")
    try:
        widths = selected(args.widths, WIDTHS, True)
        heights = selected(args.heights, ALLOWED_HEIGHTS, True)
        modes = selected(args.modes, MODES)
        layouts = selected(args.layouts, LAYOUTS)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    self_check()
    import visiong as vg

    processor, started = vg.IVE(), time.monotonic()
    identities = [file_identity(__file__)]
    for name in ("visiong", "_visiong"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            identities.append(file_identity(module.__file__))
    rows, stopped = [], False
    planned = (
        sum(len(tuple(variants(mode, 0, args.full_matrix))) for mode in modes)
        * len(widths)
        * len(heights)
        * len(layouts)
    )

    def save():
        counts = dict(Counter(row["status"] for row in rows))
        document = dict(
            schema=1,
            complete=not stopped
            and len(rows) == planned
            and counts.get("passed", 0) == planned,
            planned_cases=planned,
            completed_cases=len(rows),
            counts=counts,
            loops=args.loops,
            profile="full_matrix" if args.full_matrix else "boundary",
            elapsed_seconds=time.monotonic() - started,
            stopped_at_deadline=stopped,
            provenance=identities,
            results=rows,
        )
        args.output.write_text(
            json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        return document

    for width_index, width in enumerate(widths):
        for height_index, height in enumerate(heights):
            rgb, yuv = patterns(width, height)
            for mode_index, mode in enumerate(modes):
                family = mode.split("_")[0]
                for layout_index, layout in enumerate(layouts):
                    seed = width_index + 2 * height_index + mode_index + layout_index
                    for bgr, vu, full_input in variants(mode, seed, args.full_matrix):
                        if time.monotonic() - started >= args.max_seconds:
                            stopped = True
                            document = save()
                            print(
                                json.dumps(
                                    {
                                        key: value
                                        for key, value in document.items()
                                        if key != "results"
                                    }
                                )
                            )
                            return 2
                        canonical = reference(width, height, mode, full_input)
                        source, destination = (
                            ("YUV422SP", layout)
                            if family.startswith("YUV")
                            else (layout, "YUV422SP")
                        )
                        data = (
                            chroma_order(yuv, width, height, vu)
                            if family.startswith("YUV")
                            else color_order(rgb, width, height, layout, bgr)
                        )
                        expected = (
                            color_order(
                                canonical,
                                width,
                                height,
                                layout,
                                bgr and family == "YUV2RGB",
                            )
                            if family.startswith("YUV")
                            else chroma_order(canonical, width, height, vu)
                        )
                        del canonical
                        row = run_case(
                            processor,
                            vg,
                            data,
                            expected,
                            width,
                            height,
                            source,
                            destination,
                            mode,
                            bgr,
                            vu,
                            full_input,
                            args.loops,
                        )
                        rows.append(row)
                        del data, expected
                        save()
                        if row["status"] != "passed":
                            print(json.dumps(row), flush=True)
            del rgb, yuv
    document = save()
    print(
        json.dumps({key: value for key, value in document.items() if key != "results"})
    )
    return 0 if document["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
