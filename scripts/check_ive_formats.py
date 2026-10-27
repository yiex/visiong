# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check typed buffers, signed casts and hardware color conversion.
检查带类型缓冲区、有符号转换及硬件色彩转换。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import visiong as vg

from check_ive_components import interrupt_count


def typed_buffers():
    cases = [
        ("s8c1", np.int8, [[-128, -1], [0, 127]]),
        ("s16c1", np.int16, [[-32768, -1], [0, 32767]]),
        ("u16c1", np.uint16, [[0, 255], [32768, 65535]]),
        ("u64c1", np.uint64, [[0, 2**32], [2**48, 2**64 - 1]]),
    ]
    for name, dtype, values in cases:
        source = np.asarray(values, dtype=dtype)
        image = vg.ImageBuffer.from_numpy(source, name)
        for actual in (image.to_numpy(), np.asarray(image)):
            assert actual.dtype == source.dtype and actual.shape == source.shape
            assert actual.tobytes() == source.tobytes() and not actual.flags.writeable
        assert image.data == source.tobytes() and image.format == name.upper()
        assert image.to_numpy(copy=True).flags.writeable
        restored = vg.ImageBuffer.from_bytes(source.tobytes(), 2, 2, name)
        assert restored.to_numpy().tobytes() == source.tobytes()
    return {"passed": True, "formats": [case[0] for case in cases]}


def casts(processor):
    signed = [
        -32768,
        -1024,
        -129,
        -128,
        -65,
        -64,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        64,
        65,
        128,
        1024,
        32767,
    ]
    unsigned = [0, 1, 2, 3, 4, 16, 128, 1020, 1024, 32768, 65535]
    modes = [
        (
            vg.Cast16to8Mode.S16_TO_S8,
            signed,
            np.int16,
            "s16c1",
            lambda x: max(-128, min(127, x / 128)),
        ),
        (
            vg.Cast16to8Mode.S16_TO_U8_ABS,
            signed,
            np.int16,
            "s16c1",
            lambda x: min(255, abs(x) / 4),
        ),
        (
            vg.Cast16to8Mode.S16_TO_U8_BIAS,
            signed,
            np.int16,
            "s16c1",
            lambda x: max(0, min(255, x / 4 + 127)),
        ),
        (
            vg.Cast16to8Mode.U16_TO_U8,
            unsigned,
            np.uint16,
            "u16c1",
            lambda x: min(255, x / 4),
        ),
    ]
    errors = {}
    for mode, values, dtype, name, reference in modes:
        source = np.asarray(
            [[values[x % len(values)] for x in range(64)] for _ in range(64)],
            dtype=dtype,
        )
        image = vg.ImageBuffer.from_numpy(source, name)
        outputs = []
        for _ in range(5):
            result = processor.cast_16bit_to_8bit(image, mode)
            actual = result.to_numpy().copy()
            outputs.append(actual.tobytes())
            error = max(
                abs(int(actual[y, x]) - reference(int(source[y, x])))
                for y in range(64)
                for x in range(64)
            )
            assert error <= 1
        assert len(set(outputs)) == 1
        errors[mode.name] = error
    return {"passed": True, "maximum_errors": errors}


def hsv_errors(actual, expected):
    left, right = actual.tobytes(), expected.tobytes()
    hue, saturation, value = 0, 0, 0
    for index in range(0, len(left), 3):
        difference = abs(left[index] - right[index])
        hue = max(hue, min(difference, 256 - difference))
        saturation = max(saturation, abs(left[index + 1] - right[index + 1]))
        value = max(value, abs(left[index + 2] - right[index + 2]))
    return [hue, saturation, value]


def colors(processor, width, height):
    palette = [
        (0, 0, 0),
        (80, 80, 80),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
    image = np.asarray(
        [
            [palette[(x // 16) % len(palette)] for x in range(width)]
            for _ in range(height)
        ],
        dtype=np.uint8,
    )
    source = vg.ImageBuffer.from_numpy(image, "rgb888")
    reference_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    hsv = processor.rgb_to_hsv(source)
    assert hsv.format == "HSV888"
    direct_hsv_error = hsv_errors(hsv.to_numpy(), reference_hsv)
    assert max(direct_hsv_error) <= 2
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).tobytes()
    pixels = width * height
    y = yuv[:pixels]
    u = yuv[pixels : pixels + pixels // 4]
    v = yuv[pixels + pixels // 4 :]
    raw_nv12 = y + bytes(value for pair in zip(u, v) for value in pair)
    raw_nv21 = y + bytes(value for pair in zip(v, u) for value in pair)
    reference_rgb = cv2.cvtColor(
        np.frombuffer(raw_nv12, dtype=np.uint8).reshape(height * 3 // 2, width),
        cv2.COLOR_YUV2RGB_NV12,
    )
    errors = {}
    for name, raw in [("nv12", raw_nv12), ("nv21", raw_nv21)]:
        packed = vg.ImageBuffer.from_bytes(raw, width, height, name)
        assert packed.data == raw
        actual = np.frombuffer(
            processor.yuv_to_rgb(packed).data, dtype=np.uint8
        ).reshape(height, width, 3)
        error = int(cv2.absdiff(actual, reference_rgb).max())
        assert error <= 3
        errors[name] = error
    limited = processor.rgb_to_yuv(source, full_range=False)
    limited_error = max(abs(a - b) for a, b in zip(limited.data, raw_nv12))
    assert limited_error <= 2
    full = processor.rgb_to_yuv(source, full_range=True)
    recovered = np.frombuffer(
        processor.yuv_to_rgb(full, input_full_range=True).data, dtype=np.uint8
    ).reshape(height, width, 3)
    roundtrip_error = int(cv2.absdiff(recovered, image).max())
    assert roundtrip_error <= 3
    return {
        "passed": True,
        "size": [width, height],
        "hsv_errors": direct_hsv_error,
        "yuv_to_rgb_errors": errors,
        "video_yuv_error": limited_error,
        "full_roundtrip_error": roundtrip_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ive_formats.json"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    processor = vg.IVE()
    report = []
    for name, call in [
        ("typed_buffers", typed_buffers),
        ("signed_casts", lambda: casts(processor)),
    ] + [
        (
            "color_" + str(width),
            lambda width=width, height=height: colors(processor, width, height),
        )
        for width, height in [(320, 240), (322, 242), (640, 480)]
    ]:
        start = interrupt_count()
        try:
            row = call()
        except Exception as error:
            row = {"passed": False, "error": type(error).__name__ + ": " + str(error)}
        row.update(name=name, interrupts=interrupt_count() - start)
        report.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(row["passed"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
