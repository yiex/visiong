# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check a full RGB cube from capture_ive_hsv.py against an integer model.
以整数模型逐字节检查完整 RGB 色立方体的实机采集结果。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ive_hsv_reference import rgb_pixel_to_ive_hsv, rgb_to_ive_hsv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube", type=Path, help="HSV bytes in R,G,B cube order")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cube.stat().st_size != 256**3 * 3:
        parser.error("The capture must contain exactly 256**3 HSV888 pixels.")
    actual = np.memmap(args.cube, dtype=np.uint8, mode="r", shape=(256, 256, 256, 3))
    green, blue = np.indices((256, 256), dtype=np.uint8)
    mismatches = [0, 0, 0]
    maximum = [0, 0, 0]
    digest = hashlib.sha256()
    rgb = np.empty((256, 256, 3), dtype=np.uint8)
    rgb[..., 1], rgb[..., 2] = green, blue
    for red in range(256):
        rgb[..., 0] = red
        predicted = rgb_to_ive_hsv(rgb)
        digest.update(predicted.tobytes())
        error = np.abs(predicted.astype(np.int16) - actual[red].astype(np.int16))
        error[..., 0] = np.minimum(error[..., 0], 256 - error[..., 0])
        for channel in range(3):
            mismatches[channel] += int(np.count_nonzero(error[..., channel]))
            maximum[channel] = max(maximum[channel], int(error[..., channel].max()))
        for green_value in range(256):
            for blue_value in (0, 1, 127, 128, 254, 255):
                if tuple(predicted[green_value, blue_value]) != rgb_pixel_to_ive_hsv(
                    red, green_value, blue_value
                ):
                    raise AssertionError("Scalar and array references disagree.")
    report = {
        "pixels": 256**3,
        "channel_order": "HSV",
        "mismatched_pixels_per_channel": mismatches,
        "maximum_error_per_channel": maximum,
        "predicted_sha256": digest.hexdigest(),
        "passed": not any(mismatches),
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
