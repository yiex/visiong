# SPDX-License-Identifier: LGPL-3.0-or-later
"""Track an ordered image sequence using the optional hardware wrapper.
使用可选硬件封装跟踪有序图像序列。
"""

import argparse
import json

import visiong


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-points", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.fps < float("inf"):
        parser.error("--fps must be finite and positive")
    tracker = visiong.LKTracker(max_points=args.max_points, max_interval=None)
    for index, filename in enumerate(args.frames):
        frame = visiong.ImageBuffer.load(filename).to_grayscale()
        result = tracker.update(frame, timestamp=index / args.fps)
        print(json.dumps(result), flush=True)
    tracker.reset()


if __name__ == "__main__":
    main()
