# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check bounded GMM model establishment, recall and independent snapshots.
检查有界 GMM 模型建立、复现和独立输出快照，不推断全部高斯更新公式。
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


VALUES = (0, 255, 64, 192, 128)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def interrupt_count():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def frame(width, height, value):
    raw = bytearray([80]) * (width * height)
    for y in range(4, height // 2):
        raw[y * width + 3 : y * width + width // 2] = bytes([value]) * (width // 2 - 3)
    return bytes(raw)


def split(raw, width, height):
    inside, outside = bytearray(), bytearray()
    for y in range(height):
        row = raw[y * width : (y + 1) * width]
        if 4 <= y < height // 2:
            inside.extend(row[3 : width // 2])
            outside.extend(row[:3])
            outside.extend(row[width // 2 :])
        else:
            outside.extend(row)
    return bytes(inside), bytes(outside)


def sequence(models):
    result = [0] * 12
    for value in VALUES[1:models]:
        result.extend([value, value])
    result.extend(VALUES[:models] * 2)
    return result


def rejected(function):
    before = interrupt_count()
    try:
        function()
    except (ValueError, RuntimeError, TypeError):
        after = interrupt_count()
        if before is not None and after != before:
            raise AssertionError("Rejected input submitted hardware work")
        return
    raise AssertionError("Invalid model state or count was accepted")


def check(vg, width, height, models, algorithm, loops):
    processor = vg.IVE()
    model = vg.IVEModel(width, height)
    factor_raw = b"\x08\x08" * (width * height)
    factor = vg.ImageBuffer.from_bytes(factor_raw, width, height, "u16c1")

    def call(source, first=False, count=models):
        if algorithm == 1:
            return processor.gmm_config(source, model, count, first)
        return processor.gmm2_config(source, factor, model, count, first)

    initial = vg.ImageBuffer.from_bytes(frame(width, height, 0), width, height, "gray8")
    rejected(lambda: call(initial))
    signatures = []
    runs = []
    for repeat in range(loops):
        seen, held, rows = set(), [], []
        for index, value in enumerate(sequence(models)):
            source_raw = frame(width, height, value)
            source = vg.ImageBuffer.from_bytes(source_raw, width, height, "gray8")
            before = interrupt_count()
            outputs = call(source, index == 0)
            after = interrupt_count()
            if before is not None and after != before + 1:
                raise AssertionError("GMM hardware interrupt count differs from one")
            raws = [bytes(output.data) for output in outputs]
            if len(raws) != 3 or any(
                output.format != "GRAY8"
                or output.width != width
                or output.height != height
                or len(raw) != width * height
                for output, raw in zip(outputs, raws)
            ):
                raise AssertionError("Unexpected output type, size or count")
            foreground, background, matched = raws
            inside = [split(raw, width, height)[0] for raw in raws]
            outside = [split(raw, width, height)[1] for raw in raws]
            expected_count = len(seen | {value})
            if any(
                item >> 4 != expected_count
                or bool(item & 1) != (value in seen)
                or item & 1
                and ((item >> 1) & 7) >= (item >> 4)
                for item in inside[2]
            ):
                raise AssertionError(
                    "Bounded component count or matched record differs"
                )
            if set(foreground) - {0, 255} or any(len(set(raw)) != 1 for raw in inside):
                raise AssertionError(
                    "Nonbinary foreground or unequal constant ROI outputs"
                )
            if index == 0:
                if (
                    foreground != b"\xff" * (width * height)
                    or background != source_raw
                    or matched != b"\x10" * (width * height)
                ):
                    raise AssertionError(
                        "Initialization differs from the measured contract"
                    )
            elif any(outside[0]) or set(outside[1]) != {80} or set(outside[2]) != {17}:
                raise AssertionError("Unchanged outside region was corrupted")
            if 0 < index < 12 and (any(inside[0]) or set(inside[1]) != {0}):
                raise AssertionError("Constant training region is not quiet")
            if bytes(source.data) != source_raw or bytes(factor.data) != factor_raw:
                raise AssertionError("Input was modified")
            # A different hardware operator overwrites shared work buffers.
            # 用其他硬件算子覆盖共享工作区，再检查模型和已返回的快照。
            before = interrupt_count()
            interference = processor.logic_op(source, source, vg.LogicOp.XOR)
            after = interrupt_count()
            if bytes(interference.data) != bytes(width * height):
                raise AssertionError("Independent XOR reference failed")
            if before is not None and after != before + 1:
                raise AssertionError("XOR hardware interrupt count differs from one")
            if bytes(source.data) != source_raw or bytes(factor.data) != factor_raw:
                raise AssertionError("Input was modified by the interfering operation")
            if any(bytes(output.data) != raw for output, raw in zip(outputs, raws)):
                raise AssertionError(
                    "The interfering operation changed current outputs"
                )
            if any(digest(bytes(output.data)) != saved for output, saved in held):
                raise AssertionError("A retained output snapshot was changed")
            if len(held) < 6:
                held.extend((output, digest(raw)) for output, raw in zip(outputs, raws))
            rows.append([digest(raw) for raw in raws])
            seen.add(value)
            if index == 1:
                rejected(lambda: call(source, count=0))
                rejected(lambda: call(source, count=6))
        signatures.append(rows)
        runs.append(dict(frames=len(rows), sha256=rows))
        before = interrupt_count()
        model.reset()
        if before is not None and interrupt_count() != before:
            raise AssertionError("Model reset submitted hardware work")
        rejected(lambda: call(initial))
    if any(rows != signatures[0] for rows in signatures[1:]):
        raise AssertionError("Reset and replay changed the bounded sequence")
    return dict(
        status="passed",
        size=[width, height],
        algorithm=algorithm,
        max_models=models,
        runs=runs,
        irq_available=interrupt_count() is not None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="64x64,66x70")
    parser.add_argument("--models", default="1,2,3,4,5")
    parser.add_argument("--loops", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path, default=Path("ive_gmm.json"))
    args = parser.parse_args()
    try:
        sizes = [tuple(map(int, value.split("x"))) for value in args.sizes.split(",")]
        models = [int(value) for value in args.models.split(",")]
        if any(len(size) != 2 or min(size) < 64 or max(size) > 640 for size in sizes):
            raise ValueError()
        if not models or any(value not in range(1, 6) for value in models):
            raise ValueError()
    except ValueError:
        parser.error("Use dimensions in [64,640] and model counts in [1,5]")
    import visiong as vg

    report = dict(
        scope="Bounded establishment and two recall cycles; not a Gaussian update or long-term retention model",
        results=[],
    )
    for width, height in sizes:
        for algorithm in (1, 2):
            for count in models:
                start = time.monotonic()
                try:
                    row = check(vg, width, height, count, algorithm, args.loops)
                except Exception as error:
                    row = dict(
                        status="failed",
                        size=[width, height],
                        algorithm=algorithm,
                        max_models=count,
                        error=str(error),
                    )
                row["elapsed_seconds"] = time.monotonic() - start
                report["results"].append(row)
                report["counts"] = dict(
                    Counter(item["status"] for item in report["results"])
                )
                args.output.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {key: value for key, value in row.items() if key != "runs"}
                    ),
                    flush=True,
                )
    return 1 if report["counts"].get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
