# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture YUV-to-RGB/HSV grids with independent 2x2 chroma cells.
以独立 2×2 色度单元采集 YUV 转 RGB/HSV 的传递网格。
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import visiong as vg


MODE_NAMES = (
    "YUV2RGB_BT601_FULL",
    "YUV2RGB_BT709_FULL",
    "YUV2RGB_BT601_LIMITED",
    "YUV2RGB_BT709_LIMITED",
    "YUV2HSV_BT601_FULL",
    "YUV2HSV_BT709_FULL",
    "YUV2HSV_BT601_LIMITED",
    "YUV2HSV_BT709_LIMITED",
)
READBACK_CHUNK_BYTES = 1024 * 1024


def interrupts():
    for line in Path("/proc/interrupts").read_text().splitlines():
        if line.rstrip().endswith(" rve"):
            return int(line.split()[1])
    raise RuntimeError("IVE interrupt counter is unavailable")


def verify_outputs(files, digests):
    """Synchronize capture files and compare their readback SHA256 values.
    同步采集文件并比对读回数据的 SHA256。
    """
    for output in files.values():
        output.flush()
        os.fsync(output.fileno())
    verified = {}
    for name, output in files.items():
        digest, size = hashlib.sha256(), 0
        path = Path(output.name)
        with path.open("rb") as source:
            while True:
                chunk = source.read(READBACK_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        actual = digest.hexdigest()
        if actual != digests[name].hexdigest():
            raise RuntimeError("Capture readback SHA256 mismatch: " + path.name)
        verified[path.name] = dict(bytes=size, sha256=actual)
    return verified


def write_report(path, report):
    """Publish completion metadata only after its temporary file is synchronized.
    临时文件完成同步后才发布完成元数据。
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("yuv_capture"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--modes",
        default=",".join(MODE_NAMES),
        help="Comma-separated CSC mode names; default: all eight / 逗号分隔的 CSC 模式名，默认全部八种",
    )
    parser.add_argument(
        "--v-start",
        type=int,
        help="First V value in a continuous interval, default 0 / 连续区间的首个 V 值，默认 0",
    )
    parser.add_argument(
        "--v-stop",
        type=int,
        help="Exclusive V limit, default 256; either V option selects continuous layers / V 的开区间上界，默认 256；任一 V 参数均选择连续层",
    )
    args = parser.parse_args()
    names = [name.strip() for name in args.modes.split(",")]
    if any(name not in MODE_NAMES for name in names):
        parser.error("--modes must contain only supported CSC mode names")
    if len(set(names)) != len(names):
        parser.error("--modes must not repeat a mode name")
    if args.v_start is not None or args.v_stop is not None:
        start = 0 if args.v_start is None else args.v_start
        stop = 256 if args.v_stop is None else args.v_stop
        if not 0 <= start < stop <= 256:
            parser.error("V interval must satisfy 0 <= v-start < v-stop <= 256")
        layers = list(range(start, stop))
    else:
        layers = (
            list(range(256))
            if args.full
            else [0, 1, 16, 63, 64, 127, 128, 129, 191, 192, 239, 240, 254, 255]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "capture.json"
    metadata.unlink(missing_ok=True)
    modes = [(name, getattr(vg.CscMode, name)) for name in names]
    y_bytes = b"".join(bytes([y]) * 1024 for y in range(256))
    processor = vg.IVE()
    before = interrupts()
    digests = {name: hashlib.sha256() for name, _ in modes}
    with ExitStack() as stack:
        files = {
            name: stack.enter_context((args.output / (name + ".bin")).open("wb"))
            for name, _ in modes
        }
        for v in layers:
            uv_row = bytes(value for u in range(256) for value in (u, v))
            source = vg.ImageBuffer.from_bytes(y_bytes + uv_row * 256, 512, 512, "nv12")
            for name, mode in modes:
                output = processor.csc(source, mode)
                data = (
                    np.frombuffer(output.data, dtype=np.uint8)
                    .reshape(512, 512, 3)[::2, ::2]
                    .tobytes()
                )
                if files[name].write(data) != len(data):
                    raise OSError("Incomplete capture write: " + name)
                digests[name].update(data)
            print(json.dumps({"v": v, "interrupts": interrupts() - before}), flush=True)
        verified = verify_outputs(files, digests)
    report = {
        "order": "V layer, Y row, U column, RGB or HSV bytes",
        "v_layers": layers,
        "shape": [len(layers), 256, 256, 3],
        "interrupts": interrupts() - before,
        "files": {
            name + ".bin": digest.hexdigest() for name, digest in digests.items()
        },
        "storage_verification": {
            "fsync": True,
            "readback_chunk_bytes": READBACK_CHUNK_BYTES,
            "files": verified,
        },
        "module_sha256": hashlib.sha256(
            (Path(vg.__file__).parent / "_visiong.so").read_bytes()
        ).hexdigest(),
    }
    write_report(metadata, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
