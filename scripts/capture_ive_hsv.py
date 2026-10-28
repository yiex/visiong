# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture hardware HSV values for a deterministic RGB cube.
采集确定性 RGB 色立方体对应的硬件 HSV 结果。
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import visiong as vg


READBACK_CHUNK_BYTES = 1024 * 1024


def interrupt_count():
    for line in Path("/proc/interrupts").read_text().splitlines():
        if line.rstrip().endswith(" rve"):
            return int(line.split()[1])
    raise RuntimeError("IVE interrupt counter was not found")


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
    parser.add_argument("--output", type=Path, default=Path("hsv_cube"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "capture.json"
    metadata.unlink(missing_ok=True)
    layers = (
        list(range(256))
        if args.full
        else [0, 1, 2, 15, 16, 17, 63, 64, 127, 128, 191, 192, 239, 240, 254, 255]
    )
    modes = [
        ("full601", vg.CscMode.RGB2HSV_BT601_FULL),
        ("full709", vg.CscMode.RGB2HSV_BT709_FULL),
        ("limit601", vg.CscMode.RGB2HSV_BT601_LIMITED),
        ("limit709", vg.CscMode.RGB2HSV_BT709_LIMITED),
    ]
    grid = (
        np.frombuffer(
            bytes(
                value
                for green in range(256)
                for blue in range(256)
                for value in (0, green, blue)
            ),
            dtype=np.uint8,
        )
        .reshape(256, 256, 3)
        .copy()
    )
    processor = vg.IVE()
    digests = {name: hashlib.sha256() for name, _ in modes}
    repeated = []
    before = interrupt_count()
    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context((args.output / (name + ".hsv")).open("wb"))
            for name, _ in modes
        }
        for red in layers:
            grid[:, :, 0] = red
            source = vg.ImageBuffer.from_numpy(grid, "rgb888")
            for name, mode in modes:
                result = processor.csc(source, mode)
                assert result.format == "HSV888"
                data = result.data
                assert len(data) == 256 * 256 * 3
                if outputs[name].write(data) != len(data):
                    raise OSError("Incomplete capture write: " + name)
                digests[name].update(data)
                if red in (0, 127, 255):
                    signatures = {
                        hashlib.sha256(processor.csc(source, mode).data).hexdigest()
                        for _ in range(3)
                    }
                    signatures.add(hashlib.sha256(data).hexdigest())
                    repeated.append(dict(red=red, mode=name, unique=len(signatures)))
            if red % 16 == 0:
                print(
                    json.dumps(dict(red=red, interrupts=interrupt_count() - before)),
                    flush=True,
                )
        verified = verify_outputs(outputs, digests)
    if any(row["unique"] != 1 for row in repeated):
        raise RuntimeError("Repeated HSV captures produced different outputs")
    module = Path(vg.__file__).parent / "_visiong.so"
    report = dict(
        order="R layer, G row, B column, H/S/V bytes",
        shape=[len(layers), 256, 256, 3],
        red_layers=layers,
        module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        files={name + ".hsv": digest.hexdigest() for name, digest in digests.items()},
        storage_verification=dict(
            fsync=True,
            readback_chunk_bytes=READBACK_CHUNK_BYTES,
            files=verified,
        ),
        repeat_checks=repeated,
        interrupts=interrupt_count() - before,
    )
    write_report(metadata, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
