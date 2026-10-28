# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capture controlled IVE gradients and compare directional models offline.
采集受控 IVE 梯度，并离线比较方向量化候选模型。

Capture preserves observations without assigning an algorithm definition.
采集只保留观测；候选模型相符不等于已确定硬件算法。
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import zipfile


np = None
PAIR_SIDE = 511
PAIR_COUNT = PAIR_SIDE * PAIR_SIDE
WEIGHTS = (1, 127)
SCHEMA = 1


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def save_npz(path, **arrays):
    """Write simple numeric NPY members without optional NumPy archive helpers.
    不依赖可选的 NumPy 归档函数，写入简单数值 NPY 数据。
    """
    with path.open("xb") as output:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in arrays.items():
                if value.dtype.str not in ("<i2", "<u2", "|u1", "<u4"):
                    raise ValueError("Unsupported capture array type.")
                header = repr(
                    dict(
                        descr=value.dtype.str,
                        fortran_order=False,
                        shape=tuple(value.shape),
                    )
                ).encode("ascii")
                header += b" " * (-(10 + len(header) + 1) % 64) + b"\n"
                if len(header) > 65535:
                    raise ValueError("Capture array header is too large.")
                archive.writestr(
                    name + ".npy",
                    b"\x93NUMPY\x01\x00"
                    + struct.pack("<H", len(header))
                    + header
                    + value.tobytes(order="C"),
                )
        output.flush()
        os.fsync(output.fileno())


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def interrupt_count():
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if line.rstrip().endswith(" rve"):
                fields = line.split(":", 1)[1].split()
                counts = []
                for field in fields:
                    if not field.isdecimal():
                        break
                    counts.append(int(field))
                return sum(counts) if counts else None
    except (OSError, ValueError, IndexError):
        pass
    return None


def module_fingerprints(vg):
    paths = set()
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("visiong") and filename:
            path = Path(filename)
            if path.suffix == ".so":
                paths.add(path)
    package = Path(vg.__file__).parent
    paths.update(package.glob("_visiong*.so"))
    if not paths:
        raise RuntimeError("The loaded VisionG extension could not be identified.")
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and "librve" in Path(fields[5]).name:
                path = Path(fields[5])
                if path.is_file():
                    paths.add(path)
    except OSError:
        pass
    return sorted(
        [dict(name=path.name, sha256=file_hash(path)) for path in paths],
        key=lambda item: (item["name"], item["sha256"]),
    )


def pairs(start=0, count=PAIR_COUNT):
    indices = np.arange(start, start + count, dtype=np.int32)
    return (
        (indices // PAIR_SIDE - 255).astype("<i2"),
        (indices % PAIR_SIDE - 255).astype("<i2"),
    )


def center_coordinates(count, image_size):
    side = image_size // 5
    indices = np.arange(count, dtype=np.int32)
    return indices // side * 5 + 2, indices % side * 5 + 2


def gradient_frame(gx, gy, image_size):
    if len(gx) != len(gy) or len(gx) > (image_size // 5) ** 2:
        raise ValueError("Gradient pairs do not fit the tile grid.")
    frame = np.zeros((image_size, image_size), dtype=np.uint8)
    rows, columns = center_coordinates(len(gx), image_size)
    frame[rows, columns - 1] = np.where(gx < 0, -gx, 0).astype(np.uint8)
    frame[rows, columns + 1] = np.where(gx > 0, gx, 0).astype(np.uint8)
    frame[rows - 1, columns] = np.where(gy < 0, -gy, 0).astype(np.uint8)
    frame[rows + 1, columns] = np.where(gy > 0, gy, 0).astype(np.uint8)
    return frame


def gradient_mask(weight):
    mask = [0] * 25
    mask[11], mask[13] = -weight, weight
    return mask


def center_output(buffer, count, image_size, item_size):
    result = buffer.to_numpy()
    if result.shape != (image_size, image_size):
        raise ValueError("IVE returned an unexpected gradient image shape.")
    if result.dtype.kind != "u" or result.dtype.itemsize != item_size:
        raise ValueError("IVE returned an unexpected gradient output dtype.")
    rows, columns = center_coordinates(count, image_size)
    return np.asarray(result[rows, columns], dtype="<u2" if item_size == 2 else "u1")


def l1_candidate(gx, gy, weight):
    return (np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))) * weight


def directional_models(gx, gy):
    x, y = gx.astype(np.int64), gy.astype(np.int64)
    ax, ay = np.abs(x), np.abs(y)
    horizontal = np.where(x >= 0, 0, 4)
    vertical = np.where(y >= 0, 2, 6)
    diagonal = np.where(x >= 0, np.where(y >= 0, 1, 7), np.where(y >= 0, 3, 5))
    angle = np.arctan2(y, x) * (4.0 / np.pi)
    nearest = (np.floor(angle + 0.5).astype(np.int64) & 7).astype(np.uint8)
    ratio_axis_tie = np.where(
        2 * ay <= ax, horizontal, np.where(2 * ax <= ay, vertical, diagonal)
    )
    ratio_diagonal_tie = np.where(
        2 * ay < ax, horizontal, np.where(2 * ax < ay, vertical, diagonal)
    )
    squared_sum = (ax + ay) ** 2
    tan_22_5 = np.where(
        squared_sum < 2 * ax**2,
        horizontal,
        np.where(squared_sum < 2 * ay**2, vertical, diagonal),
    )
    return dict(
        nearest_atan2=nearest,
        ratio_half_axis_tie=ratio_axis_tie.astype(np.uint8),
        ratio_half_diagonal_tie=ratio_diagonal_tie.astype(np.uint8),
        ratio_tan_22_5_exact=tan_22_5.astype(np.uint8),
    )


def capture(args):
    import visiong as vg

    if not hasattr(vg.IVE, "mag_and_ang_ex"):
        raise RuntimeError("VisionG mag_and_ang_ex is required for controlled capture.")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "capture.json").exists() or any(args.output.glob("*.npz")):
        raise FileExistsError(
            "The output directory already contains gradient captures."
        )
    processor = vg.IVE()
    capacity = (args.image_size // 5) ** 2
    manifest = dict(
        schema=SCHEMA,
        kind="ive_gradient_capture",
        evidence="unverified",
        complete=False,
        pair_count_per_weight=PAIR_COUNT,
        pair_order="gx major, gy minor, each integer from -255 through 255",
        image_shape=[args.image_size, args.image_size],
        tile_shape=[5, 5],
        tile_center=[2, 2],
        input_gradient_units="right-left and bottom-top before kernel weight",
        threshold=0,
        weights=args.weights,
        masks={str(weight): gradient_mask(weight) for weight in args.weights},
        modules=module_fingerprints(vg),
        script_sha256=file_hash(Path(__file__)),
        numpy_version=np.__version__,
        arrays=dict(gx="int16", gy="int16", magnitude="uint16", angle="uint8"),
        batches=[],
        candidate_note="L1 agreement is a model comparison, not an algorithm definition.",
    )
    manifest_path = args.output / "capture.json"
    save_json(manifest_path, manifest)
    for weight in args.weights:
        for batch, start in enumerate(range(0, PAIR_COUNT, capacity)):
            count = min(capacity, PAIR_COUNT - start)
            gx, gy = pairs(start, count)
            frame = gradient_frame(gx, gy, args.image_size)
            source_digest = array_hash(frame)
            source = vg.ImageBuffer.from_numpy(frame, "gray8")
            before = interrupt_count()
            started = time.perf_counter()
            magnitude_buffer, angle_buffer = processor.mag_and_ang_ex(
                source, 0, gradient_mask(weight)
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            after = interrupt_count()
            magnitude = center_output(magnitude_buffer, count, args.image_size, 2)
            angle = center_output(angle_buffer, count, args.image_size, 1)
            unchanged = (
                array_hash(frame) == source_digest
                and array_hash(source.to_numpy()) == source_digest
            )
            filename = "weight_%03d_batch_%03d.npz" % (weight, batch)
            path = args.output / filename
            save_npz(
                path,
                gx=gx,
                gy=gy,
                magnitude=magnitude,
                angle=angle,
                weight=np.asarray(weight, dtype="<u2"),
                start=np.asarray(start, dtype="<u4"),
            )
            expected = l1_candidate(gx, gy, weight)
            differences = np.abs(magnitude.astype(np.int32) - expected)
            delta = after - before if before is not None and after is not None else None
            row = dict(
                file=filename,
                weight=weight,
                start=start,
                count=count,
                evidence="measured",
                sha256=file_hash(path),
                array_sha256={
                    name: array_hash(value)
                    for name, value in dict(
                        gx=gx, gy=gy, magnitude=magnitude, angle=angle
                    ).items()
                },
                source_sha256=source_digest,
                source_unchanged=unchanged,
                interrupts=dict(before=before, after=after, delta=delta),
                hardware_activity=(
                    "observed" if delta is not None and delta > 0 else "unverified"
                ),
                elapsed_ms=elapsed_ms,
                l1_candidate=dict(
                    evidence="inferred",
                    mismatches=int(np.count_nonzero(differences)),
                    maximum_difference=int(differences.max(initial=0)),
                ),
            )
            manifest["batches"].append(row)
            manifest["evidence"] = "measured"
            save_json(manifest_path, manifest)
            print(json.dumps(row), flush=True)
            if not unchanged:
                raise RuntimeError("The source tile image changed during capture.")
    manifest["complete"] = True
    manifest["captured_pairs"] = len(args.weights) * PAIR_COUNT
    manifest["all_hardware_activity_observed"] = all(
        row["hardware_activity"] == "observed" for row in manifest["batches"]
    )
    save_json(manifest_path, manifest)
    print(
        json.dumps(
            dict(
                evidence="measured",
                complete=True,
                weights=args.weights,
                captured_pairs=manifest["captured_pairs"],
                all_hardware_activity_observed=manifest[
                    "all_hardware_activity_observed"
                ],
            )
        ),
        flush=True,
    )


def load_capture(source):
    manifest_path = (
        source / "capture.json" if source.is_dir() else source.parent / "capture.json"
    )
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.is_dir():
        if (
            manifest.get("kind") != "ive_gradient_capture"
            or manifest.get("schema") != SCHEMA
        ):
            raise ValueError(
                "The directory has no supported gradient capture manifest."
            )
        records = manifest.get("batches", [])
        files = []
        for row in records:
            name = row["file"]
            if Path(name).name != name:
                raise ValueError("Capture filenames must be plain basenames.")
            files.append((source / name, row))
    else:
        record = next(
            (row for row in manifest.get("batches", []) if row["file"] == source.name),
            {},
        )
        files = [(source, record)]
    if not files:
        raise ValueError("No measured gradient batches are available.")
    groups = {}
    for path, record in files:
        digest = file_hash(path)
        if record.get("sha256") and digest != record["sha256"]:
            raise ValueError("A gradient batch checksum does not match its manifest.")
        with np.load(path, allow_pickle=False) as batch:
            gx, gy = batch["gx"], batch["gy"]
            magnitude, angle = batch["magnitude"], batch["angle"]
            weight = int(batch["weight"].item())
            start = int(batch["start"].item())
            arrays = dict(gx=gx, gy=gy, magnitude=magnitude, angle=angle)
            for name, value in arrays.items():
                expected_dtype = np.dtype(
                    "<i2"
                    if name in ("gx", "gy")
                    else "<u2" if name == "magnitude" else "u1"
                )
                if value.ndim != 1 or value.dtype != expected_dtype:
                    raise ValueError(
                        "A gradient array has an unexpected dtype or shape."
                    )
            if len({len(value) for value in arrays.values()}) != 1:
                raise ValueError("Gradient batch arrays have different lengths.")
            if weight not in WEIGHTS or start < 0 or start + len(gx) > PAIR_COUNT:
                raise ValueError(
                    "Gradient batch controls are outside the capture domain."
                )
            expected_x, expected_y = pairs(start, len(gx))
            if not np.array_equal(gx, expected_x) or not np.array_equal(gy, expected_y):
                raise ValueError(
                    "Gradient batch coordinates do not match the specified ordering."
                )
            if record:
                if (
                    record["weight"] != weight
                    or record["start"] != start
                    or record["count"] != len(gx)
                ):
                    raise ValueError(
                        "Gradient batch controls do not match its manifest."
                    )
                for name, value in arrays.items():
                    if array_hash(value) != record["array_sha256"][name]:
                        raise ValueError(
                            "A gradient array checksum does not match its manifest."
                        )
            group = groups.setdefault(weight, dict(batches=[], files=[]))
            group["batches"].append(
                (start, {name: value.copy() for name, value in arrays.items()})
            )
            group["files"].append(
                dict(name=path.name, sha256=digest, manifest_verified=bool(record))
            )
    for weight, group in groups.items():
        group["batches"].sort(key=lambda item: item[0])
        coverage = np.zeros(PAIR_COUNT, dtype=np.uint8)
        for start, arrays in group["batches"]:
            if np.any(coverage[start : start + len(arrays["gx"])]):
                raise ValueError(
                    "Gradient batches contain overlapping coordinate ranges."
                )
            coverage[start : start + len(arrays["gx"])] = 1
        group["arrays"] = {
            name: np.concatenate([arrays[name] for _, arrays in group["batches"]])
            for name in ("gx", "gy", "magnitude", "angle")
        }
        group["complete"] = bool(np.all(coverage))
        group["count"] = int(np.count_nonzero(coverage))
        del group["batches"]
    return manifest, groups


def mismatch_file(path, gx, gy, weight, actual, expected, selected):
    indices = np.flatnonzero(selected)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pair_index",
                "gx",
                "gy",
                "weight",
                "effective_gx",
                "effective_gy",
                "actual",
                "candidate",
            ]
        )
        for index in indices:
            x, y = int(gx[index]), int(gy[index])
            writer.writerow(
                [
                    (x + 255) * PAIR_SIDE + y + 255,
                    x,
                    y,
                    weight,
                    x * weight,
                    y * weight,
                    int(actual[index]),
                    int(expected[index]),
                ]
            )
    return dict(file=path.name, rows=len(indices), sha256=file_hash(path))


def boundary_file(path, gx, gy, weight, magnitude, angle):
    ax, ay = np.abs(gx.astype(np.int32)), np.abs(gy.astype(np.int32))
    nonzero = (ax + ay) != 0
    masks = dict(
        ratio_half_horizontal_tie=nonzero & (2 * ay == ax),
        ratio_half_vertical_tie=nonzero & (2 * ax == ay),
        equal_absolute_components=nonzero & (ax == ay),
        horizontal_axis=nonzero & (ay == 0),
        vertical_axis=nonzero & (ax == 0),
        zero_gradient=~nonzero,
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "relation",
                "gx",
                "gy",
                "weight",
                "effective_gx",
                "effective_gy",
                "magnitude",
                "angle",
            ]
        )
        for relation, mask in masks.items():
            for index in np.flatnonzero(mask):
                x, y = int(gx[index]), int(gy[index])
                writer.writerow(
                    [
                        relation,
                        x,
                        y,
                        weight,
                        x * weight,
                        y * weight,
                        int(magnitude[index]),
                        int(angle[index]),
                    ]
                )
    return dict(
        file=path.name,
        sha256=file_hash(path),
        counts={name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
        tan_22_5_nonzero_integer_equalities=0,
        note="tan(pi/8) is irrational; nonzero integer pairs cannot lie exactly on that boundary.",
    )


def analyze(args):
    manifest, groups = load_capture(args.analyze)
    directory = args.report_dir
    if directory is None:
        directory = (
            args.analyze / "analysis"
            if args.analyze.is_dir()
            else args.analyze.parent / (args.analyze.stem + "_analysis")
        )
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema=SCHEMA,
        kind="ive_gradient_analysis",
        evidence=(
            "measured"
            if manifest.get("kind") == "ive_gradient_capture"
            and manifest.get("evidence") == "measured"
            and all(
                item["manifest_verified"]
                for group in groups.values()
                for item in group["files"]
            )
            else "unverified"
        ),
        modules=manifest.get("modules", []),
        algorithm_definition="unverified",
        model_comparisons="inferred",
        zero_gradient="Excluded from directional model comparisons; its measured outputs are reported separately.",
        definitions=dict(
            nearest_atan2="floor(atan2(gy,gx)*4/pi+0.5) modulo 8, excluding zero gradient",
            ratio_half_axis_tie="horizontal if 2*abs(gy)<=abs(gx), vertical if 2*abs(gx)<=abs(gy), otherwise diagonal",
            ratio_half_diagonal_tie="same ratio-0.5 tests with strict inequalities; equality selects diagonal",
            ratio_tan_22_5_exact="horizontal if (abs(gx)+abs(gy))^2<2*gx^2, vertical if that sum is <2*gy^2, otherwise diagonal",
            sign_mapping="+X=0, (+X,+Y)=1, +Y=2, (-X,+Y)=3, -X=4, (-X,-Y)=5, -Y=6, (+X,-Y)=7",
            l1="abs(gx*weight)+abs(gy*weight), compared without fitting or tolerance",
        ),
        note="The exact tan(pi/8) candidate and nearest 45-degree direction are mathematically equivalent away from zero; agreement does not identify the circuit implementation.",
        weights=[],
    )
    for weight, group in sorted(groups.items()):
        gx, gy, magnitude, angle = (
            group["arrays"][name] for name in ("gx", "gy", "magnitude", "angle")
        )
        nonzero = (gx != 0) | (gy != 0)
        expected = l1_candidate(gx, gy, weight)
        l1_diff = magnitude.astype(np.int32) - expected
        prefix = "weight_%03d_" % weight
        row = dict(
            weight=weight,
            count=group["count"],
            complete=group["complete"],
            files=group["files"],
            angle_values=sorted(int(value) for value in np.unique(angle)),
            zero_gradient_outputs=[
                dict(magnitude=int(magnitude[index]), angle=int(angle[index]))
                for index in np.flatnonzero(~nonzero)
            ],
            l1_candidate=dict(
                evidence="inferred",
                mismatches=int(np.count_nonzero(l1_diff)),
                maximum_absolute_difference=int(np.abs(l1_diff).max(initial=0)),
                differences=mismatch_file(
                    directory / (prefix + "l1_mismatches.csv"),
                    gx,
                    gy,
                    weight,
                    magnitude,
                    expected,
                    l1_diff != 0,
                ),
            ),
            directions={},
        )
        for name, candidate in directional_models(
            gx.astype(np.int32) * weight, gy.astype(np.int32) * weight
        ).items():
            differences = nonzero & (angle != candidate)
            row["directions"][name] = dict(
                evidence="inferred",
                compared=int(np.count_nonzero(nonzero)),
                mismatches=int(np.count_nonzero(differences)),
                differences=mismatch_file(
                    directory / (prefix + name + "_mismatches.csv"),
                    gx,
                    gy,
                    weight,
                    angle,
                    candidate,
                    differences,
                ),
            )
        row["integer_boundaries"] = boundary_file(
            directory / (prefix + "integer_boundaries.csv"),
            gx,
            gy,
            weight,
            magnitude,
            angle,
        )
        report["weights"].append(row)
    requested = manifest.get("weights", list(groups))
    report["complete"] = all(
        weight in groups and groups[weight]["complete"] for weight in requested
    )
    save_json(directory / "analysis.json", report)
    print(json.dumps(report), flush=True)


def self_test():
    gx, gy = pairs()
    assert len(gx) == PAIR_COUNT and int(gx.min()) == -255 and int(gx.max()) == 255
    assert np.array_equal(
        (gx.astype(np.int32) + 255) * PAIR_SIDE + gy.astype(np.int32) + 255,
        np.arange(PAIR_COUNT),
    )
    capacity = 64 * 64
    checked = 0
    try:
        import cv2
    except ImportError:
        cv2 = None
    for weight in WEIGHTS:
        for start in range(0, PAIR_COUNT, capacity):
            count = min(capacity, PAIR_COUNT - start)
            x, y = pairs(start, count)
            frame = gradient_frame(x, y, 320)
            rows, columns = center_coordinates(count, 320)
            horizontal = (
                frame[rows, columns + 1].astype(np.int32)
                - frame[rows, columns - 1].astype(np.int32)
            ) * weight
            vertical = (
                frame[rows + 1, columns].astype(np.int32)
                - frame[rows - 1, columns].astype(np.int32)
            ) * weight
            assert np.array_equal(horizontal, x.astype(np.int32) * weight)
            assert np.array_equal(vertical, y.astype(np.int32) * weight)
            assert np.all(np.abs(horizontal) < 32768) and np.all(
                np.abs(vertical) < 32768
            )
            if cv2 is not None:
                mask = np.asarray(gradient_mask(weight), dtype=np.int16).reshape(5, 5)
                filtered_x = cv2.filter2D(
                    frame, cv2.CV_16S, mask, borderType=cv2.BORDER_CONSTANT
                )
                filtered_y = cv2.filter2D(
                    frame, cv2.CV_16S, mask.T, borderType=cv2.BORDER_CONSTANT
                )
                assert np.array_equal(filtered_x[rows, columns], horizontal)
                assert np.array_equal(filtered_y[rows, columns], vertical)
            checked += count
    nonzero = (gx != 0) | (gy != 0)
    models = directional_models(gx, gy)
    assert np.array_equal(
        models["nearest_atan2"][nonzero], models["ratio_tan_22_5_exact"][nonzero]
    )
    half_tie = nonzero & (
        (2 * np.abs(gy.astype(np.int32)) == np.abs(gx.astype(np.int32)))
        | (2 * np.abs(gx.astype(np.int32)) == np.abs(gy.astype(np.int32)))
    )
    assert np.array_equal(
        (models["ratio_half_axis_tie"] != models["ratio_half_diagonal_tie"]) & nonzero,
        half_tie,
    )
    axis_x = np.asarray([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int16)
    axis_y = np.asarray([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int16)
    assert all(
        np.array_equal(value, np.arange(8))
        for value in directional_models(axis_x, axis_y).values()
    )
    assert int(l1_candidate(gx, gy, 127).max()) == 64770
    result = dict(
        evidence="unverified",
        hardware_capture="not_run",
        host_reference_checks="passed",
        gradient_pairs_checked=checked,
        full_domain_pairs=PAIR_COUNT,
        weights=list(WEIGHTS),
        batches_per_weight=64,
        maximum_absolute_component=32385,
        maximum_l1_candidate=64770,
        opencv_reference="passed" if cv2 is not None else "unverified",
        exact_tan_matches_nearest_atan2=True,
        ratio_half_integer_ties=int(np.count_nonzero(half_tie)),
    )
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--analyze",
        type=Path,
        help="Analyze a capture directory or one raw batch without importing VisionG.",
    )
    actions.add_argument(
        "--self-test",
        action="store_true",
        help="Check fixture construction and candidate arithmetic without hardware.",
    )
    parser.add_argument("--output", type=Path, default=Path("gradient_capture"))
    parser.add_argument(
        "--report-dir", type=Path, help="Offline analysis output directory."
    )
    parser.add_argument(
        "--weights", type=int, nargs="+", default=list(WEIGHTS), choices=WEIGHTS
    )
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()
    if args.image_size < 5 or args.image_size > 4096 or args.image_size % 5:
        parser.error("image-size must be a multiple of 5 in [5,4096]")
    if len(args.weights) != len(set(args.weights)):
        parser.error("weights must not contain duplicates")
    global np
    import numpy as np

    if args.self_test:
        self_test()
    elif args.analyze is not None:
        analyze(args)
    else:
        capture(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(
            "Gradient operation failed (%s)." % type(error).__name__
        ) from None
