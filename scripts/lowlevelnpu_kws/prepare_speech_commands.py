#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common import (
    ALL_LABELS,
    KEYWORDS,
    NUM_FRAMES,
    NUM_MEL_BINS,
    SAMPLE_RATE,
    SILENCE_LABEL,
    UNKNOWN_LABEL,
    create_augmented_audio,
    create_silence_audio,
    ensure_dir,
    feature_tensor_from_audio,
    read_wav_pcm16,
    reset_dir,
    seed_everything,
    write_json,
    write_lines,
    write_wav_pcm16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Google Speech Commands features and board bundles for KWS.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unknown-ratio", type=float, default=0.12)
    parser.add_argument("--silence-ratio", type=float, default=0.12)
    parser.add_argument("--board-per-label", type=int, default=10)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_split_file(path: Path) -> set:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def list_background_noises(raw_root: Path) -> List[np.ndarray]:
    noise_dir = raw_root / "_background_noise_"
    noises = []
    for wav_path in sorted(noise_dir.glob("*.wav")):
        noises.append(read_wav_pcm16(wav_path))
    if not noises:
        raise ValueError("No background noise WAV files found under {}".format(noise_dir))
    return noises


def discover_samples(raw_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    validation_set = read_split_file(raw_root / "validation_list.txt")
    testing_set = read_split_file(raw_root / "testing_list.txt")
    samples = {
        "train": {"known": [], "unknown": []},
        "val": {"known": [], "unknown": []},
        "test": {"known": [], "unknown": []},
    }

    for label_dir in sorted(raw_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label.startswith("_"):
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            relative = wav_path.relative_to(raw_root).as_posix()
            if relative in validation_set:
                split = "val"
            elif relative in testing_set:
                split = "test"
            else:
                split = "train"
            bucket = "known" if label in KEYWORDS else "unknown"
            samples[split][bucket].append(wav_path)
    return samples


def sample_unknowns(paths: Sequence[Path], target_count: int, seed: int) -> List[Path]:
    if target_count <= 0:
        return []
    if not paths:
        return []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    selected = [paths[int(i)] for i in indices[: min(target_count, len(paths))]]
    return sorted(selected)


def build_split_specs(
    split_name: str,
    split_samples: Dict[str, List[Path]],
    unknown_ratio: float,
    silence_ratio: float,
    seed: int,
) -> Tuple[List[Dict], Dict[str, int], List[Path]]:
    known_paths = sorted(split_samples["known"])
    unknown_paths = sorted(split_samples["unknown"])
    known_count = len(known_paths)
    unknown_target = int(round(known_count * unknown_ratio))
    silence_target = int(round(known_count * silence_ratio))

    selected_unknowns = sample_unknowns(unknown_paths, unknown_target, seed + 101)
    specs: List[Dict] = []
    label_counts = {label: 0 for label in ALL_LABELS}

    for wav_path in known_paths:
        label = wav_path.parent.name
        specs.append({"kind": "wav", "path": wav_path, "label": label})
        label_counts[label] += 1

    for wav_path in selected_unknowns:
        specs.append({"kind": "wav", "path": wav_path, "label": UNKNOWN_LABEL})
        label_counts[UNKNOWN_LABEL] += 1

    for silence_index in range(silence_target):
        specs.append({"kind": "silence", "index": silence_index, "label": SILENCE_LABEL})
        label_counts[SILENCE_LABEL] += 1

    rng = np.random.default_rng(seed + 303)
    order = np.arange(len(specs))
    if split_name == "train":
        rng.shuffle(order)
    specs = [specs[int(index)] for index in order]
    return specs, label_counts, selected_unknowns


def materialize_split(
    split_name: str,
    specs: Sequence[Dict],
    background_noises: Sequence[np.ndarray],
    output_root: Path,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features = np.zeros((len(specs), 1, NUM_FRAMES, NUM_MEL_BINS), dtype=np.float32)
    labels = np.zeros((len(specs),), dtype=np.int64)
    sources: List[str] = []

    for index, spec in enumerate(specs):
        rng = np.random.default_rng(seed + index * 1009)
        if spec["kind"] == "silence":
            audio = create_silence_audio(rng, background_noises, deterministic_index=index)
            source = "generated_silence_{:05d}.wav".format(index)
        else:
            audio = read_wav_pcm16(spec["path"])
            source = spec["path"].as_posix()
            if split_name == "train":
                audio = create_augmented_audio(
                    audio,
                    rng=rng,
                    background_noises=background_noises,
                    allow_noise=True,
                    allow_shift=True,
                )
        feature = feature_tensor_from_audio(audio)
        features[index] = feature[0]
        labels[index] = ALL_LABELS.index(spec["label"])
        sources.append(source)

    split_dir = ensure_dir(output_root / "features")
    np.savez_compressed(split_dir / "{}.npz".format(split_name), features=features, labels=labels, sources=np.array(sources))
    return features, labels, sources


def export_calibration(features: np.ndarray, calibration_root: Path, count: int) -> None:
    npy_root = reset_dir(calibration_root / "npy")
    selected = min(count, features.shape[0])
    quant_lines = []
    for index in range(selected):
        path = npy_root / "{:04d}.npy".format(index)
        np.save(path, features[index : index + 1].astype(np.float32))
        quant_lines.append(str(path.resolve()))
    write_lines(calibration_root / "quant_dataset.txt", quant_lines)


def export_board_bundle(
    raw_root: Path,
    output_root: Path,
    board_per_label: int,
    known_test_paths: Sequence[Path],
    selected_unknown_test_paths: Sequence[Path],
    background_noises: Sequence[np.ndarray],
    seed: int,
) -> None:
    bundle_root = reset_dir(output_root / "board_bundle")
    eval_root = ensure_dir(bundle_root / "eval")
    labels_path = bundle_root / "labels.txt"
    write_lines(labels_path, ALL_LABELS)

    manifest_rows = []
    keyword_buckets: Dict[str, List[Path]] = {label: [] for label in KEYWORDS}
    for wav_path in known_test_paths:
        label = wav_path.parent.name
        if label in keyword_buckets:
            keyword_buckets[label].append(wav_path)

    for label in KEYWORDS:
        target_dir = ensure_dir(eval_root / label)
        for wav_path in keyword_buckets[label][:board_per_label]:
            target_path = target_dir / wav_path.name
            shutil.copyfile(wav_path, target_path)
            manifest_rows.append((target_path.relative_to(bundle_root).as_posix(), label))

    unknown_dir = ensure_dir(eval_root / UNKNOWN_LABEL)
    for wav_path in list(selected_unknown_test_paths)[:board_per_label]:
        target_name = "{}__{}".format(wav_path.parent.name, wav_path.name)
        target_path = unknown_dir / target_name
        shutil.copyfile(wav_path, target_path)
        manifest_rows.append((target_path.relative_to(bundle_root).as_posix(), UNKNOWN_LABEL))

    silence_dir = ensure_dir(eval_root / SILENCE_LABEL)
    for index in range(board_per_label):
        rng = np.random.default_rng(seed + 5000 + index)
        audio = create_silence_audio(rng, background_noises, deterministic_index=index)
        target_path = silence_dir / "silence_{:03d}.wav".format(index)
        write_wav_pcm16(target_path, audio, sample_rate=SAMPLE_RATE)
        manifest_rows.append((target_path.relative_to(bundle_root).as_posix(), SILENCE_LABEL))

    manifest_path = bundle_root / "eval_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relative_path", "label"])
        for row in manifest_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    ensure_dir(output_root)
    seed_everything(args.seed)

    background_noises = list_background_noises(raw_root)
    discovered = discover_samples(raw_root)

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "labels": ALL_LABELS,
        "frontend": {
            "sample_rate": SAMPLE_RATE,
            "num_frames": NUM_FRAMES,
            "num_mel_bins": NUM_MEL_BINS,
        },
        "splits": {},
    }

    train_specs, train_counts, _ = build_split_specs(
        "train",
        discovered["train"],
        args.unknown_ratio,
        args.silence_ratio,
        args.seed + 11,
    )
    val_specs, val_counts, _ = build_split_specs(
        "val",
        discovered["val"],
        args.unknown_ratio,
        args.silence_ratio,
        args.seed + 22,
    )
    test_specs, test_counts, selected_unknown_test_paths = build_split_specs(
        "test",
        discovered["test"],
        args.unknown_ratio,
        args.silence_ratio,
        args.seed + 33,
    )

    train_features, train_labels, _ = materialize_split(
        "train",
        train_specs,
        background_noises,
        output_root,
        args.seed + 1000,
    )
    materialize_split("val", val_specs, background_noises, output_root, args.seed + 2000)
    materialize_split("test", test_specs, background_noises, output_root, args.seed + 3000)

    export_calibration(train_features, ensure_dir(output_root / "calibration"), args.calibration_count)
    export_board_bundle(
        raw_root=raw_root,
        output_root=output_root,
        board_per_label=args.board_per_label,
        known_test_paths=discovered["test"]["known"],
        selected_unknown_test_paths=selected_unknown_test_paths,
        background_noises=background_noises,
        seed=args.seed,
    )
    write_lines(output_root / "labels.txt", ALL_LABELS)

    summary["splits"]["train"] = train_counts
    summary["splits"]["val"] = val_counts
    summary["splits"]["test"] = test_counts
    summary["artifacts"] = {
        "train_features": str((output_root / "features" / "train.npz").resolve()),
        "val_features": str((output_root / "features" / "val.npz").resolve()),
        "test_features": str((output_root / "features" / "test.npz").resolve()),
        "quant_dataset": str((output_root / "calibration" / "quant_dataset.txt").resolve()),
        "board_bundle": str((output_root / "board_bundle").resolve()),
    }
    write_json(output_root / "prepare_summary.json", summary)

    print("Prepared dataset under {}".format(output_root))
    print("Train split counts: {}".format(train_counts))
    print("Val split counts: {}".format(val_counts))
    print("Test split counts: {}".format(test_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
