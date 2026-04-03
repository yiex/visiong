#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import ALL_LABELS, KeywordSpottingDSCNN, ensure_dir, seed_everything, softmax, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DS-CNN keyword spotter on prepared Speech Commands features.")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(prepared_root: Path, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(prepared_root / "features" / "{}.npz".format(split_name), allow_pickle=False)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def export_onnx(model: nn.Module, output_path: Path, input_shape: Tuple[int, ...]) -> None:
    cpu_model = KeywordSpottingDSCNN(num_classes=len(ALL_LABELS), channels=model.classifier.in_features)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()
    sample = torch.zeros(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            cpu_model,
            sample,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=12,
            do_constant_folding=True,
        )


def evaluate_onnx(onnx_path: Path, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    total_correct = 0
    confidence_sum = 0.0
    for index in range(features.shape[0]):
        logits = session.run(["logits"], {input_name: features[index : index + 1]})[0].reshape(-1)
        probs = softmax(logits)
        prediction = int(np.argmax(probs))
        total_correct += int(prediction == int(labels[index]))
        confidence_sum += float(np.max(probs))
    total = max(features.shape[0], 1)
    return {
        "accuracy": total_correct / total,
        "avg_confidence": confidence_sum / total,
    }


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    seed_everything(args.seed)
    device = choose_device(args.device)

    train_features, train_labels = load_split(prepared_root, "train")
    val_features, val_labels = load_split(prepared_root, "val")
    test_features, test_labels = load_split(prepared_root, "test")

    train_loader = make_loader(train_features, train_labels, args.batch_size, shuffle=True)
    val_loader = make_loader(val_features, val_labels, args.batch_size, shuffle=False)
    test_loader = make_loader(test_features, test_labels, args.batch_size, shuffle=False)

    model = KeywordSpottingDSCNN(num_classes=len(ALL_LABELS), channels=args.channels).to(device)

    class_counts = np.bincount(train_labels, minlength=len(ALL_LABELS)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))

        scheduler.step()
        train_metrics = {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": total_correct / max(total_examples, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "lr": float(scheduler.get_last_lr()[0]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(
            "epoch {:02d} lr={:.6f} train_loss={:.4f} train_acc={:.3f} val_loss={:.4f} val_acc={:.3f}".format(
                epoch,
                history[-1]["lr"],
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = output_root / "kws_ds_cnn.pt"
    torch.save(
        {
            "state_dict": best_state,
            "labels": ALL_LABELS,
            "channels": args.channels,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
        },
        checkpoint_path,
    )

    onnx_path = output_root / "kws_ds_cnn.onnx"
    export_onnx(model, onnx_path, input_shape=(1,) + tuple(train_features.shape[1:]))
    (output_root / "labels.txt").write_text("\n".join(ALL_LABELS) + "\n", encoding="utf-8")

    test_metrics = evaluate(model, test_loader, criterion, device)
    onnx_metrics = evaluate_onnx(onnx_path, test_features, test_labels)

    summary = {
        "prepared_root": str(prepared_root),
        "output_root": str(output_root),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "channels": args.channels,
        "labels": ALL_LABELS,
        "class_counts": class_counts.tolist(),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "onnx_accuracy": onnx_metrics["accuracy"],
        "onnx_avg_confidence": onnx_metrics["avg_confidence"],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "onnx": str(onnx_path),
            "labels": str(output_root / "labels.txt"),
        },
        "history": history,
    }
    write_json(output_root / "training_summary.json", summary)

    print("best_epoch={}".format(best_epoch))
    print("best_val_accuracy={:.4f}".format(best_val_accuracy))
    print("test_accuracy={:.4f}".format(test_metrics["accuracy"]))
    print("onnx_accuracy={:.4f}".format(onnx_metrics["accuracy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
