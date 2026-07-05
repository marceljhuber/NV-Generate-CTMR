# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18

from .oct_data import define_oct_image_transform, load_oct_manifest


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
SOURCE_TO_CLASSIFIER_LABEL = {1: 0, 2: 1, 3: 2, 4: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reusable 4-class OCT classifier for downstream evaluation and counterfactual guidance.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--train-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/train_manifest.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/val_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/test_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_classifier_real_full_resnet18"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/oct_classifier_real_full_resnet18"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Stratified fraction of the train manifest to use. Supports Arunava-style low-data experiments.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use a class-balanced weighted sampler for training.")
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oct-maisi")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb:
        return None
    import wandb

    run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    return run


def plain_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        return value.as_tensor()
    return value


def classifier_label(batch_labels: torch.Tensor) -> torch.Tensor:
    labels = plain_tensor(batch_labels).long()
    mapped = torch.empty_like(labels)
    for source, target in SOURCE_TO_CLASSIFIER_LABEL.items():
        mapped[labels == source] = target
    return mapped


def stratified_fraction(records: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 1.0:
        return records
    if fraction <= 0.0:
        raise ValueError("--train-fraction must be > 0")
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    sampled = []
    for label, label_records in sorted(by_label.items()):
        pool = list(label_records)
        rng.shuffle(pool)
        count = max(1, round(len(pool) * fraction))
        sampled.extend(pool[:count])
    rng.shuffle(sampled)
    return sampled


def class_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["label"]] += 1
    return dict(sorted(counts.items()))


def make_model() -> nn.Module:
    model = resnet18(weights=None, num_classes=len(CLASS_NAMES))
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def make_loader(records: list[dict], args: argparse.Namespace, train: bool) -> DataLoader:
    transform = define_oct_image_transform(args.image_size, is_train=train, output_dtype=torch.float32, random_aug=train)
    dataset = Dataset(data=records, transform=transform)
    sampler = None
    shuffle = train
    if train and args.balanced_sampler:
        counts = class_counts(records)
        weights = [1.0 / counts[record["label"]] for record in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float | list[list[int]]]:
    confusion = confusion.cpu().long()
    total = int(confusion.sum())
    correct = int(torch.diag(confusion).sum())
    recalls = []
    precisions = []
    f1s = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    return {
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1s) / len(f1s)),
        "macro_precision": float(sum(precisions) / len(precisions)),
        "macro_recall": float(sum(recalls) / len(recalls)),
        "confusion_matrix": confusion.tolist(),
    }


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion, optimizer=None, scaler: GradScaler | None = None, amp: bool = False) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    batches = 0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for batch in loader:
        images = plain_tensor(batch["image"]).to(device, non_blocking=True)
        labels = classifier_label(batch["class_label"]).to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if train:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        preds = logits.detach().argmax(dim=1).cpu()
        for truth, pred in zip(labels.detach().cpu(), preds):
            confusion[int(truth), int(pred)] += 1
        total_loss += float(loss.detach().cpu())
        batches += 1
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "architecture": "resnet18_1ch",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT classifier training.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_records_full = load_oct_manifest(args.train_manifest, args.dataset_root)
    train_records = stratified_fraction(train_records_full, args.train_fraction, args.seed)
    val_records = load_oct_manifest(args.val_manifest, args.dataset_root)
    test_records = load_oct_manifest(args.test_manifest, args.dataset_root)
    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "class_names": CLASS_NAMES,
        "train_counts_full": class_counts(train_records_full),
        "train_counts_used": class_counts(train_records),
        "val_counts": class_counts(val_records),
        "test_counts": class_counts(test_records),
    }
    write_json(args.output_dir / "run_setup.json", summary)

    train_loader = make_loader(train_records, args, train=True)
    val_loader = make_loader(val_records, args, train=False)
    test_loader = make_loader(test_records, args, train=False)
    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.amp)
    wandb_run = setup_wandb(args, summary)

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    latest_metrics = {}
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler, amp=args.amp)
        val_metrics = run_epoch(model, val_loader, device, criterion, amp=args.amp)
        elapsed = time.perf_counter() - start
        latest_metrics = {"epoch": epoch, "epoch_sec": elapsed, "train": train_metrics, "val": val_metrics}
        write_json(args.output_dir / "latest_metrics.json", latest_metrics)
        save_checkpoint(args.model_dir / "classifier_latest.pt", model, args, epoch, latest_metrics)
        improved = val_metrics["macro_f1"] > best_macro_f1 + args.early_stop_min_delta
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            write_json(args.output_dir / "best_metrics.json", latest_metrics)
            save_checkpoint(args.model_dir / "classifier_best.pt", model, args, epoch, latest_metrics)
        else:
            epochs_without_improvement += 1
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                "train/macro_f1": train_metrics["macro_f1"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
            })
        print(f"epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}, sec={elapsed:.1f}", flush=True)
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping after {epochs_without_improvement} epochs without macro-F1 improvement", flush=True)
            break

    best_ckpt = torch.load(args.model_dir / "classifier_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, criterion, amp=args.amp)
    final_summary = {"best": best_ckpt["metrics"], "latest": latest_metrics, "test": test_metrics}
    write_json(args.output_dir / "final_summary.json", final_summary)
    if wandb_run:
        wandb_run.log({
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/balanced_accuracy": test_metrics["balanced_accuracy"],
            "test/macro_f1": test_metrics["macro_f1"],
        })
        wandb_run.finish()
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
