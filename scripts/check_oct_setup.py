# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OCT-MAISI dependencies, manifests, transforms, and model config.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT"))
    parser.add_argument("--train-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/train_manifest.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/val_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/test_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/config_maisi_vae_train_oct_128.json"))
    parser.add_argument("--check-wandb", action="store_true", help="Also require W&B to be importable.")
    return parser.parse_args()


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing dependency: {module_name}")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()

    require_module("torch")
    require_module("monai")
    if args.check_wandb:
        require_module("wandb")

    import argparse as argparse_module

    import torch

    from .oct_data import define_oct_image_transform, load_oct_manifest
    from .utils import define_instance

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    train_records = load_oct_manifest(args.train_manifest, args.dataset_root)
    val_records = load_oct_manifest(args.val_manifest, args.dataset_root)
    test_records = load_oct_manifest(args.test_manifest, args.dataset_root)
    print(f"train records: {len(train_records)}")
    print(f"val records: {len(val_records)}")
    print(f"test records: {len(test_records)}")

    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        if not records:
            raise RuntimeError(f"{name} manifest is empty")
        first_image = Path(records[0]["image"])
        if not first_image.is_file():
            raise FileNotFoundError(f"First {name} image not found: {first_image}")

    train_config = load_json(args.train_config)
    network_config = load_json(args.network_config)
    transform = define_oct_image_transform(image_size=train_config["data_option"]["image_size"], is_train=False, output_dtype=torch.float32)
    sample = transform(train_records[0])
    image = sample["image"]
    print(f"sample image shape: {tuple(image.shape)} dtype={image.dtype} min={float(image.min()):.4f} max={float(image.max()):.4f}")
    if tuple(image.shape) != (1, train_config["data_option"]["image_size"], train_config["data_option"]["image_size"]):
        raise RuntimeError(f"Unexpected transformed image shape: {tuple(image.shape)}")

    config_ns = argparse_module.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def")
    total_params = sum(param.numel() for param in autoencoder.parameters())
    print(f"autoencoder parameters: {total_params:,}")
    print("OCT setup check passed.")


if __name__ == "__main__":
    main()
