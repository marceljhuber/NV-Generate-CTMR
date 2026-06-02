# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset

from .oct_data import define_oct_image_transform, load_oct_manifest
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute 2D OCT VAE latents for diffusion training.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT"))
    parser.add_argument("--train-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/train_manifest.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/val_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/test_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_latents_128"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit per split.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["latent_path", "relative_path", "label", "class_label", "patient_id", "image_index", "source_split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_loader(records: list[dict], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = define_oct_image_transform(image_size=image_size, is_train=False, output_dtype=torch.float32, random_aug=False)
    dataset = Dataset(data=records, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


@torch.inference_mode()
def encode_split(args: argparse.Namespace, split: str, records: list[dict], autoencoder: torch.nn.Module, device: torch.device) -> dict:
    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(records, args.image_size, args.batch_size, args.num_workers)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    manifest_rows: list[dict] = []
    count = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        images = batch["image"].to(device)
        z_mu, _ = autoencoder.encode(images)
        z_mu = z_mu.detach().cpu().to(out_dtype)

        for item_idx in range(z_mu.shape[0]):
            relative_path = batch["relative_path"][item_idx]
            label = batch["label"][item_idx]
            patient_id = str(batch["patient_id"][item_idx])
            image_index = int(batch["image_index"][item_idx])
            source_split = batch["source_split"][item_idx]
            safe_stem = Path(relative_path).with_suffix("").as_posix().replace("/", "__")
            latent_path = split_dir / f"{safe_stem}.pt"
            class_label = int(batch["class_label"][item_idx])
            torch.save(
                {
                    "latent": z_mu[item_idx],
                    "label": label,
                    "class_label": class_label,
                    "patient_id": patient_id,
                    "image_index": image_index,
                    "relative_path": relative_path,
                    "source_split": source_split,
                },
                latent_path,
            )
            manifest_rows.append(
                {
                    "latent_path": str(latent_path.relative_to(args.output_dir)),
                    "relative_path": relative_path,
                    "label": label,
                    "class_label": class_label,
                    "patient_id": patient_id,
                    "image_index": image_index,
                    "source_split": source_split,
                }
            )
            count += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"{split}: encoded {count} images")

    manifest_path = args.output_dir / f"{split}_latents_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    return {"split": split, "num_latents": count, "manifest": str(manifest_path)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT latent precomputation.")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    autoencoder.eval()

    manifests = {"train": args.train_manifest, "val": args.val_manifest, "test": args.test_manifest}
    summary = {"checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir), "image_size": args.image_size, "dtype": args.dtype, "splits": []}
    for split in args.splits:
        records = load_oct_manifest(manifests[split], args.dataset_root)
        summary["splits"].append(encode_split(args, split, records, autoencoder, device))

    with (args.output_dir / "latents_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
