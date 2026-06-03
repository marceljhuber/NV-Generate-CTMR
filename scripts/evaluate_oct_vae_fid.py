# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.metrics.fid import FIDMetric
from torchvision.models import Inception_V3_Weights, inception_v3

from .oct_data import OCT_LABEL_TO_ID, define_oct_image_transform, load_oct_manifest
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID between OCT images and VAE reconstructions.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/val_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_vae_fid"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-percent", type=float, default=1.0, help="Percent of each class to evaluate, e.g. 1.0 for 1 percent.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--save-recon-grid", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def plain_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        return value.as_tensor()
    return value


def stratified_sample(records: list[dict], sample_percent: float, seed: int, max_per_class: int | None) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    sampled = []
    for label in sorted(OCT_LABEL_TO_ID):
        if label == "unknown" or label not in by_label:
            continue
        class_records = list(by_label[label])
        rng.shuffle(class_records)
        n = max(1, round(len(class_records) * sample_percent / 100.0))
        if max_per_class is not None:
            n = min(n, max_per_class)
        sampled.extend(class_records[:n])
    rng.shuffle(sampled)
    return sampled


class InceptionFeatures(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model
        self.mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().clamp(0, 1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        mean = self.mean.to(images.device)
        std = self.std.to(images.device)
        return self.model((images - mean) / std)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT VAE FID evaluation.")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_oct_manifest(args.manifest, args.dataset_root)
    sampled_records = stratified_sample(records, args.sample_percent, args.seed, args.max_per_class)
    if len(sampled_records) < 2:
        raise ValueError("Need at least two sampled records for FID.")

    transform = define_oct_image_transform(image_size=args.image_size, is_train=False, output_dtype=torch.float32, random_aug=False)
    dataset = Dataset(data=sampled_records, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    autoencoder.eval()

    feature_net = InceptionFeatures().to(device).eval()
    real_features = []
    recon_features = []
    counts: dict[str, int] = defaultdict(int)

    for batch_idx, batch in enumerate(loader):
        images = plain_tensor(batch["image"]).to(device, non_blocking=True).contiguous()
        reconstruction, _, _ = autoencoder(images)
        real_features.append(feature_net(images).detach().cpu())
        recon_features.append(feature_net(reconstruction).detach().cpu())
        for label in batch["label"]:
            counts[str(label)] += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"processed {(batch_idx + 1) * args.batch_size} sampled images", flush=True)

    real_features_tensor = torch.cat(real_features, dim=0)
    recon_features_tensor = torch.cat(recon_features, dim=0)
    fid_metric = FIDMetric()
    fid = float(fid_metric(real_features_tensor, recon_features_tensor))

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root),
        "image_size": args.image_size,
        "sample_percent": args.sample_percent,
        "num_images": int(real_features_tensor.shape[0]),
        "counts_by_class": dict(sorted(counts.items())),
        "feature_model": "torchvision.inception_v3_imagenet_fc_features",
        "fid_reconstruction": fid,
    }
    out_path = args.output_dir / f"vae_fid_{args.sample_percent:g}pct.json"
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
