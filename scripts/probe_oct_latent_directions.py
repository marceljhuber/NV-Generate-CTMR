# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .oct_data import OCT_ID_TO_LABEL
from .train_oct_diffusion import load_latent_split
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe interpretable directions in OCT VAE latent space with PCA and class-mean directions.")
    parser.add_argument("--latents-dir", type=Path, default=Path("outputs/oct_latents_256_adv_overnight"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_latent_direction_probe_256"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-pca-samples", type=int, default=4096)
    parser.add_argument("--num-base-images", type=int, default=8)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-3.0, -1.5, 0.0, 1.5, 3.0])
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def sample_indices(labels: torch.Tensor, num_items: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    indices = list(range(labels.shape[0]))
    rng.shuffle(indices)
    return torch.tensor(indices[: min(num_items, len(indices))], dtype=torch.long)


def balanced_base_indices(labels: torch.Tensor, num_images: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(idx)
    chosen = []
    per_class = max(1, num_images // max(len(by_label), 1))
    for label in sorted(by_label):
        pool = list(by_label[label])
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    if len(chosen) < num_images:
        rest = [idx for idx in range(labels.shape[0]) if idx not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: num_images - len(chosen)])
    return torch.tensor(chosen[:num_images], dtype=torch.long)


def latent_stats(flat: torch.Tensor) -> dict[str, float]:
    values = flat.detach().float().cpu()
    q_values = values.flatten()
    if q_values.numel() > 2_000_000:
        q_values = q_values[torch.randperm(q_values.numel())[:2_000_000]]
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p01": float(torch.quantile(q_values, 0.01)),
        "p99": float(torch.quantile(q_values, 0.99)),
    }


@torch.inference_mode()
def decode(autoencoder: torch.nn.Module, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    return autoencoder.decode(latents.to(device)).detach().float().cpu().clamp(0, 1)


def make_direction_grid(autoencoder: torch.nn.Module, base_latents: torch.Tensor, direction: torch.Tensor, alphas: list[float], sigma: float, device: torch.device) -> torch.Tensor:
    rows = []
    direction = direction.view_as(base_latents[:1])
    for base in base_latents:
        variants = torch.cat([(base.unsqueeze(0) + alpha * sigma * direction) for alpha in alphas], dim=0)
        rows.append(decode(autoencoder, variants, device))
    return torch.cat(rows, dim=0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoding OCT latent direction probes.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    autoencoder.eval()

    latents, labels = load_latent_split(args.latents_dir, args.split)
    latents = latents.float()
    labels = labels.long()
    flat = latents.flatten(1)
    pca_indices = sample_indices(labels, args.num_pca_samples, args.seed)
    pca_flat = flat[pca_indices]
    mean = pca_flat.mean(dim=0, keepdim=True)
    centered = pca_flat - mean
    _, singular_values, components = torch.pca_lowrank(centered, q=args.num_components, center=False)
    component_sigmas = singular_values / max((pca_flat.shape[0] - 1) ** 0.5, 1.0)
    base_indices = balanced_base_indices(labels, args.num_base_images, args.seed + 1)
    base_latents = latents[base_indices]

    save_image(make_grid(decode(autoencoder, base_latents, device), nrow=args.num_base_images, padding=2), args.output_dir / "base_reconstructions.png")
    pca_summary = []
    for comp_idx in range(args.num_components):
        direction = components[:, comp_idx]
        sigma = float(component_sigmas[comp_idx])
        grid = make_direction_grid(autoencoder, base_latents, direction, args.alphas, sigma, device)
        out_path = args.output_dir / f"pca_component_{comp_idx:02d}.png"
        save_image(make_grid(grid, nrow=len(args.alphas), padding=2), out_path)
        pca_summary.append({"component": comp_idx, "sigma": sigma, "path": str(out_path), "explained_proxy": float(singular_values[comp_idx] ** 2 / torch.sum(singular_values ** 2))})

    class_mean_summary = []
    class_means = {int(label): flat[labels == label].mean(dim=0) for label in sorted(set(labels.tolist()))}
    normal_label = 4
    if normal_label in class_means:
        for target_label, target_mean in class_means.items():
            if target_label == normal_label:
                continue
            direction = target_mean - class_means[normal_label]
            sigma = float(direction.norm())
            if sigma <= 0:
                continue
            direction = direction / sigma
            grid = make_direction_grid(autoencoder, base_latents, direction, args.alphas, sigma, device)
            label_name = OCT_ID_TO_LABEL.get(target_label, str(target_label)).lower()
            out_path = args.output_dir / f"class_mean_normal_to_{label_name}.png"
            save_image(make_grid(grid, nrow=len(args.alphas), padding=2), out_path)
            class_mean_summary.append({"from": "NORMAL", "to": OCT_ID_TO_LABEL.get(target_label, str(target_label)), "sigma": sigma, "path": str(out_path)})

    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_latents": int(latents.shape[0]),
        "latent_shape": list(latents.shape[1:]),
        "labels": {OCT_ID_TO_LABEL.get(int(label), str(label)): int((labels == label).sum()) for label in sorted(set(labels.tolist()))},
        "base_indices": base_indices.tolist(),
        "base_labels": [OCT_ID_TO_LABEL.get(int(labels[idx]), str(int(labels[idx]))) for idx in base_indices.tolist()],
        "latent_stats": latent_stats(flat),
        "pca": pca_summary,
        "class_mean_directions": class_mean_summary,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
