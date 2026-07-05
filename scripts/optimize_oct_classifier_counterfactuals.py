# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import Dataset
from torchvision.utils import make_grid, save_image

from .oct_data import define_oct_image_transform, load_oct_manifest
from .train_oct_classifier import CLASS_NAMES, make_model, plain_tensor
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize OCT VAE latents toward classifier target classes while preserving image identity.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/test_manifest.csv"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("models/oct_classifier_real_full_resnet18/classifier_best.pt"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_classifier_guided_counterfactuals_smoke"))
    parser.add_argument("--source-class", choices=CLASS_NAMES, default="NORMAL")
    parser.add_argument("--target-class", choices=CLASS_NAMES, default="DME")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda-image", type=float, default=8.0)
    parser.add_argument("--lambda-latent", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2029)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def total_variation(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean() + (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()


def select_records(args: argparse.Namespace) -> list[dict]:
    records = load_oct_manifest(args.manifest, args.dataset_root)
    candidates = [record for record in records if record["label"] == args.source_class]
    if len(candidates) < args.num_images:
        raise ValueError(f"Requested {args.num_images} {args.source_class} images, found {len(candidates)}")
    rng = random.Random(args.seed)
    return rng.sample(candidates, args.num_images)


def load_images(records: list[dict], image_size: int) -> torch.Tensor:
    transform = define_oct_image_transform(image_size, is_train=False, output_dtype=torch.float32, random_aug=False)
    dataset = Dataset(data=records, transform=transform)
    return torch.stack([plain_tensor(dataset[index]["image"]) for index in range(len(dataset))])


def classifier_probs(classifier: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    return torch.softmax(classifier(images), dim=1)


def load_models(args: argparse.Namespace, device: torch.device):
    classifier = make_model().to(device)
    ckpt = torch.load(args.classifier_checkpoint, map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)

    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)
    return classifier, autoencoder, ckpt


def make_review_grid(originals: torch.Tensor, recon: torch.Tensor, edited: torch.Tensor, trajectory: list[torch.Tensor]) -> torch.Tensor:
    diff = (edited - recon).abs()
    rows = [originals.cpu(), recon.cpu()]
    rows.extend([item.cpu() for item in trajectory])
    rows.extend([edited.cpu(), diff.cpu()])
    return make_grid(torch.cat(rows, dim=0), nrow=originals.shape[0], padding=2)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for classifier-guided latent counterfactual optimization.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    records = select_records(args)
    images = load_images(records, args.image_size).to(device)
    classifier, autoencoder, classifier_ckpt = load_models(args, device)
    target_idx = CLASS_NAMES.index(args.target_class)
    target = torch.full((images.shape[0],), target_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        z0, _ = autoencoder.encode(images)
        recon = autoencoder.decode(z0).detach().clamp(0, 1)
        original_probs = classifier_probs(classifier, images).detach().cpu()
        recon_probs = classifier_probs(classifier, recon).detach().cpu()

    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=args.lr)
    trajectory: list[torch.Tensor] = []
    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        decoded = autoencoder.decode(z).clamp(0, 1)
        logits = classifier(decoded)
        class_loss = F.cross_entropy(logits, target)
        image_loss = F.l1_loss(decoded, recon)
        latent_loss = F.mse_loss(z, z0)
        tv_loss = total_variation(decoded - recon)
        loss = class_loss + args.lambda_image * image_loss + args.lambda_latent * latent_loss + args.lambda_tv * tv_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            target_prob = float(probs[:, target_idx].mean().detach().cpu())
            if step % args.save_every == 0 or step == args.steps:
                trajectory.append(decoded.detach().cpu())
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "class_loss": float(class_loss.detach().cpu()),
                "image_l1": float(image_loss.detach().cpu()),
                "latent_mse": float(latent_loss.detach().cpu()),
                "tv": float(tv_loss.detach().cpu()),
                "mean_target_probability": target_prob,
            })
        if step % 25 == 0 or step == 1:
            print(f"step {step}: loss={history[-1]['loss']:.4f}, target_prob={history[-1]['mean_target_probability']:.4f}, image_l1={history[-1]['image_l1']:.4f}", flush=True)

    with torch.no_grad():
        edited = autoencoder.decode(z).detach().clamp(0, 1)
        edited_probs = classifier_probs(classifier, edited).detach().cpu()
    save_image(make_review_grid(images.detach().cpu(), recon.detach().cpu(), edited.cpu(), trajectory), args.output_dir / "counterfactual_review_grid.png")
    save_image(make_grid(images.detach().cpu(), nrow=args.num_images, padding=2), args.output_dir / "originals.png")
    save_image(make_grid(recon.detach().cpu(), nrow=args.num_images, padding=2), args.output_dir / "reconstructions.png")
    save_image(make_grid(edited.cpu(), nrow=args.num_images, padding=2), args.output_dir / "edited_counterfactuals.png")
    save_image(make_grid((edited.cpu() - recon.detach().cpu()).abs(), nrow=args.num_images, padding=2), args.output_dir / "absolute_difference.png")

    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "classifier_epoch": classifier_ckpt.get("epoch"),
        "classifier_metrics": classifier_ckpt.get("metrics"),
        "records": records,
        "class_names": CLASS_NAMES,
        "original_probs": original_probs.tolist(),
        "reconstruction_probs": recon_probs.tolist(),
        "edited_probs": edited_probs.tolist(),
        "history": history,
        "row_order": ["original", "reconstruction"] + [f"trajectory_step_{min((idx + 1) * args.save_every, args.steps)}" for idx in range(len(trajectory))] + ["final_edit", "abs_difference"],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(args.output_dir), "final_mean_target_probability": float(edited_probs[:, target_idx].mean())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
