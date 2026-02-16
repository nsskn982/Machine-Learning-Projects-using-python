"""Training script for JetBot multi-task navigation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

from dataset import JetBotNavDataset
from model import MultiTaskJetBotNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task JetBot navigation model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Dataset root (contains labels.csv, images/)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    return parser.parse_args()


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    epochs = int(cfg["epochs"])
    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    image_size = int(cfg["image_size"])
    val_split = float(cfg.get("val_split", 0.2))
    num_workers = int(cfg.get("num_workers", 2))

    dataset_root = Path(args.dataset_root)
    csv_path = dataset_root / "labels.csv"

    full_dataset = JetBotNavDataset(
        csv_path=str(csv_path),
        image_root=str(dataset_root),
        image_size=image_size,
        train=True,
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Ensure deterministic eval transforms for validation subset.
    val_dataset.dataset.transform = JetBotNavDataset(
        csv_path=str(csv_path),
        image_root=str(dataset_root),
        image_size=image_size,
        train=False,
    ).transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    steering_classes = cfg.get("steering_classes", ["left", "straight", "right", "stop"])
    obstacle_classes = cfg.get("obstacle_classes", ["clear", "obstacle"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskJetBotNet(
        num_steering_classes=len(steering_classes),
        num_obstacle_classes=len(obstacle_classes),
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifacts_dir / "best_model.pt"
    class_map_path = artifacts_dir / "class_mapping.json"

    with class_map_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "steering": {idx: name for idx, name in enumerate(steering_classes)},
                "obstacle": {idx: name for idx, name in enumerate(obstacle_classes)},
            },
            fp,
            indent=2,
        )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, steering_target, obstacle_target in train_loader:
            images = images.to(device)
            steering_target = steering_target.to(device)
            obstacle_target = obstacle_target.to(device)

            steering_logits, obstacle_logits = model(images)
            steering_ce = criterion(steering_logits, steering_target)
            obstacle_ce = criterion(obstacle_logits, obstacle_target)
            loss = alpha * steering_ce + beta * obstacle_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= max(1, train_size)

        model.eval()
        val_loss = 0.0
        steering_acc_sum = 0.0
        obstacle_acc_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, steering_target, obstacle_target in val_loader:
                images = images.to(device)
                steering_target = steering_target.to(device)
                obstacle_target = obstacle_target.to(device)

                steering_logits, obstacle_logits = model(images)
                steering_ce = criterion(steering_logits, steering_target)
                obstacle_ce = criterion(obstacle_logits, obstacle_target)
                loss = alpha * steering_ce + beta * obstacle_ce

                val_loss += loss.item() * images.size(0)
                steering_acc_sum += compute_accuracy(steering_logits, steering_target)
                obstacle_acc_sum += compute_accuracy(obstacle_logits, obstacle_target)
                num_batches += 1

        val_loss /= max(1, val_size)
        steering_acc = steering_acc_sum / max(1, num_batches)
        obstacle_acc = obstacle_acc_sum / max(1, num_batches)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"steering_acc={steering_acc:.4f} | obstacle_acc={obstacle_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "backbone": args.backbone,
                },
                best_model_path,
            )
            print(f"Saved best checkpoint to {best_model_path}")


if __name__ == "__main__":
    main()
