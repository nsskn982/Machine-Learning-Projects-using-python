"""Dataset utilities for JetBot navigation multi-task training."""

from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class JetBotNavDataset(Dataset):
    """Loads image paths and steering/obstacle labels from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        self.records = pd.read_csv(csv_path)
        self.image_root = image_root

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        image_path = f"{self.image_root}/{row['image_path']}"

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        steering_target = torch.tensor(int(row["steering_label"]), dtype=torch.long)
        obstacle_target = torch.tensor(int(row["obstacle_label"]), dtype=torch.long)

        return image_tensor, steering_target, obstacle_target
