"""Multi-task navigation model for JetBot."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskJetBotNet(nn.Module):
    """Backbone + two heads for steering and obstacle classification."""

    def __init__(
        self,
        num_steering_classes: int,
        num_obstacle_classes: int,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = base_model.classifier[0].in_features
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = base_model.fc.in_features
        else:
            raise ValueError("backbone must be 'mobilenet_v3_small' or 'resnet18'")

        # Shared representation: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        self.shared_dropout = nn.Dropout(p=0.2)

        # Head 1: steering logits with shape [B, num_steering_classes]
        self.steering_head = nn.Linear(feature_dim, num_steering_classes)

        # Head 2: obstacle logits with shape [B, num_obstacle_classes]
        self.obstacle_head = nn.Linear(feature_dim, num_obstacle_classes)

    def forward(self, x: torch.Tensor):
        # x shape: [B, 3, H, W]
        feats = self.feature_extractor(x)
        pooled = self.pool(feats)
        flat = torch.flatten(pooled, 1)
        shared = self.shared_dropout(flat)

        steering_logits = self.steering_head(shared)
        obstacle_logits = self.obstacle_head(shared)

        return steering_logits, obstacle_logits
