"""Inference utilities for JetBot ground-texture navigation deployment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms


DEFAULT_PREPROCESS = {
    "image_size": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class NavigationInference:
    """Loads trained model artifacts and provides smoothed predictions."""

    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        preprocess_path: Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        obstacle_threshold: float = 0.6,
        steering_threshold: float = 0.4,
        ema_alpha: float = 0.4,
        majority_window: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.class_map = self._load_class_map(class_map_path)
        self.transforms = self._build_transform(preprocess_path)

        self.obstacle_threshold = obstacle_threshold
        self.steering_threshold = steering_threshold
        self.ema_alpha = ema_alpha
        self.majority_window = max(1, majority_window)

        self._ema_obstacle: float | None = None
        self._majority_buffer: Deque[str] = deque(maxlen=self.majority_window)

    def _load_model(self, model_path: Path) -> nn.Module:
        try:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model
        except RuntimeError:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, nn.Module):
                checkpoint.eval()
                return checkpoint
            raise RuntimeError(
                "best_model.pt is not a TorchScript module. Provide a scripted/traced model for deployment."
            )

    @staticmethod
    def _load_class_map(class_map_path: Path) -> Dict[str, List[str]]:
        with class_map_path.open("r", encoding="utf-8") as fh:
            class_map = json.load(fh)

        if "steering" not in class_map:
            raise ValueError("Class mapping JSON must include a 'steering' list.")
        class_map.setdefault("obstacle", ["clear", "obstacle"])
        return class_map

    @staticmethod
    def _build_transform(preprocess_path: Path | None) -> transforms.Compose:
        config = DEFAULT_PREPROCESS.copy()
        if preprocess_path and preprocess_path.exists():
            with preprocess_path.open("r", encoding="utf-8") as fh:
                file_cfg = json.load(fh)
            config.update(file_cfg)

        width, height = config["image_size"]
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )

    def _smooth_obstacle(self, prob: float) -> float:
        if self._ema_obstacle is None:
            self._ema_obstacle = prob
        else:
            self._ema_obstacle = (self.ema_alpha * prob) + ((1.0 - self.ema_alpha) * self._ema_obstacle)
        return self._ema_obstacle

    def _smooth_steering(self, label: str) -> str:
        self._majority_buffer.append(label)
        counts = Counter(self._majority_buffer)
        return counts.most_common(1)[0][0]

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, float | str | bool]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.transforms(rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(x)

        if isinstance(outputs, dict):
            steering_logits = outputs["steering"]
            obstacle_logits = outputs["obstacle"]
        elif isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            steering_logits, obstacle_logits = outputs
        else:
            raise TypeError("Model output must be a dict with steering/obstacle or tuple(steering, obstacle).")

        steering_probs = torch.softmax(steering_logits, dim=-1).squeeze(0)
        obstacle_probs = torch.softmax(obstacle_logits, dim=-1).squeeze(0)

        steering_confidence, steering_idx = torch.max(steering_probs, dim=-1)
        obstacle_confidence, obstacle_idx = torch.max(obstacle_probs, dim=-1)

        steering_label = self.class_map["steering"][int(steering_idx)]
        obstacle_label = self.class_map["obstacle"][int(obstacle_idx)]

        smoothed_obstacle = self._smooth_obstacle(float(obstacle_probs[-1]))
        smoothed_steering = self._smooth_steering(steering_label)

        steering_low_conf = float(steering_confidence) < self.steering_threshold
        obstacle_detected = smoothed_obstacle >= self.obstacle_threshold

        return {
            "steering_pred": smoothed_steering,
            "steering_raw": steering_label,
            "steering_confidence": float(steering_confidence),
            "steering_low_conf": steering_low_conf,
            "obstacle_pred": obstacle_label,
            "obstacle_confidence": float(obstacle_confidence),
            "obstacle_prob": float(obstacle_probs[-1]),
            "obstacle_prob_smoothed": smoothed_obstacle,
            "obstacle_detected": obstacle_detected,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot inference on webcam frame.")
    parser.add_argument("--model", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--class-map", type=Path, default=Path("class_mapping.json"))
    parser.add_argument("--preprocess", type=Path, default=Path("preprocess_config.json"))
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer = NavigationInference(args.model, args.class_map, args.preprocess)

    cap = cv2.VideoCapture(args.camera_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame from camera.")

    print(json.dumps(infer.predict(frame), indent=2))


if __name__ == "__main__":
    main()
