"""Train a texture classifier from labeled images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from texture_features import extract_features


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a texture classifier.")
    parser.add_argument("--data-root", required=True, help="Root directory containing label subfolders.")
    parser.add_argument("--model-out", required=True, help="Path to save the trained model.")
    return parser.parse_args()


def load_dataset(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for label_dir in sorted(data_root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            features.append(extract_features(image))
            labels.append(label)

    if not features:
        raise RuntimeError("No images found. Check your data directory.")

    return np.vstack(features), np.array(labels)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        raise RuntimeError(f"Data directory not found: {data_root}")

    x_data, y_data = load_dataset(data_root)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, gamma="scale")),
        ]
    )

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": sorted(set(y_data))}, output_path)

    print(f"Model saved to {output_path}")
    print(f"Validation accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
