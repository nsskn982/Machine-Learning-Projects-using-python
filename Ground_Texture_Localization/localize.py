"""Run online texture classification + particle filter localization."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from texture_features import extract_features


@dataclass
class MapConfig:
    cell_size_m: float
    grid: list[list[str]]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0


@dataclass
class Particle:
    x: float
    y: float
    weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Texture-based localization demo.")
    parser.add_argument("--model", required=True, help="Path to trained model (joblib).")
    parser.add_argument("--map", dest="map_path", required=True, help="Path to map JSON.")
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--particles", type=int, default=200, help="Number of particles.")
    parser.add_argument("--speed", type=float, default=0.1, help="Assumed speed in m/s.")
    parser.add_argument("--heading", type=float, default=0.0, help="Heading in radians (0 = +x).")
    return parser.parse_args()


def load_map(path: Path) -> MapConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return MapConfig(cell_size_m=float(data["cell_size_m"]), grid=data["grid"])


def init_particles(count: int, map_cfg: MapConfig) -> list[Particle]:
    particles = []
    for _ in range(count):
        # Sample particles uniformly across the map area.
        x = random.uniform(0, map_cfg.width * map_cfg.cell_size_m)
        y = random.uniform(0, map_cfg.height * map_cfg.cell_size_m)
        particles.append(Particle(x=x, y=y, weight=1.0 / count))
    return particles


def motion_update(
    particles: list[Particle],
    dt: float,
    speed: float,
    heading: float,
    map_cfg: MapConfig,
) -> None:
    for particle in particles:
        # Add Gaussian noise to simulate uncertain motion.
        noisy_speed = random.gauss(speed, 0.02)
        noisy_heading = random.gauss(heading, 0.1)

        particle.x += noisy_speed * dt * math.cos(noisy_heading)
        particle.y += noisy_speed * dt * math.sin(noisy_heading)

        # Keep particles inside the map bounds.
        particle.x = min(max(particle.x, 0.0), map_cfg.width * map_cfg.cell_size_m)
        particle.y = min(max(particle.y, 0.0), map_cfg.height * map_cfg.cell_size_m)


def measurement_update(
    particles: list[Particle],
    predicted_label: str,
    map_cfg: MapConfig,
) -> None:
    total_weight = 0.0
    for particle in particles:
        col = min(int(particle.x / map_cfg.cell_size_m), map_cfg.width - 1)
        row = min(int(particle.y / map_cfg.cell_size_m), map_cfg.height - 1)
        cell_label = map_cfg.grid[row][col]

        # Match = higher likelihood; mismatch = lower likelihood.
        particle.weight = 0.9 if cell_label == predicted_label else 0.1
        total_weight += particle.weight

    # Normalize weights.
    if total_weight == 0.0:
        total_weight = 1.0
    for particle in particles:
        particle.weight /= total_weight


def resample_particles(particles: list[Particle]) -> list[Particle]:
    weights = np.array([p.weight for p in particles])
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return [Particle(x=particles[i].x, y=particles[i].y, weight=1.0 / len(particles)) for i in indices]


def estimate_position(particles: list[Particle]) -> tuple[float, float]:
    x = sum(p.x * p.weight for p in particles)
    y = sum(p.y * p.weight for p in particles)
    return x, y


def main() -> None:
    args = parse_args()

    model_data = joblib.load(args.model)
    model = model_data["model"]

    map_cfg = load_map(Path(args.map_path))
    particles = init_particles(args.particles, map_cfg)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check device index and permissions.")

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame; stopping.")
            break

        # Run the classifier on the current frame.
        features = extract_features(frame).reshape(1, -1)
        predicted_label = model.predict(features)[0]

        # Particle filter updates.
        now = time.time()
        dt = now - last_time
        last_time = now

        motion_update(particles, dt, args.speed, args.heading, map_cfg)
        measurement_update(particles, predicted_label, map_cfg)
        particles = resample_particles(particles)
        est_x, est_y = estimate_position(particles)

        cv2.putText(
            frame,
            f"Texture: {predicted_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Est (m): x={est_x:.2f}, y={est_y:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Localization", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
