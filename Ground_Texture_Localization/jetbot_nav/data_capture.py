"""Data capture utility for JetBot ground-texture navigation datasets.

This script reads frames from a CSI camera, allows interactive labeling,
and persists both image files and labels CSV records.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2


STEERING_CLASSES = ["left", "straight", "right", "stop"]
OBSTACLE_CLASSES = ["clear", "obstacle"]


def gstreamer_pipeline(
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build a CSI-camera pipeline string for Jetson/JetBot devices."""
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


@dataclass
class LabelState:
    steering_label: int = 1  # default: straight
    obstacle_label: int = 0  # default: clear


class OptionalGamepad:
    """Optional joystick/gamepad wrapper using pygame when available."""

    def __init__(self) -> None:
        self.enabled = False
        self.joystick = None
        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.enabled = True
                self._pygame = pygame
        except Exception:
            self.enabled = False

    def poll(self) -> Optional[Tuple[int, int]]:
        """Poll gamepad and return (steering_label, obstacle_label) when mapped."""
        if not self.enabled:
            return None
        pygame = self._pygame
        pygame.event.pump()

        steering = None
        obstacle = None

        # D-pad horizontal on common controllers.
        try:
            hat_x, _ = self.joystick.get_hat(0)
            if hat_x < 0:
                steering = 0
            elif hat_x > 0:
                steering = 2
        except Exception:
            pass

        # Left analog horizontal fallback.
        axis = self.joystick.get_axis(0)
        if axis < -0.5:
            steering = 0
        elif axis > 0.5:
            steering = 2
        elif -0.2 < axis < 0.2:
            steering = 1

        # A/B button mapping for obstacle presence.
        if self.joystick.get_numbuttons() > 1:
            if self.joystick.get_button(0):
                obstacle = 0
            if self.joystick.get_button(1):
                obstacle = 1

        if steering is None and obstacle is None:
            return None
        return (
            steering if steering is not None else -1,
            obstacle if obstacle is not None else -1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture JetBot navigation dataset samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Dataset root directory")
    parser.add_argument("--camera-index", type=int, default=-1, help="OpenCV camera index (used if --use-csi is off)")
    parser.add_argument("--use-csi", action="store_true", help="Use CSI gstreamer pipeline")
    parser.add_argument("--width", type=int, default=640, help="Display width")
    parser.add_argument("--height", type=int, default=360, help="Display height")
    return parser.parse_args()


def ensure_dataset_files(output_dir: Path) -> Tuple[Path, Path]:
    images_dir = output_dir / "images"
    labels_csv = output_dir / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        with labels_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["image_path", "steering_label", "obstacle_label", "timestamp"])
    return images_dir, labels_csv


def key_to_labels(key: int) -> Dict[str, int]:
    """Keyboard labeling map.

    Steering classes:
      a -> left (0), s -> straight (1), d -> right (2), x -> stop (3)
    Obstacle classes:
      c -> clear (0), o -> obstacle (1)
    """
    mapping: Dict[str, int] = {}
    if key == ord("a"):
        mapping["steering"] = 0
    elif key == ord("s"):
        mapping["steering"] = 1
    elif key == ord("d"):
        mapping["steering"] = 2
    elif key == ord("x"):
        mapping["steering"] = 3
    elif key == ord("c"):
        mapping["obstacle"] = 0
    elif key == ord("o"):
        mapping["obstacle"] = 1
    return mapping


def main() -> None:
    args = parse_args()
    images_dir, labels_csv = ensure_dataset_files(args.output_dir)

    # Stage 1: Camera read setup.
    if args.use_csi:
        cap = cv2.VideoCapture(gstreamer_pipeline(display_width=args.width, display_height=args.height), cv2.CAP_GSTREAMER)
    else:
        camera_index = args.camera_index if args.camera_index >= 0 else 0
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Check CSI pipeline or camera index.")

    print("Controls: a/s/d/x=steering, c/o=obstacle, space=save sample, q=quit")
    label_state = LabelState()
    gamepad = OptionalGamepad()

    with labels_csv.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)

        while True:
            # Stage 1: Camera read.
            ok, frame = cap.read()
            if not ok:
                print("Warning: camera read failed, retrying...")
                time.sleep(0.05)
                continue

            # Stage 2: Label assignment from optional gamepad input.
            gamepad_update = gamepad.poll()
            if gamepad_update:
                steering, obstacle = gamepad_update
                if steering >= 0:
                    label_state.steering_label = steering
                if obstacle >= 0:
                    label_state.obstacle_label = obstacle

            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Steering: {STEERING_CLASSES[label_state.steering_label]} ({label_state.steering_label})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                f"Obstacle: {OBSTACLE_CLASSES[label_state.obstacle_label]} ({label_state.obstacle_label})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                "Press SPACE to save sample | q to quit",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow("jetbot_nav_capture", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            mapping = key_to_labels(key)
            if "steering" in mapping:
                label_state.steering_label = mapping["steering"]
            if "obstacle" in mapping:
                label_state.obstacle_label = mapping["obstacle"]

            if key == ord(" "):
                # Stage 3: Persistence (save image + append CSV row).
                timestamp = datetime.utcnow().isoformat()
                image_name = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), frame)

                relative_path = image_path.relative_to(args.output_dir)
                writer.writerow(
                    [
                        str(relative_path).replace("\\", "/"),
                        label_state.steering_label,
                        label_state.obstacle_label,
                        timestamp,
                    ]
                )
                fp.flush()
                print(
                    f"Saved {relative_path} | steering={label_state.steering_label} "
                    f"obstacle={label_state.obstacle_label}"
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
