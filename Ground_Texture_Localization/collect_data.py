"""Capture labeled texture images from a live camera feed."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect texture images for training.")
    parser.add_argument("--label", required=True, help="Texture label (e.g., tile, carpet).")
    parser.add_argument("--out", required=True, help="Output directory for images.")
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum number of images to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the camera. Use --device to select CSI/USB index on Jetson/Pi.
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check device index and permissions.")

    print("Press 'q' to stop capturing.")
    count = 0
    last_time = 0.0

    while count < args.max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame; stopping.")
            break

        cv2.putText(
            frame,
            f"Label: {args.label} | Saved: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture", frame)

        # Save frames at a fixed interval to avoid near-duplicates and reduce storage.
        now = time.time()
        if now - last_time >= args.interval:
            filename = output_dir / f"{args.label}_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            last_time = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
