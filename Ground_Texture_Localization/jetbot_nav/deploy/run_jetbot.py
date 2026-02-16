"""Runtime loop for JetBot deployment."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from controller import JetBotController, MotorCommand
from inference import NavigationInference


class MotorDriver:
    """Simple motor abstraction. Replace internals with jetbot.Robot on device."""

    def __init__(self) -> None:
        self.last_command = MotorCommand(0.0, 0.0, "idle")

    def apply(self, cmd: MotorCommand) -> None:
        # On Jetson replace with:
        # from jetbot import Robot
        # self.robot = Robot()
        # self.robot.set_motors(cmd.left_pwm, cmd.right_pwm)
        self.last_command = cmd

    def stop(self) -> None:
        self.apply(MotorCommand(0.0, 0.0, "stop"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JetBot navigation loop.")
    parser.add_argument("--model", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--class-map", type=Path, default=Path("class_mapping.json"))
    parser.add_argument("--preprocess", type=Path, default=Path("preprocess_config.json"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("logs/runtime.jsonl"))
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--obstacle-threshold", type=float, default=0.6)
    parser.add_argument("--steering-threshold", type=float, default=0.4)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--majority-window", type=int, default=5)
    return parser.parse_args()


def log_telemetry(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    infer = NavigationInference(
        model_path=args.model,
        class_map_path=args.class_map,
        preprocess_path=args.preprocess,
        obstacle_threshold=args.obstacle_threshold,
        steering_threshold=args.steering_threshold,
        ema_alpha=args.ema_alpha,
        majority_window=args.majority_window,
    )
    controller = JetBotController(obstacle_threshold=args.obstacle_threshold)
    motors = MotorDriver()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    period_s = 1.0 / max(args.loop_hz, 0.5)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame capture failed")

            pred = infer.predict(frame)
            cmd = controller.decide(
                steering_pred=str(pred["steering_pred"]),
                obstacle_probability=float(pred["obstacle_prob_smoothed"]),
            )
            motors.apply(cmd)

            telemetry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "steering_pred": pred["steering_pred"],
                "obstacle_pred": pred["obstacle_pred"],
                "confidence": {
                    "steering": pred["steering_confidence"],
                    "obstacle": pred["obstacle_confidence"],
                },
                "motor_cmd": {
                    "left_pwm": cmd.left_pwm,
                    "right_pwm": cmd.right_pwm,
                    "action": cmd.action,
                },
            }
            log_telemetry(args.log_file, telemetry)

            elapsed = time.time() - t0
            sleep_s = max(0.0, period_s - elapsed)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Keyboard interrupt: shutting down motors safely.")
    except Exception as exc:
        print(f"Runtime error: {exc}")
        raise
    finally:
        motors.stop()
        cap.release()


if __name__ == "__main__":
    main()
