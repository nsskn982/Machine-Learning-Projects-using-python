"""Control logic for converting model predictions to JetBot motor commands."""

from __future__ import annotations

import argparse
import signal
from dataclasses import dataclass


@dataclass
class MotorCommand:
    left_pwm: float
    right_pwm: float
    action: str


class JetBotController:
    def __init__(
        self,
        base_speed: float = 0.35,
        turn_gain: float = 0.2,
        obstacle_threshold: float = 0.6,
        evade_turn_speed: float = 0.25,
    ) -> None:
        self.base_speed = base_speed
        self.turn_gain = turn_gain
        self.obstacle_threshold = obstacle_threshold
        self.evade_turn_speed = evade_turn_speed
        self._emergency_stop = False

        signal.signal(signal.SIGINT, self._set_emergency_stop)
        signal.signal(signal.SIGTERM, self._set_emergency_stop)

    def _set_emergency_stop(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        del signum, frame
        self._emergency_stop = True

    @property
    def emergency_stop(self) -> bool:
        return self._emergency_stop

    def trigger_emergency_stop(self) -> None:
        self._emergency_stop = True

    def clear_emergency_stop(self) -> None:
        self._emergency_stop = False

    @staticmethod
    def _clip(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def steering_to_command(self, steering_pred: str) -> MotorCommand:
        if steering_pred == "left":
            return MotorCommand(
                left_pwm=self._clip(self.base_speed - self.turn_gain),
                right_pwm=self._clip(self.base_speed + self.turn_gain),
                action="turn_left",
            )
        if steering_pred == "right":
            return MotorCommand(
                left_pwm=self._clip(self.base_speed + self.turn_gain),
                right_pwm=self._clip(self.base_speed - self.turn_gain),
                action="turn_right",
            )
        return MotorCommand(left_pwm=self.base_speed, right_pwm=self.base_speed, action="forward")

    def obstacle_override(self, obstacle_probability: float) -> MotorCommand | None:
        if obstacle_probability < self.obstacle_threshold:
            return None

        # Evasive behavior: pivot right in-place.
        return MotorCommand(
            left_pwm=self.evade_turn_speed,
            right_pwm=-self.evade_turn_speed,
            action="evade",
        )

    def decide(self, steering_pred: str, obstacle_probability: float) -> MotorCommand:
        if self._emergency_stop:
            return MotorCommand(left_pwm=0.0, right_pwm=0.0, action="emergency_stop")

        obstacle_cmd = self.obstacle_override(obstacle_probability)
        if obstacle_cmd is not None:
            return obstacle_cmd

        return self.steering_to_command(steering_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick controller decision test.")
    parser.add_argument("--steering", default="forward", choices=["left", "right", "forward"])
    parser.add_argument("--obstacle-prob", type=float, default=0.0)
    args = parser.parse_args()

    controller = JetBotController()
    cmd = controller.decide(args.steering, args.obstacle_prob)
    print(cmd)
