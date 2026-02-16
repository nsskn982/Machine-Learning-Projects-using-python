"""Rule-based alert generation for JetBot runtime logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from metrics_collector import parse_runtime_jsonl


@dataclass
class Alert:
    rule: str
    severity: str
    message: str
    value: float | int
    threshold: float | int



def evaluate_alerts(
    log_path: str | Path,
    low_conf_threshold: float = 0.4,
    repeated_low_conf_limit: int = 10,
    no_obstacle_limit: int = 300,
    emergency_stop_limit: int = 25,
) -> List[Alert]:
    events = list(parse_runtime_jsonl(log_path))
    alerts: List[Alert] = []

    # Rule 1: repeated low-confidence predictions.
    low_conf_count = sum(1 for event in events if event.confidence is not None and event.confidence < low_conf_threshold)
    if low_conf_count >= repeated_low_conf_limit:
        alerts.append(
            Alert(
                rule="repeated_low_confidence",
                severity="high",
                message="Model confidence is frequently below safe levels.",
                value=low_conf_count,
                threshold=repeated_low_conf_limit,
            )
        )

    # Rule 2: no obstacle detections over long intervals.
    max_consecutive_no_obstacle = 0
    current_streak = 0
    for event in events:
        if event.obstacle_detected is False:
            current_streak += 1
            max_consecutive_no_obstacle = max(max_consecutive_no_obstacle, current_streak)
        elif event.obstacle_detected is True:
            current_streak = 0
    if max_consecutive_no_obstacle >= no_obstacle_limit:
        alerts.append(
            Alert(
                rule="no_obstacle_detections",
                severity="medium",
                message="No obstacles detected for an extended interval; sensor or pipeline may be stale.",
                value=max_consecutive_no_obstacle,
                threshold=no_obstacle_limit,
            )
        )

    # Rule 3: excessive emergency stops.
    emergency_stops = sum(1 for event in events if event.is_stop)
    if emergency_stops >= emergency_stop_limit:
        alerts.append(
            Alert(
                rule="excessive_emergency_stops",
                severity="high",
                message="Emergency stop rate is above configured limit.",
                value=emergency_stops,
                threshold=emergency_stop_limit,
            )
        )

    return alerts



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rule-based monitoring alerts")
    parser.add_argument("--log", default="logs/runtime.jsonl", help="Path to runtime JSONL log")
    parser.add_argument("--low-conf-threshold", type=float, default=0.4)
    parser.add_argument("--repeated-low-conf-limit", type=int, default=10)
    parser.add_argument("--no-obstacle-limit", type=int, default=300)
    parser.add_argument("--emergency-stop-limit", type=int, default=25)
    return parser



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    alerts = evaluate_alerts(
        args.log,
        low_conf_threshold=args.low_conf_threshold,
        repeated_low_conf_limit=args.repeated_low_conf_limit,
        no_obstacle_limit=args.no_obstacle_limit,
        emergency_stop_limit=args.emergency_stop_limit,
    )

    payload: Dict[str, Any] = {
        "alert_count": len(alerts),
        "alerts": [asdict(alert) for alert in alerts],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
