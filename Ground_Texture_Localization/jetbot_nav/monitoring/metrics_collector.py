"""Utilities for collecting rolling runtime metrics for JetBot navigation logs.

Expected JSONL schema (best effort / optional fields):
- timestamp or ts: numeric seconds or ISO string
- confidence: float [0, 1]
- action or event: may indicate stop events
- stop / emergency_stop: boolean stop marker
- obstacle_detected / obstacle: boolean obstacle marker
- fps: precomputed frame rate (optional)

If fps is not logged directly, FPS is estimated from timestamp deltas.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class RuntimeEvent:
    """Normalized runtime event row from JSONL logs."""

    timestamp: Optional[float]
    confidence: Optional[float]
    is_stop: bool
    obstacle_detected: Optional[bool]
    fps: Optional[float]



def _as_timestamp_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    return None



def _normalize_event(raw: Dict[str, Any]) -> RuntimeEvent:
    timestamp = _as_timestamp_seconds(raw.get("timestamp", raw.get("ts")))

    confidence = raw.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else None

    action = str(raw.get("action", raw.get("event", ""))).lower()
    is_stop = bool(raw.get("stop") or raw.get("emergency_stop") or "stop" in action)

    obstacle_val = raw.get("obstacle_detected", raw.get("obstacle"))
    obstacle_detected = (
        bool(obstacle_val)
        if isinstance(obstacle_val, (bool, int, float, str)) and str(obstacle_val).strip() != ""
        else None
    )

    fps = raw.get("fps")
    fps = float(fps) if isinstance(fps, (int, float)) else None

    return RuntimeEvent(
        timestamp=timestamp,
        confidence=confidence,
        is_stop=is_stop,
        obstacle_detected=obstacle_detected,
        fps=fps,
    )



def parse_runtime_jsonl(path: str | Path) -> Iterator[RuntimeEvent]:
    """Yield normalized events from a runtime JSONL file."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            yield _normalize_event(payload)



def _window_fps(events: Iterable[RuntimeEvent]) -> Optional[float]:
    events_list = list(events)
    logged = [event.fps for event in events_list if event.fps is not None]
    if logged:
        return mean(logged)

    ts = [event.timestamp for event in events_list if event.timestamp is not None]
    if len(ts) < 2:
        return None

    deltas = [t2 - t1 for t1, t2 in zip(ts, ts[1:]) if (t2 - t1) > 0]
    if not deltas:
        return None
    avg_delta = mean(deltas)
    return 1.0 / avg_delta if avg_delta > 0 else None



def rolling_metrics(events: Iterable[RuntimeEvent], window_size: int = 100) -> Iterator[Tuple[int, Dict[str, Optional[float]]]]:
    """Compute rolling metrics for a stream of events.

    Yields tuples: (event_index, metric_dict)
    """
    window: Deque[RuntimeEvent] = deque(maxlen=window_size)
    for idx, event in enumerate(events, 1):
        window.append(event)
        subset = list(window)

        conf = [e.confidence for e in subset if e.confidence is not None]
        obstacle = [e.obstacle_detected for e in subset if e.obstacle_detected is not None]

        metrics = {
            "avg_confidence": mean(conf) if conf else None,
            "stop_frequency": sum(1 for e in subset if e.is_stop) / len(subset),
            "obstacle_detected_rate": (sum(bool(v) for v in obstacle) / len(obstacle)) if obstacle else None,
            "fps": _window_fps(subset),
        }
        yield idx, metrics



def aggregate_latest_metrics(path: str | Path, window_size: int = 100) -> Dict[str, Optional[float]]:
    """Return latest rolling metrics from runtime log."""
    latest: Dict[str, Optional[float]] = {
        "avg_confidence": None,
        "stop_frequency": None,
        "obstacle_detected_rate": None,
        "fps": None,
    }
    for _, metrics in rolling_metrics(parse_runtime_jsonl(path), window_size=window_size):
        latest = metrics
    return latest



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate rolling metrics from logs/runtime.jsonl")
    parser.add_argument("--log", default="logs/runtime.jsonl", help="Path to runtime JSONL log")
    parser.add_argument("--window", type=int, default=100, help="Rolling window size")
    return parser



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metrics = aggregate_latest_metrics(args.log, window_size=args.window)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
