"""Monitoring dashboard plots for JetBot runtime and labeled replay analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from metrics_collector import parse_runtime_jsonl



def _load_runtime(log_path: str | Path) -> List[Dict[str, Optional[float]]]:
    data: List[Dict[str, Optional[float]]] = []
    for idx, event in enumerate(parse_runtime_jsonl(log_path), 1):
        data.append(
            {
                "index": float(idx),
                "timestamp": event.timestamp,
                "confidence": event.confidence,
                "speed": None,
                "predicted_label": None,
                "true_label": None,
            }
        )
    # Optional enrichment using raw json lines (speed / labels if present)
    with Path(log_path).open("r", encoding="utf-8") as f:
        runtime_idx = 0
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if runtime_idx >= len(data):
                break
            speed = obj.get("speed")
            if isinstance(speed, (int, float)):
                data[runtime_idx]["speed"] = float(speed)
            pred = obj.get("predicted_label")
            true = obj.get("true_label")
            data[runtime_idx]["predicted_label"] = str(pred) if pred is not None else None
            data[runtime_idx]["true_label"] = str(true) if true is not None else None
            runtime_idx += 1
    return data



def _confusion_like_counts(rows: List[Dict[str, Optional[float]]]) -> Dict[str, int]:
    summary = {"correct": 0, "incorrect": 0, "unlabeled": 0}
    for row in rows:
        pred = row.get("predicted_label")
        true = row.get("true_label")
        if pred is None or true is None:
            summary["unlabeled"] += 1
        elif pred == true:
            summary["correct"] += 1
        else:
            summary["incorrect"] += 1
    return summary



def build_dashboard(
    log_path: str | Path,
    output_path: str | Path = "monitoring_dashboard.png",
    confidence_threshold: float = 0.4,
    speed_threshold: float = 0.6,
) -> Path:
    rows = _load_runtime(log_path)

    x = [row["index"] for row in rows]
    confidence = [row["confidence"] if row["confidence"] is not None else float("nan") for row in rows]
    speed = [row["speed"] if row["speed"] is not None else 0.0 for row in rows]

    risky_x = [
        row["index"]
        for row in rows
        if row["confidence"] is not None
        and row["speed"] is not None
        and row["confidence"] < confidence_threshold
        and row["speed"] > speed_threshold
    ]
    risky_y = [
        row["confidence"]
        for row in rows
        if row["confidence"] is not None
        and row["speed"] is not None
        and row["confidence"] < confidence_threshold
        and row["speed"] > speed_threshold
    ]

    summary = _confusion_like_counts(rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(x, confidence, label="confidence", color="tab:blue")
    axes[0].axhline(confidence_threshold, linestyle="--", color="tab:red", label="low confidence threshold")
    if risky_x:
        axes[0].scatter(risky_x, risky_y, color="tab:orange", label="risky intervals", zorder=4)
    axes[0].set_title("Confidence trend with risky interval highlighting")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("confidence")
    axes[0].legend(loc="best")

    labels = list(summary.keys())
    values = [summary[k] for k in labels]
    axes[1].bar(labels, values, color=["tab:green", "tab:red", "tab:gray"])
    axes[1].set_title("Confusion-like summary from labeled replays")
    axes[1].set_ylabel("count")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate monitoring dashboard visualizations")
    parser.add_argument("--log", default="logs/runtime.jsonl", help="Path to runtime JSONL log")
    parser.add_argument("--output", default="monitoring_dashboard.png", help="Output image path")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--speed-threshold", type=float, default=0.6)
    return parser



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output = build_dashboard(
        log_path=args.log,
        output_path=args.output,
        confidence_threshold=args.confidence_threshold,
        speed_threshold=args.speed_threshold,
    )
    print(str(output))


if __name__ == "__main__":
    main()
