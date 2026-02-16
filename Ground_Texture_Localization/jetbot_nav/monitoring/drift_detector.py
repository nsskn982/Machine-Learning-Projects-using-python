"""Simple drift detection helpers for JetBot embedding/statistics monitoring."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass
class DriftResult:
    distance: float
    threshold: float
    drift: bool



def _flatten_vector(payload: Mapping[str, object], key: str) -> Optional[List[float]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return None
    vector: List[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            vector.append(float(item))
    return vector if vector else None



def _euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError("Vectors must have the same length for distance comparison")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a_list, b_list)))



def _stats_distance(current: Mapping[str, float], baseline: Mapping[str, float], keys: Iterable[str]) -> float:
    diffs = []
    for key in keys:
        if key not in current or key not in baseline:
            continue
        diffs.append(abs(float(current[key]) - float(baseline[key])))
    return sum(diffs) / len(diffs) if diffs else 0.0



def detect_drift(
    current_payload: Mapping[str, object],
    baseline_payload: Mapping[str, object],
    threshold: float,
    embedding_weight: float = 0.7,
) -> DriftResult:
    """Compare current features to baseline and return drift status.

    Uses a weighted combination of:
    - Euclidean distance between `embedding` vectors (if present)
    - Mean absolute difference across scalar statistics keys.
    """
    current_embedding = _flatten_vector(current_payload, "embedding")
    baseline_embedding = _flatten_vector(baseline_payload, "embedding")

    embed_distance = 0.0
    if current_embedding and baseline_embedding:
        embed_distance = _euclidean_distance(current_embedding, baseline_embedding)

    stats_keys = set(current_payload.keys()) & set(baseline_payload.keys())
    stats_keys = {k for k in stats_keys if k != "embedding"}
    current_stats = {
        k: float(v)
        for k, v in current_payload.items()
        if k in stats_keys and isinstance(v, (int, float))
    }
    baseline_stats = {
        k: float(v)
        for k, v in baseline_payload.items()
        if k in stats_keys and isinstance(v, (int, float))
    }
    stat_distance = _stats_distance(current_stats, baseline_stats, stats_keys)

    if current_embedding and baseline_embedding:
        distance = embedding_weight * embed_distance + (1 - embedding_weight) * stat_distance
    else:
        distance = stat_distance

    return DriftResult(distance=distance, threshold=threshold, drift=distance > threshold)



def _load_json(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect drift using baseline and current feature snapshots")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
    parser.add_argument("--current", required=True, help="Path to current JSON")
    parser.add_argument("--threshold", type=float, default=1.0, help="Distance threshold for drift warning")
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.7,
        help="Weight of embedding distance in [0, 1]",
    )
    return parser



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    baseline = _load_json(args.baseline)
    current = _load_json(args.current)

    result = detect_drift(
        current_payload=current,
        baseline_payload=baseline,
        threshold=args.threshold,
        embedding_weight=args.embedding_weight,
    )

    output = {
        "distance": result.distance,
        "threshold": result.threshold,
        "drift": result.drift,
        "status": "warning" if result.drift else "ok",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
