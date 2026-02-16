# JetBot Navigation

## Monitoring

The `monitoring/` package provides lightweight, file-based observability for runtime safety and model quality.

### Components

- `metrics_collector.py`
  - Parses `logs/runtime.jsonl`.
  - Computes rolling averages/rates for:
    - average confidence,
    - stop frequency,
    - obstacle-detected rate,
    - FPS (logged or inferred from timestamps).

- `drift_detector.py`
  - Compares a current feature snapshot to a training baseline snapshot.
  - Uses a weighted distance across embedding vectors and scalar stats.
  - Raises a drift warning when the distance exceeds a threshold.

- `dashboard.py`
  - Produces a monitoring chart with:
    - confidence trend over time,
    - confusion-like summary (`correct`, `incorrect`, `unlabeled`) from replay labels,
    - highlighted risky intervals where confidence is low and speed is high.

- `alerts.py`
  - Rule-based alerts for operational safety:
    - repeated low-confidence predictions,
    - long intervals with no obstacle detections,
    - excessive emergency stops.

### Recommended usage

From `Ground_Texture_Localization/jetbot_nav`:

```bash
python monitoring/metrics_collector.py --log logs/runtime.jsonl --window 100
python monitoring/drift_detector.py --baseline logs/baseline_stats.json --current logs/current_stats.json --threshold 1.2
python monitoring/dashboard.py --log logs/runtime.jsonl --output monitoring_dashboard.png
python monitoring/alerts.py --log logs/runtime.jsonl
```

### How to interpret outputs

- **Average confidence**
  - Healthy: stable and above task-defined floor (for example >0.6).
  - Risky: persistent drops suggest distribution shift, lighting changes, or sensor degradation.

- **Stop frequency**
  - Healthy: occasional stops in cluttered scenes.
  - Risky: consistently high values indicate over-conservative behavior or noisy obstacle predictions.

- **Obstacle-detected rate**
  - Healthy: aligns with environment complexity.
  - Risky low rate over long sessions can indicate camera framing issues, stale data, or detector faults.

- **FPS**
  - Healthy: near deployment target with low variance.
  - Risky: sustained drops can increase control latency and navigation instability.

- **Drift distance**
  - Healthy: below configured threshold and stable over time.
  - Risky: threshold crossings suggest retraining, recalibration, or environment-specific adaptation is needed.

- **Risky intervals (dashboard)**
  - Highlighted regions combine low confidence with high speed and should be prioritized for replay review.

- **Alerts**
  - `repeated_low_confidence`: investigate model robustness and scene conditions.
  - `no_obstacle_detections`: verify sensor pipeline and replay examples.
  - `excessive_emergency_stops`: tune control thresholds and inspect false-positive triggers.
