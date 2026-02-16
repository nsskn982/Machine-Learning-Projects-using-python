# JetBot Navigation Multi-Task Module

This module provides end-to-end utilities for collecting and training a lightweight
multi-task navigation model for JetBot:

- steering classification (left/straight/right/stop)
- obstacle classification (clear/obstacle)

## Dataset schema

Expected dataset layout:

```text
dataset/
├── images/
│   ├── img_*.jpg
└── labels.csv
```

`labels.csv` columns:

1. `image_path` - relative path from dataset root, e.g. `images/img_20250101_120000_000001.jpg`
2. `steering_label` - integer class index for steering
3. `obstacle_label` - integer class index for obstacle state
4. `timestamp` - UTC ISO-8601 timestamp when sample was captured

## Class names

Default steering classes:

- `0`: left
- `1`: straight
- `2`: right
- `3`: stop

Default obstacle classes:

- `0`: clear
- `1`: obstacle

Class names are also exported into `artifacts/class_mapping.json` after training.

## Data capture

From `Ground_Texture_Localization/jetbot_nav/`:

```bash
python data_capture.py --use-csi --output-dir dataset
```

Keyboard labeling controls:

- Steering: `a` left, `s` straight, `d` right, `x` stop
- Obstacle: `c` clear, `o` obstacle
- Save sample: `space`
- Quit: `q`

## Training command

```bash
python train.py --config configs/train.yaml --dataset-root dataset --backbone mobilenet_v3_small
```

Training writes:

- best checkpoint: `artifacts/best_model.pt`
- class map: `artifacts/class_mapping.json`
