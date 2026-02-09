# Ground Texture Recognition & Localization (Raspberry Pi HQ / Jetson Nano)

This mini-project shows how to build a **ground texture recognition** and **localization** pipeline using a Raspberry Pi HQ camera (or any CSI/USB camera) and a Jetson Nano / Raspberry Pi. It is designed to work with the parts referenced by the user (HQ cam, C/CS lenses, chassis, etc.), but the algorithm is hardware-agnostic.

## What this does

1. **Collect data**: Capture short clips or images of the floor textures you care about (e.g., tile, wood, carpet, asphalt, grass).
2. **Extract features**: Compute texture descriptors (Local Binary Patterns + color histograms).
3. **Train a classifier**: Learn to predict the texture label from camera frames.
4. **Localize on a map**: Use a simple **particle filter** over a known floor map made of texture labels.

## Hardware mapping (from provided links)

- **Cameras**: Raspberry Pi HQ camera or Pi cam v2 (CSI) or USB bundle.
- **Compute**: Jetson Nano or Raspberry Pi.
- **Mobile base**: chassis + wheels + caster wheel.
- **Optional sensors**: Qwiic motor driver & IMU/encoders for better motion.

> The algorithm uses only camera frames and an optional odometry estimate. You can start without wheel encoders and add them later.

## Hardware specifications (typical)

These are **typical** specs for the referenced parts. Always verify exact specs from the vendor pages you linked.

- **Raspberry Pi HQ Camera (IMX477)**: ~12.3MP sensor, supports interchangeable C/CS lenses.
- **Raspberry Pi Camera v2 (IMX219)**: ~8MP sensor, fixed-focus lens.
- **Raspberry Pi 1.6MP Shutter Cam**: ~1.6MP global shutter sensor (better for motion, less rolling-shutter blur).
- **Jetson Nano (4GB)**: 128‑core Maxwell GPU, quad‑core ARM CPU; sufficient for classical CV or small CNNs.

> Lens choices (e.g., 8–50mm, 16mm telephoto) affect field of view and texture scale. Pick a lens so each frame covers multiple texture “patches” for robust classification.

## Quick start

### 1) Install dependencies

```bash
pip install numpy opencv-python scikit-learn
```

### 2) Collect data

```bash
python collect_data.py --label tile --out data/tile
python collect_data.py --label carpet --out data/carpet
```

### 3) Train the model

```bash
python train_model.py --data-root data --model-out models/texture_model.pkl
```

### 4) Run localization

```bash
python localize.py --model models/texture_model.pkl --map maps/lab_floor.json
```

## Map format

The localization script uses a JSON map with grid cells labeled by texture.

```json
{
  "cell_size_m": 0.25,
  "grid": [
    ["tile", "tile", "wood"],
    ["tile", "carpet", "wood"],
    ["tile", "carpet", "carpet"]
  ]
}
```

## Notes for real robots

- Keep camera **facing downward** with a stable mount.
- Use **consistent lighting** (add LED ring if needed).
- Add **motion blur reduction** (higher shutter or slower speed).
- For better localization, fuse **wheel odometry** with the particle filter.

## Files

- `collect_data.py`: capture labeled images for each texture.
- `train_model.py`: train a texture classifier (LBP + color histogram).
- `localize.py`: online texture classification + particle filter localization.
- `maps/lab_floor.json`: example texture map.

---

If you want a higher-accuracy pipeline, swap in CNN features or add a small semantic segmentation model.
