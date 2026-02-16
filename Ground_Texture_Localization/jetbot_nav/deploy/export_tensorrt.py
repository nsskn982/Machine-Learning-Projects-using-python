"""Export TorchScript model to ONNX and TensorRT engine."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX and TensorRT")
    parser.add_argument("--model", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("best_model.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("best_model.engine"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def export_onnx(model_path: Path, onnx_path: Path, opset: int, width: int, height: int) -> None:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 3, height, width)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["steering_logits", "obstacle_logits"],
        dynamic_axes={"image": {0: "batch"}},
    )


def export_tensorrt(onnx_path: Path, engine_path: Path, fp16: bool) -> None:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=2048",
    ]
    if fp16:
        cmd.append("--fp16")

    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    export_onnx(args.model, args.onnx, args.opset, args.width, args.height)
    print(f"Wrote ONNX model: {args.onnx}")

    try:
        export_tensorrt(args.onnx, args.engine, args.fp16)
        print(f"Wrote TensorRT engine: {args.engine}")
    except FileNotFoundError:
        print("trtexec not found. Install TensorRT tools on Jetson to build engine.")


if __name__ == "__main__":
    main()
