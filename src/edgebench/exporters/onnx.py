"""ONNX export.

Exports the detector's raw (pre-NMS, decode-free) graph with a static
1×3×H×W input. Detector adapters call this from their ``export_onnx``;
PicoDet is Paddle-native and uses paddle2onnx instead of this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_onnx(
    model: Any,
    output_path: str,
    *,
    input_size: tuple[int, int] = (640, 640),
    opset: int = 11,
) -> Path:
    """Export a PyTorch detection model to ONNX with static input shape."""
    import torch

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(input_size[0]), int(input_size[1])
    dummy = torch.zeros(1, 3, height, width)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(target),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    return target
