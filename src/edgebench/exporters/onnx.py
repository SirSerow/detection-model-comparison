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
    # PyTorch's dynamo exporter may use this sidecar even for small models.
    # Keep benchmark artifacts self-contained and remove a stale sidecar from
    # an earlier export before replacing the graph.
    external_data = target.with_suffix(f"{target.suffix}.data")
    if external_data.is_file():
        external_data.unlink()
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
            external_data=False,
            # The legacy exporter honors the adapter-selected opset. The new
            # dynamo exporter currently promotes these detector graphs to
            # opset 18, which older Jetson TensorRT releases cannot parse.
            dynamo=False,
        )
    return target


def convert_onnx_to_fp16(path: str | Path) -> Path:
    """Convert graph computation, weights, and public I/O to FP16."""
    import onnx

    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise ImportError(
            "FP16 ONNX export requires `onnxconverter-common`"
        ) from exc

    target = Path(path)
    model = onnx.load(str(target), load_external_data=True)
    converted = float16.convert_float_to_float16(model, keep_io_types=False)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*converted.graph.value_info, *converted.graph.output)
        if value.type.HasField("tensor_type")
    }
    for node in converted.graph.node:
        if node.op_type != "Cast":
            continue
        desired_type = next(
            (value_types[name] for name in node.output if name in value_types),
            None,
        )
        if desired_type is None:
            continue
        cast_type = next((item for item in node.attribute if item.name == "to"), None)
        if cast_type is not None:
            cast_type.i = desired_type
    onnx.checker.check_model(converted)
    onnx.save_model(converted, str(target), save_as_external_data=False)
    return target
