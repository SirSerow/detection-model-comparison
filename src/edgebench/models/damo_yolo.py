"""DAMO-YOLO-T adapter using the official TinyNAS-L25-S configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from edgebench.models._common import (
    ConfiguredDetector,
    decode_xyxy_scores,
    output_sequence,
    prepare_image,
)

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing import ResizeMeta
    from edgebench.types import Detection

_NUM_CLASSES = 80


class DAMOYOLOTAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "damo_yolo_t"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, ResizeMeta]:
        # DAMO's demo converts to RGB, stretches and ToTensor-scales to 0..1.
        return prepare_image(
            image,
            self.input_size,
            geometry="resize",
            rgb=True,
            scale=255.0,
        )

    def postprocess(self, raw_output: Any, metadata: ResizeMeta) -> list[Detection]:
        outputs = output_sequence(raw_output)
        boxes = [item for item in outputs if item.ndim >= 2 and item.shape[-1] == 4]
        scores = [
            item for item in outputs if item.ndim >= 2 and item.shape[-1] == _NUM_CLASSES
        ]
        if len(boxes) != 1 or len(scores) != 1:
            raise ValueError(
                "Unexpected DAMO-YOLO output shapes; expected boxes (*,4) and "
                f"scores (*,80), received {[item.shape for item in outputs]}"
            )
        return decode_xyxy_scores(
            boxes[0],
            scores[0],
            metadata,
            threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    def load_pytorch(self) -> Any:
        try:
            import torch
            from damo.base_models.core.ops import RepConv
            from damo.config.base import parse_config
            from damo.detectors.detector import build_local_model
        except ImportError as exc:
            raise ImportError(
                "DAMO-YOLO-T requires the official tinyvision/DAMO-YOLO "
                "repository and its dependencies on PYTHONPATH."
            ) from exc

        config_path = self.upstream_config_path()
        previous_directory = os.getcwd()
        try:
            # Official configs read TinyNAS structure files relative to repo root.
            os.chdir(config_path.parent.parent)
            config = parse_config(str(config_path))
        finally:
            os.chdir(previous_directory)
        model = build_local_model(config, "cpu")
        checkpoint = torch.load(self.checkpoint_path(), map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
        model.head.nms = False
        for layer in model.modules():
            if isinstance(layer, RepConv):
                layer.switch_to_deploy()
        return model.eval()

    def export_onnx(self, output_path: str) -> None:
        from edgebench.exporters.onnx import export_onnx

        export_onnx(
            self.load_pytorch(), output_path, input_size=self.input_size, opset=11
        )
