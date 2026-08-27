"""Ultralytics YOLO26n one-to-many-head adapter.

YOLO26 defaults to an NMS-free one-to-one head. This benchmark deliberately
selects the one-to-many head so the configured confidence/IoU thresholds and
postprocessing cost are consistent with the other CNN detectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from edgebench.models._common import (
    ConfiguredDetector,
    decode_yolo_output,
    prepare_image,
)

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing import LetterboxMeta
    from edgebench.types import Detection


class YOLO26nAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "yolo26n"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        return prepare_image(
            image,
            self.input_size,
            geometry="letterbox",
            rgb=True,
            scale=255.0,
        )

    def postprocess(
        self, raw_output: Any, metadata: LetterboxMeta
    ) -> list[Detection]:
        return decode_yolo_output(
            raw_output,
            metadata,
            threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    def load_pytorch(self) -> Any:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO26n requires a current `ultralytics` installation."
            ) from exc

        model = YOLO(str(self.checkpoint_path())).model
        head = model.model[-1]
        if hasattr(head, "end2end"):
            head.end2end = False

        class OneToManyOutput(torch.nn.Module):
            def __init__(self, detector: Any) -> None:
                super().__init__()
                self.detector = detector

            def forward(self, images: Any) -> Any:
                output = self.detector(images)
                return output[0] if isinstance(output, (tuple, list)) else output

        return OneToManyOutput(model).eval()

    def export_onnx(self, output_path: str) -> None:
        from edgebench.exporters.onnx import export_onnx

        export_onnx(
            self.load_pytorch(), output_path, input_size=self.input_size, opset=12
        )
