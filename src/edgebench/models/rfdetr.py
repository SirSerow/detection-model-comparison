"""RF-DETR-Nano adapter using the official ``rfdetr`` package."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.models._common import (
    ConfiguredDetector,
    decode_detr_outputs,
    prepare_image,
)

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing import ResizeMeta
    from edgebench.types import Detection

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RFDETRNanoAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "rfdetr_nano"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, ResizeMeta]:
        return prepare_image(
            image,
            self.input_size,
            geometry="resize",
            rgb=True,
            scale=255.0,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        )

    def postprocess(self, raw_output: Any, metadata: ResizeMeta) -> list[Detection]:
        return decode_detr_outputs(
            raw_output,
            metadata,
            threshold=self.confidence_threshold,
            sparse_coco_layout=True,
        )

    def load_pytorch(self) -> Any:
        try:
            import torch
            from rfdetr import RFDETRNano
        except ImportError as exc:
            raise ImportError(
                "RF-DETR-Nano requires the official `rfdetr` package."
            ) from exc

        width, height = self.input_size
        if width != height:
            raise ValueError("RF-DETR requires a square input size")
        detector = RFDETRNano(
            pretrain_weights=str(self.checkpoint_path()), resolution=width
        )
        model = detector.model.model

        class RawOutputs(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, images: Any) -> tuple[Any, Any]:
                output = self.inner(images)
                return output["pred_boxes"], output["pred_logits"]

        return RawOutputs(model).eval()

    def export_onnx(self, output_path: str) -> None:
        try:
            from rfdetr import RFDETRNano
        except ImportError as exc:
            raise ImportError(
                "RF-DETR-Nano requires the official `rfdetr` package."
            ) from exc

        width, height = self.input_size
        if width != height:
            raise ValueError("RF-DETR requires a square input size")
        detector = RFDETRNano(
            pretrain_weights=str(self.checkpoint_path()), resolution=width
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as export_dir:
            exported = detector.export(
                output_dir=export_dir,
                shape=(height, width),
                opset_version=17,
                verbose=False,
            )
            shutil.copyfile(exported, destination)
