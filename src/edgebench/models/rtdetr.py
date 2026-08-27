"""RT-DETRv2-S adapter for lyuwenyu/RT-DETR's PyTorch implementation."""

from __future__ import annotations

import importlib
import sys
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


class RTDETRv2SAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "rtdetrv2_s"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, ResizeMeta]:
        # Official validation: RGB, direct 640x640 resize, float32 in [0, 1].
        return prepare_image(
            image,
            self.input_size,
            geometry="resize",
            rgb=True,
            scale=255.0,
        )

    def postprocess(self, raw_output: Any, metadata: ResizeMeta) -> list[Detection]:
        return decode_detr_outputs(
            raw_output, metadata, threshold=self.confidence_threshold
        )

    def load_pytorch(self) -> Any:
        import torch

        config_path = self.upstream_config_path()
        upstream_root = _find_upstream_root(config_path)
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        importlib.invalidate_caches()
        try:
            from src.core import YAMLConfig
        except ImportError as exc:
            raise ImportError(
                "RT-DETRv2-S requires the official lyuwenyu/RT-DETR "
                "rtdetrv2_pytorch source tree and dependencies."
            ) from exc

        config = YAMLConfig(str(config_path))
        # The full COCO checkpoint includes backbone weights. Disable the
        # training-time ImageNet download before materializing the model.
        if "PResNet" in config.yaml_cfg:
            config.yaml_cfg["PResNet"]["pretrained"] = False
        checkpoint = torch.load(self.checkpoint_path(), map_location="cpu")
        state = (
            checkpoint["ema"]["module"]
            if "ema" in checkpoint
            else checkpoint["model"]
        )
        config.model.load_state_dict(state)
        model = config.model.deploy()

        class RawOutputs(torch.nn.Module):
            def __init__(self, detector: Any) -> None:
                super().__init__()
                self.detector = detector

            def forward(self, images: Any) -> tuple[Any, Any]:
                output = self.detector(images)
                return output["pred_boxes"], output["pred_logits"]

        return RawOutputs(model).eval()

    def export_onnx(self, output_path: str) -> None:
        from edgebench.exporters.onnx import export_onnx

        export_onnx(
            self.load_pytorch(), output_path, input_size=self.input_size, opset=16
        )


def _find_upstream_root(config_path: Path) -> Path:
    """Find the rtdetrv2_pytorch directory that owns the official ``src``."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "src" / "core").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate rtdetrv2_pytorch/src/core above {config_path}"
    )
