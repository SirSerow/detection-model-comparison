"""RTMDet-Tiny adapter for the official MMDetection implementation.

The adapter exports and consumes the raw three-level RTMDet head. Decode,
thresholding and per-class NMS remain outside the runtime so every runtime
measures the same graph.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from edgebench.models._common import (
    ConfiguredDetector,
    decode_xyxy_scores,
    detections_from_arrays,
    output_sequence,
    prepare_image,
)
from edgebench.paths import REPO_ROOT

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing import LetterboxMeta
    from edgebench.types import Detection

_STRIDES = (8, 16, 32)
_NUM_CLASSES = 80


class RTMDetTinyAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "rtmdet_tiny"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        # Official RTMDet keeps OpenCV BGR order and normalizes 0..255 pixels.
        return prepare_image(
            image,
            self.input_size,
            geometry="letterbox",
            rgb=False,
            mean=(103.53, 116.28, 123.675),
            std=(57.375, 57.12, 58.395),
        )

    def postprocess(
        self, raw_output: Any, metadata: LetterboxMeta
    ) -> list[Detection]:
        import numpy as np

        outputs = output_sequence(raw_output)
        # MMDeploy end-to-end contract: dets=(B,N,5), labels=(B,N).
        dets = next(
            (item for item in outputs if item.ndim >= 2 and item.shape[-1] == 5),
            None,
        )
        if dets is not None and len(outputs) == 2:
            labels = next(item for item in outputs if item is not dets)
            dets = dets[0] if dets.ndim == 3 else dets
            labels = labels[0] if labels.ndim == 2 else labels
            keep = dets[:, 4] >= self.confidence_threshold
            return detections_from_arrays(
                dets[keep, :4], dets[keep, 4], labels[keep], metadata
            )

        cls_levels = [
            item
            for item in outputs
            if item.ndim == 4 and item.shape[1] == _NUM_CLASSES
        ]
        box_levels = [
            item for item in outputs if item.ndim == 4 and item.shape[1] == 4
        ]
        if len(cls_levels) != len(_STRIDES) or len(box_levels) != len(_STRIDES):
            raise ValueError(
                "Unexpected RTMDet output shapes; expected three class maps and "
                f"three box maps, received {[item.shape for item in outputs]}"
            )

        decoded_boxes: list[np.ndarray] = []
        decoded_scores: list[np.ndarray] = []
        cls_levels.sort(key=lambda item: item.shape[2] * item.shape[3], reverse=True)
        box_levels.sort(key=lambda item: item.shape[2] * item.shape[3], reverse=True)
        for stride, cls_map, box_map in zip(_STRIDES, cls_levels, box_levels):
            cls_map = (
                cls_map[0]
                .transpose(1, 2, 0)
                .reshape(-1, _NUM_CLASSES)
                .astype(np.float32)
            )
            scores = 1.0 / (1.0 + np.exp(-np.clip(cls_map, -88.0, 88.0)))
            # RTMDetSepBNHead already multiplies regression distances by stride.
            distances = box_map[0].transpose(1, 2, 0).reshape(-1, 4)
            height, width = box_map.shape[2:]
            y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            priors = np.stack([x, y], axis=-1).reshape(-1, 2) * stride
            boxes = np.column_stack(
                [
                    priors[:, 0] - distances[:, 0],
                    priors[:, 1] - distances[:, 1],
                    priors[:, 0] + distances[:, 2],
                    priors[:, 1] + distances[:, 3],
                ]
            )
            decoded_boxes.append(boxes)
            decoded_scores.append(scores)
        return decode_xyxy_scores(
            np.concatenate(decoded_boxes),
            np.concatenate(decoded_scores),
            metadata,
            threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    def load_pytorch(self) -> Any:
        upstream_root = REPO_ROOT / "third_party" / "mmdetection"
        if upstream_root.is_dir() and str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        # RTMDet's raw backbone/neck/head forward is implemented entirely with
        # PyTorch.  MMDetection nevertheless imports optional compiled MMCV ops
        # (ROIAlign, NMS, etc.) while registering unrelated model families.
        # Keep those imports lazy on new Torch/CUDA versions where no matching
        # MMCV extension wheel exists; the benchmark never calls these stubs.
        import mmcv.utils.ext_loader

        class UnavailableExtension:
            def __init__(self, functions: list[str]) -> None:
                for function in functions:
                    setattr(self, function, self._raise)

            @staticmethod
            def _raise(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "This RTMDet raw-head benchmark does not provide compiled MMCV ops"
                )

        mmcv.utils.ext_loader.load_ext = (
            lambda _name, functions: UnavailableExtension(functions)
        )
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                "RTMDet-Tiny requires MMDetection 3.x and MMEngine. Install "
                "the official mmdetection package on the target device."
            ) from exc

        import torch

        model = init_detector(
            str(self.upstream_config_path()), checkpoint=None, device="cpu"
        )
        checkpoint = torch.load(
            self.checkpoint_path(), map_location="cpu", weights_only=False
        )
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        unexpected = set(incompatible.unexpected_keys) - {
            "data_preprocessor.mean",
            "data_preprocessor.std",
        }
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Official RTMDet checkpoint does not match the configured model: "
                f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
            )

        class RawHead(torch.nn.Module):
            def __init__(self, detector: Any) -> None:
                super().__init__()
                self.detector = detector

            def forward(self, images: Any) -> Any:
                return self.detector(images, mode="tensor")

        return RawHead(model).eval()

    def export_onnx(self, output_path: str) -> None:
        from edgebench.exporters.onnx import export_onnx

        export_onnx(
            self.load_pytorch(), output_path, input_size=self.input_size, opset=11
        )
