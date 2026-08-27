"""YOLOX-Tiny adapter.

Preprocessing (YOLOX convention, Megvii reference implementation):
    BGR input (no RGB conversion), letterbox to 640×640 with 114 padding,
    no mean/std normalization, float32, HWC → CHW, batch dimension added.

Output decoding:
    Raw head output ``(1, 8400, 85)`` in grid units: ``[cx, cy, w, h,
    objectness, 80 class scores]``. Decoded with strides [8, 16, 32] over
    80×80 / 40×40 / 20×20 grids, scores = objectness × class score,
    per-class NMS (``requires_nms: true``), then mapped back to
    original-image coordinates and canonical COCO category ids.

Weights: official ``yolox_tiny.pth`` at ``configs/models/yolox_tiny.yaml``'s
checkpoint path. The model definition comes from the ``yolox`` package
(Apache 2.0); it is an optional, lazily imported dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.models.base import DetectorAdapter
from edgebench.models.coco_classes import contiguous_to_coco_id
from edgebench.paths import REPO_ROOT
from edgebench.types import Detection

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing.preprocess import LetterboxMeta

_YOLOX_STRIDES = (8, 16, 32)
_NUM_CLASSES = 80


class YOLOXTinyAdapter(DetectorAdapter):
    @property
    def name(self) -> str:
        return "yolox_tiny"

    @property
    def input_size(self) -> tuple[int, int]:
        config = getattr(self, "model_config", None)
        return config.input_size if config is not None else (640, 640)

    @property
    def _confidence_threshold(self) -> float:
        settings = getattr(self, "benchmark_settings", None)
        return settings.confidence_threshold if settings is not None else 0.25

    @property
    def _iou_threshold(self) -> float:
        settings = getattr(self, "benchmark_settings", None)
        return settings.iou_threshold if settings is not None else 0.65

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        """Letterbox + CHW float32 tensor; YOLOX applies no normalization."""
        import numpy as np

        from edgebench.preprocessing import letterbox

        resized, meta = letterbox(image, self.input_size)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(tensor), meta

    def postprocess(
        self, raw_output: np.ndarray, metadata: LetterboxMeta
    ) -> list[Detection]:
        import numpy as np

        from edgebench.preprocessing import rescale_boxes

        output = np.asarray(raw_output, dtype=np.float32)
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2 or output.shape[1] != 5 + _NUM_CLASSES:
            raise ValueError(
                f"Unexpected YOLOX output shape {output.shape}; "
                f"expected (N, {5 + _NUM_CLASSES}) raw head output"
            )

        decoded = self._decode(output)
        cx, cy, widths, heights = decoded.T[:4]
        object_class_scores = decoded[:, 4:5] * decoded[:, 5:]
        class_indices = object_class_scores.argmax(axis=1)
        scores = object_class_scores.max(axis=1)
        keep = scores >= self._confidence_threshold
        if not np.any(keep):
            return []

        x1 = cx - widths / 2.0
        y1 = cy - heights / 2.0
        boxes_xyxy = np.stack(
            [x1, y1, cx + widths / 2.0, cy + heights / 2.0], axis=1
        )[keep]
        scores = scores[keep]
        class_indices = class_indices[keep]

        keep_indices = _per_class_nms(
            boxes_xyxy, scores, class_indices, self._iou_threshold
        )
        boxes_xyxy = rescale_boxes(boxes_xyxy[keep_indices], metadata)
        return [
            Detection(
                bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(scores[keep_indices][row]),
                class_id=contiguous_to_coco_id(class_indices[keep_indices][row]),
            )
            for row, b in enumerate(boxes_xyxy)
        ]

    def load_pytorch(self) -> Any:
        """Load YOLOX-Tiny from the official checkpoint via the yolox package."""
        import torch

        checkpoint = self._checkpoint_path()
        upstream_root = REPO_ROOT / "third_party" / "YOLOX"
        if upstream_root.is_dir() and str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        try:
            from yolox.exp import get_exp
        except ImportError as exc:
            raise ImportError(
                "YOLOX-Tiny requires the `yolox` package (Apache 2.0). "
                "Install the Megvii YOLOX implementation on the target device."
            ) from exc
        exp = get_exp(exp_name="yolox-tiny")
        exp.depth = 0.33
        exp.width = 0.375
        model = exp.get_model()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        model.head.decode_in_inference = False
        return model.eval()

    def export_onnx(self, output_path: str) -> None:
        from edgebench.exporters.onnx import export_onnx

        model = self.load_pytorch()
        export_onnx(model, output_path, input_size=self.input_size, opset=11)

    def _checkpoint_path(self) -> Path:
        config = getattr(self, "model_config", None)
        checkpoint = config.checkpoint if config is not None else None
        if not checkpoint:
            raise RuntimeError(
                "yolox_tiny has no configured checkpoint; "
                "set 'checkpoint' in configs/models/yolox_tiny.yaml"
            )
        path = Path(checkpoint)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLOX-Tiny checkpoint not found: {path}. Download the official "
                "yolox_tiny.pth (see configs/models/yolox_tiny.yaml)."
            )
        return path

    def _decode(self, output: np.ndarray) -> np.ndarray:
        """Expand grid-relative predictions to input-image pixel space."""
        import numpy as np

        input_width, input_height = self.input_size
        grids, strides = [], []
        for stride in _YOLOX_STRIDES:
            grid_h, grid_w = input_height // stride, input_width // stride
            yv, xv = np.meshgrid(
                np.arange(grid_h), np.arange(grid_w), indexing="ij"
            )
            grids.append(np.stack([xv, yv], axis=2).reshape(-1, 2))
            strides.append(np.full(grid_h * grid_w, stride, dtype=np.float32))
        grid = np.concatenate(grids, axis=0).astype(np.float32)
        stride = np.concatenate(strides, axis=0)[:, np.newaxis]

        decoded = output.copy()
        decoded[:, 0:2] = (decoded[:, 0:2] + grid) * stride
        decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * stride
        return decoded


def _per_class_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Per-class NMS via OpenCV; returns kept row indices."""
    import cv2
    import numpy as np

    kept: list[int] = []
    for class_index in np.unique(class_indices):
        mask = class_indices == class_index
        rows = np.nonzero(mask)[0]
        boxes_xywh = boxes_xyxy[rows].copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        picked = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores[rows].tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        picked = np.asarray(picked).reshape(-1)
        kept.extend(rows[picked].tolist())
    return np.asarray(sorted(kept), dtype=np.int64)
