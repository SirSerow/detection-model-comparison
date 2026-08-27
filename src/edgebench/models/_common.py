"""Shared, runtime-neutral detector adapter utilities.

The benchmark runtimes return NumPy containers only.  This module turns
the output contracts used by the upstream detector projects into the
benchmark's canonical ``Detection`` records without importing those
projects at package-import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from edgebench.models.base import DetectorAdapter
from edgebench.models.coco_classes import COCO_CATEGORY_IDS, contiguous_to_coco_id
from edgebench.paths import REPO_ROOT
from edgebench.preprocessing import LetterboxMeta, ResizeMeta
from edgebench.types import Detection

if TYPE_CHECKING:
    import numpy as np

ImageMeta = LetterboxMeta | ResizeMeta


class ConfiguredDetector(DetectorAdapter):
    """Common access to model and benchmark configuration."""

    @property
    def input_size(self) -> tuple[int, int]:
        config = getattr(self, "model_config", None)
        return config.input_size if config is not None else (640, 640)

    @property
    def confidence_threshold(self) -> float:
        settings = getattr(self, "benchmark_settings", None)
        return settings.confidence_threshold if settings is not None else 0.25

    @property
    def iou_threshold(self) -> float:
        settings = getattr(self, "benchmark_settings", None)
        return settings.iou_threshold if settings is not None else 0.65

    def checkpoint_path(self) -> Path:
        return self._configured_path("checkpoint", "checkpoint")

    def upstream_config_path(self) -> Path:
        return self._configured_path("upstream_config", "upstream config")

    def _configured_path(self, attribute: str, description: str) -> Path:
        config = getattr(self, "model_config", None)
        value = getattr(config, attribute, None) if config is not None else None
        if not value:
            raise RuntimeError(
                f"{self.name} has no configured {description}; set '{attribute}' "
                f"in configs/models/{self.name}.yaml"
            )
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"{self.name} {description} not found: {path}")
        return path


def prepare_image(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    geometry: Literal["letterbox", "resize"],
    rgb: bool,
    scale: float = 1.0,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, ImageMeta]:
    """Create contiguous NCHW float32 input with explicit model conventions."""
    import numpy as np

    from edgebench.preprocessing import letterbox, resize

    transformed, meta = (
        letterbox(image, size) if geometry == "letterbox" else resize(image, size)
    )
    if rgb:
        transformed = transformed[..., ::-1]
    tensor = transformed.astype(np.float32)
    if scale != 1.0:
        tensor /= float(scale)
    if mean is not None:
        tensor -= np.asarray(mean, dtype=np.float32)
    if std is not None:
        tensor /= np.asarray(std, dtype=np.float32)
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(tensor), meta


def output_sequence(raw_output: Any) -> list[np.ndarray]:
    """Flatten nested runtime output containers while preserving order."""
    import numpy as np

    if isinstance(raw_output, dict):
        return [np.asarray(value) for value in raw_output.values()]
    if isinstance(raw_output, (list, tuple)):
        flattened: list[np.ndarray] = []
        for item in raw_output:
            flattened.extend(output_sequence(item))
        return flattened
    return [np.asarray(raw_output)]


def decode_xyxy_scores(
    boxes: np.ndarray,
    scores: np.ndarray,
    meta: ImageMeta,
    *,
    threshold: float,
    iou_threshold: float,
    labels_are_coco_ids: bool = False,
) -> list[Detection]:
    """Decode dense ``boxes (N,4)`` / class ``scores (N,C)`` and apply NMS."""
    import numpy as np

    boxes = _without_batch(boxes, last_dim=4).astype(np.float32)
    scores = _without_batch(scores).astype(np.float32)
    if scores.ndim != 2 or boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected aligned boxes (N,4) and scores (N,C), got "
            f"{boxes.shape} and {scores.shape}"
        )
    rows, class_indices = np.nonzero(scores >= threshold)
    if rows.size == 0:
        return []
    boxes = boxes[rows]
    confidences = scores[rows, class_indices]
    picked = per_class_nms(boxes, confidences, class_indices, iou_threshold)
    return detections_from_arrays(
        boxes[picked],
        confidences[picked],
        class_indices[picked],
        meta,
        labels_are_coco_ids=labels_are_coco_ids,
    )


def decode_detr_outputs(
    raw_output: Any,
    meta: ImageMeta,
    *,
    threshold: float,
    sparse_coco_layout: bool = False,
) -> list[Detection]:
    """Decode DETR ``pred_boxes``/``pred_logits`` or deployed triplet outputs."""
    import numpy as np

    if isinstance(raw_output, dict):
        if {"labels", "boxes", "scores"}.issubset(raw_output):
            return _decode_deployed_triplet(
                raw_output["boxes"], raw_output["scores"], raw_output["labels"], meta,
                threshold=threshold,
                sparse_coco_layout=sparse_coco_layout,
            )
        if {"pred_boxes", "pred_logits"}.issubset(raw_output):
            boxes = np.asarray(raw_output["pred_boxes"])
            logits = np.asarray(raw_output["pred_logits"])
        else:
            raise ValueError(
                f"Unexpected DETR output keys: {sorted(str(key) for key in raw_output)}"
            )
    else:
        outputs = output_sequence(raw_output)
        if len(outputs) == 3:
            box_candidates = [item for item in outputs if item.ndim >= 2 and item.shape[-1] == 4]
            vector_candidates = [
                item
                for item in outputs
                if not any(item is candidate for candidate in box_candidates)
            ]
            if len(box_candidates) == 1 and len(vector_candidates) == 2:
                labels = next(
                    (item for item in vector_candidates if np.issubdtype(item.dtype, np.integer)),
                    vector_candidates[0],
                )
                scores = next(item for item in vector_candidates if item is not labels)
                return _decode_deployed_triplet(
                    box_candidates[0], scores, labels, meta,
                    threshold=threshold,
                    sparse_coco_layout=sparse_coco_layout,
                )
        box_candidates = [
            item for item in outputs if item.ndim >= 2 and item.shape[-1] == 4
        ]
        logit_candidates = [
            item for item in outputs if item.ndim >= 2 and item.shape[-1] != 4
        ]
        if len(box_candidates) != 1 or len(logit_candidates) != 1:
            shapes = [item.shape for item in outputs]
            raise ValueError(
                "Expected one DETR boxes tensor and one logits tensor; "
                f"received shapes {shapes}"
            )
        boxes, logits = box_candidates[0], logit_candidates[0]

    boxes = _without_batch(boxes, last_dim=4).astype(np.float32)
    logits = _without_batch(logits).astype(np.float32)
    if logits.ndim != 2 or logits.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"Expected DETR boxes (Q,4) and logits (Q,C), got "
            f"{boxes.shape} and {logits.shape}"
        )

    probabilities = _sigmoid(logits)
    class_ids = np.arange(probabilities.shape[1], dtype=np.int64)
    if sparse_coco_layout and probabilities.shape[1] >= 91:
        class_ids = np.asarray(COCO_CATEGORY_IDS, dtype=np.int64)
        probabilities = probabilities[:, class_ids]
    elif probabilities.shape[1] == 81:
        probabilities = probabilities[:, :80]
        class_ids = class_ids[:80]

    flat = probabilities.reshape(-1)
    selection_count = min(boxes.shape[0], flat.size)
    if selection_count == 0:
        return []
    top = np.argpartition(flat, -selection_count)[-selection_count:]
    top = top[np.argsort(flat[top])[::-1]]
    scores = flat[top]
    keep = scores >= threshold
    top, scores = top[keep], scores[keep]
    if top.size == 0:
        return []
    class_count = probabilities.shape[1]
    query_indices = top // class_count
    labels = class_ids[top % class_count]

    cx, cy, width, height = boxes[query_indices].T
    xyxy = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
        axis=1,
    )
    xyxy *= np.asarray(
        [meta.orig_width, meta.orig_height, meta.orig_width, meta.orig_height],
        dtype=np.float32,
    )
    return detections_from_arrays(
        xyxy,
        scores,
        labels,
        meta,
        boxes_in_original_space=True,
        labels_are_coco_ids=sparse_coco_layout and logits.shape[1] >= 91,
    )


def decode_yolo_output(
    raw_output: Any,
    meta: ImageMeta,
    *,
    threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Decode YOLO26 one-to-many ``xywh + 80 scores`` output."""
    import numpy as np

    output = output_sequence(raw_output)[0].astype(np.float32)
    if output.ndim == 3:
        output = output[0]
    if output.shape == (84, 8400) or (output.ndim == 2 and output.shape[0] == 84):
        output = output.T
    if output.ndim == 2 and output.shape[1] == 6:
        return _decode_six_column_output(output, meta, threshold=threshold)
    if output.ndim != 2 or output.shape[1] != 84:
        raise ValueError(
            f"Unexpected YOLO26 output shape {output.shape}; expected (N,84)"
        )
    cx, cy, width, height = output[:, :4].T
    boxes = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
        axis=1,
    )
    return decode_xyxy_scores(
        boxes,
        output[:, 4:],
        meta,
        threshold=threshold,
        iou_threshold=iou_threshold,
    )


def detections_from_arrays(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: ImageMeta,
    *,
    boxes_in_original_space: bool = False,
    labels_are_coco_ids: bool = False,
) -> list[Detection]:
    """Create canonical detections and invert preprocessing geometry."""
    import numpy as np

    from edgebench.preprocessing import LetterboxMeta, rescale_boxes, rescale_resized_boxes

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Detection boxes, scores, and labels must have equal lengths")
    if not boxes_in_original_space:
        boxes = (
            rescale_boxes(boxes, meta)
            if isinstance(meta, LetterboxMeta)
            else rescale_resized_boxes(boxes, meta)
        )
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, meta.orig_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, meta.orig_height)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    detections: list[Detection] = []
    for box, score, label in zip(boxes[valid], scores[valid], labels[valid]):
        class_id = int(label) if labels_are_coco_ids else contiguous_to_coco_id(label)
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in box),
                score=float(score),
                class_id=class_id,
            )
        )
    return detections


def per_class_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Apply OpenCV NMS independently to each class."""
    import cv2
    import numpy as np

    kept: list[int] = []
    for class_index in np.unique(class_indices):
        rows = np.nonzero(class_indices == class_index)[0]
        boxes_xywh = np.asarray(boxes_xyxy[rows], dtype=np.float32).copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        picked = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            np.asarray(scores[rows], dtype=np.float32).tolist(),
            score_threshold=0.0,
            nms_threshold=float(iou_threshold),
        )
        kept.extend(rows[np.asarray(picked).reshape(-1)].tolist())
    return np.asarray(sorted(kept), dtype=np.int64)


def _decode_deployed_triplet(
    boxes: Any,
    scores: Any,
    labels: Any,
    meta: ImageMeta,
    *,
    threshold: float,
    sparse_coco_layout: bool,
) -> list[Detection]:
    import numpy as np

    boxes_array = _without_batch(np.asarray(boxes), last_dim=4)
    scores_array = _without_batch(np.asarray(scores)).reshape(-1)
    labels_array = _without_batch(np.asarray(labels)).reshape(-1).astype(np.int64)
    keep = scores_array >= threshold
    boxes_array = boxes_array[keep]
    scores_array = scores_array[keep]
    labels_array = labels_array[keep]
    # Official RT-DETR deployment returns pixel xyxy boxes.  RF-DETR's raw
    # export has only two outputs and therefore never enters this branch.
    return detections_from_arrays(
        boxes_array,
        scores_array,
        labels_array,
        meta,
        boxes_in_original_space=True,
        labels_are_coco_ids=sparse_coco_layout,
    )


def _decode_six_column_output(
    output: np.ndarray, meta: ImageMeta, *, threshold: float
) -> list[Detection]:
    import numpy as np

    keep = output[:, 4] >= threshold
    return detections_from_arrays(
        output[keep, :4], output[keep, 4], output[keep, 5].astype(np.int64), meta
    )


def _without_batch(array: np.ndarray, *, last_dim: int | None = None) -> np.ndarray:
    array = array
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    if last_dim is not None and (array.ndim != 2 or array.shape[-1] != last_dim):
        raise ValueError(f"Expected (*,{last_dim}) tensor, got {array.shape}")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(values, -88.0, 88.0)))
