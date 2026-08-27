"""Shared helpers to map boxes back to original-image coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing.preprocess import LetterboxMeta, ResizeMeta


def rescale_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from letterboxed input to original-image coordinates.

    Args:
        boxes: ``(N, 4)`` xyxy array in letterboxed-input pixel coordinates.
        meta: the :class:`LetterboxMeta` produced alongside the input.

    Returns:
        ``(N, 4)`` xyxy array clipped to the original image bounds.
    """
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float64).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] -= meta.pad_left
    boxes[:, [1, 3]] -= meta.pad_top
    boxes /= meta.ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, meta.orig_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, meta.orig_height)
    return boxes


def rescale_resized_boxes(boxes: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    """Map xyxy boxes from a stretched input back to the source image."""
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float64).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] /= meta.scale_x
    boxes[:, [1, 3]] /= meta.scale_y
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, meta.orig_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, meta.orig_height)
    return boxes
