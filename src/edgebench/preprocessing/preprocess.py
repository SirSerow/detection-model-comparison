"""Shared decode → letterbox 640×640 → normalize pipeline.

``letterbox`` resizes with preserved aspect ratio and symmetric padding,
returning both the resized image and a :class:`LetterboxMeta` that carries
everything needed to map detections back to original-image coordinates.

Model-specific mean/std normalization and RGB/BGR conversion belong on the
detector adapter and must be documented there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

DEFAULT_PAD_COLOR = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to invert a letterbox transform.

    Convention:
        orig_width/orig_height: source image dimensions in pixels
        input_width/input_height: letterboxed network input dimensions
        ratio: scale applied to the source image (same for both axes)
        pad_left/pad_top: padding inserted before the scaled image
    """

    orig_width: int
    orig_height: int
    input_width: int
    input_height: int
    ratio: float
    pad_left: float
    pad_top: float


@dataclass(frozen=True)
class ResizeMeta:
    """Geometry for a direct (aspect-ratio changing) network resize."""

    orig_width: int
    orig_height: int
    input_width: int
    input_height: int

    @property
    def scale_x(self) -> float:
        return self.input_width / self.orig_width

    @property
    def scale_y(self) -> float:
        return self.input_height / self.orig_height


def letterbox(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
    *,
    color: tuple[int, int, int] = DEFAULT_PAD_COLOR,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize ``image`` (HWC ndarray) into ``size`` preserving aspect ratio.

    Args:
        image: HWC image ndarray (any channel order; pixels are not modified
            beyond resizing and padding).
        size: ``(width, height)`` of the network input.

    Returns:
        ``(resized_image, meta)`` where ``meta`` inverts the transform via
        :func:`edgebench.preprocessing.postprocess.rescale_boxes`.
    """
    import cv2

    input_width, input_height = int(size[0]), int(size[1])
    orig_height, orig_width = image.shape[:2]
    ratio = min(input_width / orig_width, input_height / orig_height)
    new_width = int(round(orig_width * ratio))
    new_height = int(round(orig_height * ratio))
    pad_width = input_width - new_width
    pad_height = input_height - new_height
    pad_left = pad_width / 2.0
    pad_top = pad_height / 2.0

    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    top = int(round(pad_top - 0.1))
    bottom = int(round(pad_top + 0.1))
    left = int(round(pad_left - 0.1))
    right = int(round(pad_left + 0.1))
    output = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    meta = LetterboxMeta(
        orig_width=orig_width,
        orig_height=orig_height,
        input_width=input_width,
        input_height=input_height,
        ratio=ratio,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return output, meta


def resize(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, ResizeMeta]:
    """Resize an HWC image directly to ``size`` and retain inverse geometry."""
    import cv2

    input_width, input_height = int(size[0]), int(size[1])
    orig_height, orig_width = image.shape[:2]
    output = cv2.resize(
        image, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    return output, ResizeMeta(
        orig_width=orig_width,
        orig_height=orig_height,
        input_width=input_width,
        input_height=input_height,
    )
