"""Letterbox geometry tests. CPU-only, no datasets required."""

from __future__ import annotations

import numpy as np

from edgebench.preprocessing import letterbox, rescale_resized_boxes, resize
from edgebench.preprocessing.preprocess import DEFAULT_PAD_COLOR


def test_letterbox_square_image_fills_input() -> None:
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    output, meta = letterbox(image, (640, 640))
    assert output.shape == (640, 640, 3)
    assert meta.ratio == 640 / 300
    assert meta.pad_left == 0.0
    assert meta.pad_top == 0.0
    assert meta.orig_width == 300
    assert meta.orig_height == 300


def test_letterbox_wide_image_pads_top_and_bottom() -> None:
    image = np.full((240, 640, 3), 255, dtype=np.uint8)
    output, meta = letterbox(image, (640, 640))
    assert output.shape == (640, 640, 3)
    assert meta.ratio == 1.0
    assert meta.pad_left == 0.0
    assert meta.pad_top == 200.0
    # Padding rows use the pad color, content rows keep the source pixel.
    assert tuple(output[0, 0]) == DEFAULT_PAD_COLOR
    assert tuple(output[320, 320]) == (255, 255, 255)
    assert tuple(output[-1, 0]) == DEFAULT_PAD_COLOR


def test_letterbox_tall_image_pads_left_and_right() -> None:
    image = np.full((640, 240, 3), 255, dtype=np.uint8)
    output, meta = letterbox(image, (640, 640))
    assert output.shape == (640, 640, 3)
    assert meta.pad_left == 200.0
    assert meta.pad_top == 0.0


def test_letterbox_non_square_target() -> None:
    image = np.full((100, 200, 3), 0, dtype=np.uint8)
    output, meta = letterbox(image, (416, 320))
    assert output.shape[1] == 416
    assert output.shape[0] == 320
    assert meta.input_width == 416
    assert meta.input_height == 320


def test_direct_resize_geometry_is_invertible() -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    output, meta = resize(image, (640, 640))
    boxes = np.asarray([[160.0, 160.0, 480.0, 480.0]], dtype=np.float32)
    assert output.shape == (640, 640, 3)
    assert np.allclose(
        rescale_resized_boxes(boxes, meta), [[100.0, 50.0, 300.0, 150.0]]
    )
