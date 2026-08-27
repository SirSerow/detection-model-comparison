"""Box rescaling tests. CPU-only, no datasets required."""

from __future__ import annotations

import numpy as np

from edgebench.preprocessing import letterbox, rescale_boxes


def test_rescale_boxes_round_trip_square() -> None:
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    boxes = np.array([[10.0, 20.0, 110.0, 220.0]])
    restored = rescale_boxes(boxes, meta)
    expected = np.array([10.0, 20.0, 110.0, 220.0]) / meta.ratio
    assert np.allclose(restored[0], expected, atol=1.0)


def test_rescale_boxes_undoes_padding() -> None:
    # 640x240 image into 640x640: 200 px top padding, ratio 1.0.
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    boxes = np.array([[0.0, 200.0, 100.0, 300.0]])
    restored = rescale_boxes(boxes, meta)
    assert np.allclose(restored[0], [0.0, 0.0, 100.0, 100.0], atol=1e-6)


def test_rescale_boxes_clips_to_original_bounds() -> None:
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    # Box entirely inside the top padding maps to a degenerate clipped box.
    boxes = np.array([[10.0, 10.0, 50.0, 60.0]])
    restored = rescale_boxes(boxes, meta)
    assert restored[0, 1] == 0.0
    assert restored[0, 3] == 0.0
    assert restored[0, 2] <= 640.0


def test_rescale_boxes_empty_input() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    restored = rescale_boxes(np.empty((0, 4)), meta)
    assert restored.shape == (0, 4)
