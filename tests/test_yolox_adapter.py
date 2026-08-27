"""YOLOX adapter decode/postprocess tests with synthetic head outputs."""

from __future__ import annotations

import numpy as np
import pytest

from edgebench.models import get_detector
from edgebench.preprocessing import letterbox

STRIDE_8_ROWS = 80 * 80  # stride-8 grid rows precede stride-16/32 rows


def _raw_output(detections: list[tuple[int, float, float, int, float, float, float]]):
    """Build a synthetic (1, 8400, 85) YOLOX head output.

    Each detection is ``(grid_x, grid_y, obj, cls_index, cls, w, h)`` in
    stride-8 grid units; obj/cls are post-sigmoid scores.
    """
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    for grid_x, grid_y, obj, cls_index, cls, width, height in detections:
        row = grid_y * 80 + grid_x
        output[0, row, 0] = 0.0  # cx offset
        output[0, row, 1] = 0.0  # cy offset
        output[0, row, 2] = np.log(width / 8.0)
        output[0, row, 3] = np.log(height / 8.0)
        output[0, row, 4] = obj
        output[0, row, 5 + cls_index] = cls
    return output


@pytest.fixture()
def adapter():
    return get_detector("yolox_tiny")


def test_preprocess_layout_and_values(adapter) -> None:
    image = np.full((240, 640, 3), 200, dtype=np.uint8)
    tensor, meta = adapter.preprocess(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert meta.pad_top == 200.0
    # YOLOX: no normalization — content pixels keep their 0..255 value.
    assert tensor[0, 0, 300, 320] == pytest.approx(200.0)
    assert tensor[0, 0, 0, 0] == pytest.approx(114.0)


def test_postprocess_single_detection(adapter) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    # grid (40, 40) at stride 8 → center (320, 320) in input space,
    # box 80x40 → xyxy (280, 300, 360, 340); pad_top 80 → y shifts by -80.
    raw = _raw_output([(40, 40, 0.9, 2, 0.8, 80.0, 40.0)])
    detections = adapter.postprocess(raw, meta)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.score == pytest.approx(0.72, abs=1e-4)
    assert detection.class_id == 3  # contiguous index 2 → COCO id 3
    assert detection.bbox == pytest.approx((280.0, 220.0, 360.0, 260.0), abs=1.0)


def test_postprocess_confidence_filter(adapter) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    raw = _raw_output([(40, 40, 0.2, 0, 0.5, 80.0, 40.0)])  # 0.1 < 0.25
    assert adapter.postprocess(raw, meta) == []


def test_postprocess_nms_suppresses_duplicates(adapter) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    raw = _raw_output(
        [
            (40, 40, 0.9, 1, 0.9, 80.0, 40.0),
            (41, 40, 0.8, 1, 0.9, 80.0, 40.0),  # 8px-shifted overlapping duplicate
        ]
    )
    detections = adapter.postprocess(raw, meta)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.81, abs=1e-3)


def test_postprocess_rejects_bad_shape(adapter) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, meta = letterbox(image, (640, 640))
    with pytest.raises(ValueError, match="Unexpected YOLOX output shape"):
        adapter.postprocess(np.zeros((1, 8400, 42), dtype=np.float32), meta)


def test_load_pytorch_missing_checkpoint_raises(adapter) -> None:
    pytest.importorskip("torch")
    from edgebench.config import BenchmarkSettings, ModelConfig

    adapter.configure(
        ModelConfig(
            name="yolox_tiny",
            input_size=(640, 640),
            checkpoint="weights/yolox/does_not_exist.pth",
        ),
        BenchmarkSettings(),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        adapter.load_pytorch()
