"""Synthetic output-contract tests for every non-YOLOX detector adapter."""

from __future__ import annotations

import numpy as np
import pytest

from edgebench.models import get_detector


def test_yolo26_preprocess_and_decode() -> None:
    adapter = get_detector("yolo26n")
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    image[..., 2] = 255  # BGR red becomes RGB channel zero.
    tensor, meta = adapter.preprocess(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)
    assert meta.pad_top == 160.0

    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, :4, 0] = [320.0, 320.0, 64.0, 32.0]
    output[0, 4 + 2, 0] = 0.9
    detections = adapter.postprocess(output, meta)
    assert len(detections) == 1
    assert detections[0].class_id == 3
    assert detections[0].bbox == pytest.approx((288.0, 144.0, 352.0, 176.0))


def test_damo_yolo_direct_resize_decode() -> None:
    adapter = get_detector("damo_yolo_t")
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    _, meta = adapter.preprocess(image)
    boxes = np.asarray([[[160.0, 160.0, 480.0, 480.0]]], dtype=np.float32)
    scores = np.zeros((1, 1, 80), dtype=np.float32)
    scores[0, 0, 0] = 0.8
    detections = adapter.postprocess((scores, boxes), meta)
    assert len(detections) == 1
    assert detections[0].class_id == 1
    assert detections[0].bbox == pytest.approx((160.0, 80.0, 480.0, 240.0))


def test_rtmdet_raw_head_decode() -> None:
    adapter = get_detector("rtmdet_tiny")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    _, meta = adapter.preprocess(image)
    cls_levels = [
        np.full((1, 80, size, size), -20.0, dtype=np.float32)
        for size in (8, 4, 2)
    ]
    box_levels = [
        np.zeros((1, 4, size, size), dtype=np.float32) for size in (8, 4, 2)
    ]
    cls_levels[0][0, 1, 4, 4] = 4.0
    box_levels[0][0, :, 4, 4] = [8.0, 8.0, 8.0, 8.0]
    detections = adapter.postprocess((tuple(cls_levels), tuple(box_levels)), meta)
    assert len(detections) == 1
    assert detections[0].class_id == 2
    assert detections[0].score == pytest.approx(0.982, abs=1e-3)
    # Input is configured at 640, so the 64px source was scaled by ten.
    assert detections[0].bbox == pytest.approx((2.4, 2.4, 4.0, 4.0))


def test_picodet_structured_input_and_deployed_output() -> None:
    adapter = get_detector("picodet_s")
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    inputs, meta = adapter.preprocess(image)
    assert set(inputs) == {"image", "im_shape", "scale_factor"}
    assert inputs["image"].shape == (1, 3, 640, 640)
    assert inputs["scale_factor"].tolist() == [[2.0, 1.0]]

    bbox = np.asarray([[2.0, 0.9, 100.0, 50.0, 200.0, 150.0]], dtype=np.float32)
    detections = adapter.postprocess((bbox, np.asarray([1])), meta)
    assert len(detections) == 1
    assert detections[0].class_id == 3
    assert detections[0].bbox == pytest.approx((100.0, 50.0, 200.0, 150.0))


def test_rtdetrv2_raw_output_decode() -> None:
    adapter = get_detector("rtdetrv2_s")
    _, meta = adapter.preprocess(np.zeros((320, 640, 3), dtype=np.uint8))
    boxes = np.asarray([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    logits = np.full((1, 1, 80), -20.0, dtype=np.float32)
    logits[0, 0, 4] = 3.0
    detections = adapter.postprocess((boxes, logits), meta)
    assert len(detections) == 1
    assert detections[0].class_id == 5
    assert detections[0].bbox == pytest.approx((160.0, 80.0, 480.0, 240.0))


def test_rfdetr_sparse_coco_class_layout() -> None:
    adapter = get_detector("rfdetr_nano")
    _, meta = adapter.preprocess(np.zeros((320, 640, 3), dtype=np.uint8))
    boxes = np.asarray([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    logits = np.full((1, 1, 91), -20.0, dtype=np.float32)
    logits[0, 0, 13] = 3.0  # Sparse COCO category id 13 (stop sign).
    detections = adapter.postprocess({"pred_boxes": boxes, "pred_logits": logits}, meta)
    assert len(detections) == 1
    assert detections[0].class_id == 13
    assert detections[0].bbox == pytest.approx((240.0, 80.0, 400.0, 240.0))


@pytest.mark.parametrize(
    "name",
    ["rtmdet_tiny", "damo_yolo_t", "yolo26n", "rtdetrv2_s", "rfdetr_nano"],
)
def test_missing_checkpoint_has_actionable_error(name: str) -> None:
    from edgebench.config import BenchmarkSettings, ModelConfig

    adapter = get_detector(name)
    adapter.configure(
        ModelConfig(
            name=name,
            input_size=(640, 640),
            checkpoint="weights/does-not-exist.pth",
        ),
        BenchmarkSettings(),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter.checkpoint_path()
