from __future__ import annotations

from edgebench.reporting.plots import (
    _backend_style,
    _display_device,
    _display_model,
    _spread_label_positions,
)


def test_spread_label_positions_separates_nearby_accuracy_values() -> None:
    positions = _spread_label_positions(
        [("damo", 0.3685), ("yolo", 0.3679), ("yolox", 0.3412)],
        lower=0.30,
        upper=0.55,
        minimum_gap=0.015,
    )

    ordered = sorted(positions.values())
    assert all(right - left >= 0.015 for left, right in zip(ordered, ordered[1:]))
    assert ordered[0] >= 0.30
    assert ordered[-1] <= 0.55


def test_plot_labels_use_readable_names() -> None:
    assert _display_model("rfdetr_nano") == "RF-DETR Nano"
    assert _display_device("rtx_5070_laptop") == "RTX 5070 Laptop GPU"
    assert _display_model("new_model") == "New Model"


def test_backend_styles_distinguish_runtime_and_precision() -> None:
    assert _backend_style("onnxruntime", "fp32").marker == "s"
    assert not _backend_style("onnxruntime", "fp32").filled
    assert _backend_style("onnxruntime", "fp16").filled
    assert _backend_style("tensorrt", "fp16").marker == "D"
