"""Accuracy evaluation helpers. Dataset adapters own official metrics."""

from __future__ import annotations

from typing import Any


def evaluate_detections(predictions: Any, dataset: Any) -> dict[str, float]:
    raise NotImplementedError("Accuracy evaluation is not implemented yet")
