"""Accuracy evaluation glue. Dataset adapters own the official metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgebench.dataset_adapters.base import DatasetAdapter
    from edgebench.types import Detection


def evaluate_detections(
    predictions: dict[int, list[Detection]], dataset: DatasetAdapter
) -> dict[str, float]:
    """Evaluate canonical predictions through the dataset adapter.

    Returns at least ``map50`` and ``map50_95`` for COCO-style datasets.
    """
    return dataset.evaluate(predictions)
