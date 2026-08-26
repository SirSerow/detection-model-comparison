"""MS COCO 2017 adapter. Download and evaluation are not implemented yet."""

from __future__ import annotations

from typing import Any

from edgebench.dataset_adapters.base import DatasetAdapter


class CocoDataset(DatasetAdapter):
    """COCO val2017 adapter.

    Phase 0 will load ``instances_val2017.json`` and the deterministic
    split files under ``datasets/splits/``.
    """

    def __len__(self) -> int:
        raise NotImplementedError("CocoDataset is not implemented yet")

    def get_sample(self, index: int) -> Any:
        raise NotImplementedError("CocoDataset is not implemented yet")

    def get_image(self, index: int) -> Any:
        raise NotImplementedError("CocoDataset is not implemented yet")

    def get_annotations(self, index: int) -> Any:
        raise NotImplementedError("CocoDataset is not implemented yet")

    def evaluate(self, predictions: Any) -> dict[str, Any]:
        raise NotImplementedError("CocoDataset is not implemented yet")
