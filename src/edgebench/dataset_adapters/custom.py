"""Placeholder for a later proprietary / industrial dataset."""

from __future__ import annotations

from typing import Any

from edgebench.dataset_adapters.base import DatasetAdapter


class CustomIndustrialDataset(DatasetAdapter):
    """Custom dataset hook. Intentionally unimplemented in the skeleton."""

    def __len__(self) -> int:
        raise NotImplementedError("CustomIndustrialDataset is not implemented yet")

    def get_sample(self, index: int) -> Any:
        raise NotImplementedError("CustomIndustrialDataset is not implemented yet")

    def get_image(self, index: int) -> Any:
        raise NotImplementedError("CustomIndustrialDataset is not implemented yet")

    def get_annotations(self, index: int) -> Any:
        raise NotImplementedError("CustomIndustrialDataset is not implemented yet")

    def evaluate(self, predictions: Any) -> dict[str, Any]:
        raise NotImplementedError("CustomIndustrialDataset is not implemented yet")
