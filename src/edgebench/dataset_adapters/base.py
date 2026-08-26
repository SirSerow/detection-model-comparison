"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    """Expose samples and annotations without model-specific logic."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_image(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, index: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, predictions: Any) -> dict[str, Any]:
        raise NotImplementedError
