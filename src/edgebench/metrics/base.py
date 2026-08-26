"""Metric collector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricCollector(ABC):
    @abstractmethod
    def on_run_start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def before_inference(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_inference(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_run_end(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> dict[str, Any]:
        raise NotImplementedError
