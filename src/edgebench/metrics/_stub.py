"""Shared NotImplemented collector body."""

from __future__ import annotations

from typing import Any

from edgebench.metrics.base import MetricCollector


class StubCollector(MetricCollector):
    name = "metric"

    def on_run_start(self) -> None:
        raise NotImplementedError(f"{self.name} collector is not implemented yet")

    def before_inference(self) -> None:
        raise NotImplementedError(f"{self.name} collector is not implemented yet")

    def after_inference(self) -> None:
        raise NotImplementedError(f"{self.name} collector is not implemented yet")

    def on_run_end(self) -> None:
        raise NotImplementedError(f"{self.name} collector is not implemented yet")

    def result(self) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} collector is not implemented yet")
