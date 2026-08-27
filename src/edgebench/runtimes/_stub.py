"""Shared NotImplemented runtime body."""

from __future__ import annotations

from typing import Any

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig


class StubRuntime(RuntimeBackend):
    def __init__(self, session: RuntimeSessionConfig | None = None) -> None:
        self.session = session

    @property
    def name(self) -> str:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError(f"{self.name} load is not implemented yet")

    def warmup(self, input_data: Any) -> None:
        raise NotImplementedError(f"{self.name} warmup is not implemented yet")

    def infer(self, input_data: Any) -> Any:
        raise NotImplementedError(f"{self.name} infer is not implemented yet")

    def synchronize(self) -> None:
        raise NotImplementedError(f"{self.name} synchronize is not implemented yet")
