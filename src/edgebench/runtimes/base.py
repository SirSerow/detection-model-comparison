"""Runtime backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSessionConfig:
    """Device-independent session options for a runtime backend."""

    name: str
    precision: str
    device_target: str | None = None
    execution_provider: str | None = None
    threads: int | None = None
    artifact_path: str | None = None


class RuntimeBackend(ABC):
    """Given model input, execute the network."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, input_data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError
