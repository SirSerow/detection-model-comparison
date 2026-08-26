"""Device capability profile. No inference code belongs here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    architecture: str
    has_cuda: bool
    has_gpu: bool
    supported_runtimes: list[str]
    supported_precisions: list[str]
    default_threads: int | None = None
    power_monitor: str | None = None
    warmup: int = 50
    iterations: int = 500
    metric_providers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_runtime(self, runtime: str) -> bool:
        return runtime in self.supported_runtimes

    def supports_precision(self, precision: str) -> bool:
        return precision in self.supported_precisions
