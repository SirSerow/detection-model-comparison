"""Device capability profile. No inference code belongs here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendSupport:
    """One legal runtime × precision combination on a device."""

    runtime: str
    precision: str
    device_target: str | None = None
    execution_provider: str | None = None
    threads: int | None = None


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    architecture: str
    has_cuda: bool
    has_gpu: bool
    backends: tuple[BackendSupport, ...]
    default_threads: int | None = None
    warmup: int = 50
    iterations: int = 500
    metric_providers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def supported_runtimes(self) -> list[str]:
        seen: list[str] = []
        for backend in self.backends:
            if backend.runtime not in seen:
                seen.append(backend.runtime)
        return seen

    @property
    def supported_precisions(self) -> list[str]:
        seen: list[str] = []
        for backend in self.backends:
            if backend.precision not in seen:
                seen.append(backend.precision)
        return seen

    @property
    def power_monitor(self) -> str | None:
        return self.metric_providers.get("power")

    def supports_runtime(self, runtime: str) -> bool:
        key = runtime.lower()
        return any(backend.runtime.lower() == key for backend in self.backends)

    def supports_precision(self, precision: str) -> bool:
        key = precision.lower()
        return any(backend.precision.lower() == key for backend in self.backends)

    def backend(self, runtime: str, precision: str) -> BackendSupport | None:
        runtime_key = runtime.lower()
        precision_key = precision.lower()
        for item in self.backends:
            if item.runtime.lower() == runtime_key and item.precision.lower() == precision_key:
                return item
        return None

    def supports(self, runtime: str, precision: str) -> bool:
        return self.backend(runtime, precision) is not None

