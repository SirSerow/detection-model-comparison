"""Shared types for detections, capability status, and result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SupportStatus(str, Enum):
    """Whether a device/model/runtime combination can be executed."""

    OK = "ok"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Detection:
    """One detection in the canonical benchmark format.

    Convention:
        bbox: xyxy in original-image coordinates
        class_id: canonical COCO class id
    """

    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass
class BenchmarkResult:
    """Machine-readable record for a single model/runtime/device run."""

    model: str
    device: str
    runtime: str
    precision: str
    input_size: tuple[int, int]
    batch_size: int = 1
    status: SupportStatus = SupportStatus.OK
    unsupported_reason: str | None = None
    latency_model_mean_ms: float | None = None
    latency_model_p50_ms: float | None = None
    latency_model_p95_ms: float | None = None
    latency_model_p99_ms: float | None = None
    fps_model_derived: float | None = None
    latency_e2e_mean_ms: float | None = None
    latency_e2e_p50_ms: float | None = None
    latency_e2e_p95_ms: float | None = None
    fps_e2e_measured: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    ram_peak_mb: float | None = None
    vram_peak_mb: float | None = None
    power_w: float | None = None
    device_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unsupported(
        cls,
        *,
        model: str,
        device: str,
        runtime: str,
        precision: str,
        input_size: tuple[int, int],
        reason: str,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        return cls(
            model=model,
            device=device,
            runtime=runtime,
            precision=precision,
            input_size=input_size,
            batch_size=batch_size,
            status=SupportStatus.UNSUPPORTED,
            unsupported_reason=reason,
        )
