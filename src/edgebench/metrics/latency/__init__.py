"""Model-only latency collector.

Brackets ``before_inference``/``after_inference`` around the runtime call.
The benchmark runner synchronizes GPU work inside that window, so samples
reflect completed inference. End-to-end latency is measured by the runner
itself; when both are present the runner's values win.
"""

from __future__ import annotations

import time
from typing import Any

from edgebench.metrics.base import MetricCollector


class LatencyCollector(MetricCollector):
    name = "latency"

    def __init__(self) -> None:
        self._samples_ms: list[float] = []
        self._start_ns: int | None = None

    def on_run_start(self) -> None:
        self._samples_ms = []
        self._start_ns = None

    def before_inference(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def after_inference(self) -> None:
        if self._start_ns is None:
            return
        elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self._samples_ms.append(elapsed_ms)
        self._start_ns = None

    def on_run_end(self) -> None:
        self._start_ns = None

    def result(self) -> dict[str, Any]:
        if not self._samples_ms:
            return {}
        # Deferred import: edgebench.benchmark.runner depends on the metrics
        # registry, so a module-level import would create a cycle.
        from edgebench.benchmark.latency import summarize_latencies

        stats = summarize_latencies(self._samples_ms)
        return {
            "latency_model_mean_ms": stats["mean_ms"],
            "latency_model_p50_ms": stats["p50_ms"],
            "latency_model_p95_ms": stats["p95_ms"],
            "latency_model_p99_ms": stats["p99_ms"],
            "fps_model_derived": 1000.0 / stats["mean_ms"],
        }
