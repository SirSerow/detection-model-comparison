"""Benchmark orchestrator stub.

Intended loop (not implemented):
    validate device/runtime/precision capability
    load dataset, detector, runtime, collectors
    warmup (excluded from timing)
    measure model-only latency and end-to-end latency
    collect predictions
    evaluate
    write standardized result
"""

from __future__ import annotations

from edgebench.config import ExperimentConfig
from edgebench.types import BenchmarkResult


class BenchmarkRunner:
    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment

    def run(self) -> BenchmarkResult:
        raise NotImplementedError(
            "BenchmarkRunner.run is not implemented in the skeleton. "
            "Next slices: Phase 0 dataset, then YOLOX baseline."
        )
