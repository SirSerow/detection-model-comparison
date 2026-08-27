"""CPU utilization provider via psutil (percent, mean over the run)."""

from __future__ import annotations

import psutil

from edgebench.metrics._sampling import SamplingCollector


class CPUUtilizationCollector(SamplingCollector):
    name = "cpu_utilization"

    def on_run_start(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the delta measurement
        super().on_run_start()

    def sample(self) -> dict[str, float] | None:
        return {"cpu_utilization_pct": float(psutil.cpu_percent(interval=None))}
