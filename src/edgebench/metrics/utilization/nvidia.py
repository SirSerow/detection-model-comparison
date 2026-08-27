"""NVIDIA GPU utilization provider via nvidia-smi."""

from __future__ import annotations

from edgebench.metrics._sampling import SamplingCollector
from edgebench.metrics.power._nvidia_smi import query_nvidia_smi


class NvidiaUtilizationCollector(SamplingCollector):
    name = "nvidia_utilization"

    def sample(self) -> dict[str, float] | None:
        values = query_nvidia_smi(["utilization.gpu"])
        utilization = values.get("utilization.gpu")
        if utilization is None:
            return None
        return {"gpu_utilization_pct": utilization}
