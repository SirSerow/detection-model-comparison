"""NVIDIA GPU temperature provider via nvidia-smi."""

from __future__ import annotations

from edgebench.metrics._sampling import SamplingCollector
from edgebench.metrics.power._nvidia_smi import query_nvidia_smi


class NvidiaTemperatureCollector(SamplingCollector):
    name = "nvidia_temperature"

    def sample(self) -> dict[str, float] | None:
        values = query_nvidia_smi(["temperature.gpu"])
        temperature = values.get("temperature.gpu")
        return {"temperature_c": temperature} if temperature is not None else None
