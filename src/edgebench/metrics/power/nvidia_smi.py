"""nvidia-smi power provider (RTX 3060 laptop and desktop NVIDIA GPUs).

Polls ``nvidia-smi --query-gpu=power.draw`` on a background thread and
reports mean draw in watts. Requires the NVIDIA driver on PATH.
"""

from __future__ import annotations

from edgebench.metrics._sampling import SamplingCollector
from edgebench.metrics.power._nvidia_smi import query_nvidia_smi


class NvidiaSmiPowerCollector(SamplingCollector):
    name = "nvidia_smi_power"

    def sample(self) -> dict[str, float] | None:
        values = query_nvidia_smi(["power.draw"])
        power = values.get("power.draw")
        return {"power_w": power} if power is not None else None
