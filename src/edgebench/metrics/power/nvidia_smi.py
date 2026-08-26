"""nvidia-smi power provider stub."""

from edgebench.metrics._stub import StubCollector


class NvidiaSmiPowerCollector(StubCollector):
    name = "nvidia_smi_power"
