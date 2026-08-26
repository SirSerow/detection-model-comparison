"""NVIDIA utilization provider stub."""

from edgebench.metrics._stub import StubCollector


class NvidiaUtilizationCollector(StubCollector):
    name = "nvidia_utilization"
