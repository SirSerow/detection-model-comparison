"""CPU utilization provider stub."""

from edgebench.metrics._stub import StubCollector


class CPUUtilizationCollector(StubCollector):
    name = "cpu_utilization"
