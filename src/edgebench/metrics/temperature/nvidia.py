"""NVIDIA temperature provider stub."""

from edgebench.metrics._stub import StubCollector


class NvidiaTemperatureCollector(StubCollector):
    name = "nvidia_temperature"
