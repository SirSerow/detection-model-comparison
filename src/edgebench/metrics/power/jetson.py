"""Jetson tegrastats power provider stub."""

from edgebench.metrics._stub import StubCollector


class JetsonPowerCollector(StubCollector):
    name = "jetson_power"
