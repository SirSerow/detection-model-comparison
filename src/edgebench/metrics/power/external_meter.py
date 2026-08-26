"""External watt-meter provider stub."""

from edgebench.metrics._stub import StubCollector


class ExternalMeterPowerCollector(StubCollector):
    name = "external_meter_power"
