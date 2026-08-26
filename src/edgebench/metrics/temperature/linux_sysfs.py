"""linux_sysfs temperature provider stub."""

from edgebench.metrics._stub import StubCollector


class LinuxSysfsTemperatureCollector(StubCollector):
    name = "linux_sysfs_temperature"
