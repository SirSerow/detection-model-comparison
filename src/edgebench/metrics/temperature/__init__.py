"""Temperature collectors."""

from edgebench.metrics.temperature.linux_sysfs import LinuxSysfsTemperatureCollector
from edgebench.metrics.temperature.nvidia import NvidiaTemperatureCollector

__all__ = ["LinuxSysfsTemperatureCollector", "NvidiaTemperatureCollector"]
