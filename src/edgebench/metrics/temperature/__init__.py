"""Temperature collectors."""

from edgebench.metrics.temperature.linux_sysfs import LinuxSysfsTemperatureCollector
from edgebench.metrics.temperature.nvidia import NvidiaTemperatureCollector
from edgebench.metrics.temperature.raspberry_pi import RaspberryPiTemperatureCollector

__all__ = [
    "LinuxSysfsTemperatureCollector",
    "NvidiaTemperatureCollector",
    "RaspberryPiTemperatureCollector",
]
