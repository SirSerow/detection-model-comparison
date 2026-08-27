"""linux_sysfs temperature provider (Jetson, Raspberry Pi, generic Linux).

Reads ``/sys/class/thermal/thermal_zone*/temp`` and reports the mean of the
hottest zone in °C over the run.
"""

from __future__ import annotations

from pathlib import Path

from edgebench.metrics._sampling import SamplingCollector

THERMAL_GLOB = "/sys/class/thermal/thermal_zone*/temp"


class LinuxSysfsTemperatureCollector(SamplingCollector):
    name = "linux_sysfs_temperature"

    def sample(self) -> dict[str, float] | None:
        readings = []
        for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
            try:
                readings.append(int(path.read_text(encoding="utf-8").strip()) / 1000.0)
            except (ValueError, OSError):
                continue
        if not readings:
            return None
        return {"temperature_c": max(readings)}
