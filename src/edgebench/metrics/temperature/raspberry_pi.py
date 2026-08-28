"""Raspberry Pi temperature and throttling telemetry via ``vcgencmd``.

``vcgencmd`` is the authoritative Raspberry Pi firmware interface. The
collector falls back to Linux thermal sysfs for temperature when the command
is unavailable, but throttling flags are never guessed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from edgebench.metrics._sampling import SamplingCollector

_TEMPERATURE_RE = re.compile(r"temp=([-+]?[0-9]*\.?[0-9]+)'C")
_THROTTLED_RE = re.compile(r"throttled=(0x[0-9a-fA-F]+)")


def read_vcgencmd(argument: str) -> str | None:
    """Return one stripped ``vcgencmd`` response, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["vcgencmd", argument],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    return completed.stdout.strip()


def parse_temperature(value: str | None) -> float | None:
    if not value:
        return None
    match = _TEMPERATURE_RE.fullmatch(value.strip())
    return float(match.group(1)) if match else None


def parse_throttled(value: str | None) -> int | None:
    if not value:
        return None
    match = _THROTTLED_RE.fullmatch(value.strip())
    return int(match.group(1), 16) if match else None


def read_sysfs_temperature() -> float | None:
    readings: list[float] = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            readings.append(int(path.read_text(encoding="utf-8").strip()) / 1000.0)
        except (ValueError, OSError):
            continue
    return max(readings) if readings else None


class RaspberryPiTemperatureCollector(SamplingCollector):
    name = "raspberry_pi_temperature"

    def sample(self) -> dict[str, float] | None:
        temperature = parse_temperature(read_vcgencmd("measure_temp"))
        if temperature is None:
            temperature = read_sysfs_temperature()
        throttled = parse_throttled(read_vcgencmd("get_throttled"))
        sample: dict[str, float] = {}
        if temperature is not None:
            sample["temperature_c"] = temperature
        if throttled is not None:
            sample["throttled_flags"] = float(throttled)
        return sample or None

    def aggregate(
        self, samples: list[dict[str, float]]
    ) -> dict[str, float | int | None]:
        temperatures = [
            sample["temperature_c"]
            for sample in samples
            if "temperature_c" in sample
        ]
        flags = [
            int(sample["throttled_flags"])
            for sample in samples
            if "throttled_flags" in sample
        ]
        combined_flags = 0
        for value in flags:
            combined_flags |= value
        return {
            "temperature_c": (
                sum(temperatures) / len(temperatures) if temperatures else None
            ),
            "temperature_c_max": max(temperatures) if temperatures else None,
            "throttled_flags": combined_flags if flags else None,
        }
