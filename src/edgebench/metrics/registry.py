"""Map device YAML provider ids to metric collector classes."""

from __future__ import annotations

from edgebench.config import ConfigError
from edgebench.devices.base import DeviceProfile
from edgebench.metrics.base import MetricCollector
from edgebench.metrics.latency import LatencyCollector
from edgebench.metrics.memory import MemoryCollector
from edgebench.metrics.power.external_meter import ExternalMeterPowerCollector
from edgebench.metrics.power.jetson import JetsonPowerCollector
from edgebench.metrics.power.nvidia_smi import NvidiaSmiPowerCollector
from edgebench.metrics.power.raspberry_pi import RaspberryPiPowerCollector
from edgebench.metrics.temperature.linux_sysfs import LinuxSysfsTemperatureCollector
from edgebench.metrics.temperature.nvidia import NvidiaTemperatureCollector
from edgebench.metrics.temperature.raspberry_pi import RaspberryPiTemperatureCollector
from edgebench.metrics.utilization.cpu import CPUUtilizationCollector
from edgebench.metrics.utilization.nvidia import NvidiaUtilizationCollector

GENERIC_METRICS = frozenset({"latency", "memory"})

PROVIDERS: dict[tuple[str, str], type[MetricCollector]] = {
    ("latency", "default"): LatencyCollector,
    ("memory", "default"): MemoryCollector,
    ("power", "tegrastats"): JetsonPowerCollector,
    ("power", "nvidia_smi"): NvidiaSmiPowerCollector,
    ("power", "raspberry_pi"): RaspberryPiPowerCollector,
    ("power", "external_meter"): ExternalMeterPowerCollector,
    ("temperature", "linux_sysfs"): LinuxSysfsTemperatureCollector,
    ("temperature", "nvidia"): NvidiaTemperatureCollector,
    ("temperature", "raspberry_pi"): RaspberryPiTemperatureCollector,
    ("utilization", "cpu"): CPUUtilizationCollector,
    ("utilization", "nvidia"): NvidiaUtilizationCollector,
}


def get_collector(kind: str, provider: str) -> MetricCollector:
    try:
        cls = PROVIDERS[(kind, provider)]
    except KeyError as exc:
        raise ConfigError(f"Unknown metric provider '{provider}' for '{kind}'") from exc
    return cls()


def collectors_for(profile: DeviceProfile, requested: list[str]) -> list[MetricCollector]:
    """Build collectors for requested metric kinds.

    Latency and memory default to generic collectors. Hardware metrics without a
    provider on the profile are skipped. An unknown provider id is an error.
    """
    collectors: list[MetricCollector] = []
    for kind in requested:
        provider = profile.metric_providers.get(kind)
        if provider is None and kind in GENERIC_METRICS:
            provider = "default"
        if provider is None:
            continue
        collectors.append(get_collector(kind, provider))
    return collectors
