"""Load device YAML profiles and validate capabilities."""

from __future__ import annotations

from pathlib import Path

from edgebench.config import ConfigError, load_yaml
from edgebench.devices.base import DeviceProfile
from edgebench.paths import DEVICES_DIR


class DeviceRegistry:
    """Registry of device profiles loaded from ``configs/devices``."""

    def __init__(self, profiles: dict[str, DeviceProfile]) -> None:
        self._profiles = profiles

    @classmethod
    def load(cls, devices_dir: str | Path | None = None) -> DeviceRegistry:
        root = Path(devices_dir) if devices_dir is not None else DEVICES_DIR
        if not root.is_dir():
            raise ConfigError(f"Device config directory not found: {root}")
        profiles: dict[str, DeviceProfile] = {}
        for path in sorted(root.glob("*.yaml")):
            profile = cls._from_yaml(path)
            profiles[profile.name] = profile
        return cls(profiles)

    @classmethod
    def load_profile(
        cls,
        name: str,
        devices_dir: str | Path | None = None,
    ) -> DeviceProfile:
        registry = cls.load(devices_dir)
        return registry.get(name)

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def get(self, name: str) -> DeviceProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            known = ", ".join(self.names()) or "<none>"
            raise ConfigError(f"Unknown device '{name}'. Known: {known}") from exc

    @staticmethod
    def _from_yaml(path: Path) -> DeviceProfile:
        data = load_yaml(path)
        capabilities = data.get("capabilities", {})
        benchmark = data.get("benchmark", {})
        metrics = data.get("metrics", {})
        providers = {
            key: str(value.get("provider"))
            for key, value in metrics.items()
            if isinstance(value, dict) and "provider" in value
        }
        name = str(data.get("name") or path.stem)
        return DeviceProfile(
            name=name,
            architecture=str(data.get("architecture", "unknown")),
            has_cuda=bool(capabilities.get("cuda", False)),
            has_gpu=bool(capabilities.get("gpu", False)),
            supported_runtimes=list(data.get("supported_runtimes", [])),
            supported_precisions=list(data.get("supported_precisions", [])),
            default_threads=data.get("default_threads"),
            power_monitor=providers.get("power"),
            warmup=int(benchmark.get("warmup", 50)),
            iterations=int(benchmark.get("iterations", 500)),
            metric_providers=providers,
        )
