"""Load device YAML profiles and validate capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edgebench.config import ConfigError, load_yaml
from edgebench.devices.base import BackendSupport, DeviceProfile
from edgebench.paths import DEVICES_DIR

_RESERVED_KEYS = {
    "name",
    "architecture",
    "capabilities",
    "supported_runtimes",
    "supported_precisions",
    "backends",
    "benchmark",
    "metrics",
    "default_threads",
    "metadata",
}


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
            backends=_parse_backends(data, path),
            default_threads=_optional_int(data.get("default_threads")),
            warmup=int(benchmark.get("warmup", 50)),
            iterations=int(benchmark.get("iterations", 500)),
            metric_providers=providers,
            extra=_extra_metadata(data),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _extra_metadata(data: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        extra.update(metadata)
    for key, value in data.items():
        if key not in _RESERVED_KEYS:
            extra[key] = value
    return extra


def _parse_backends(data: dict[str, Any], path: Path) -> tuple[BackendSupport, ...]:
    raw = data.get("backends")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"Missing or empty 'backends' list in {path}")
    backends: list[BackendSupport] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"'backends[{index}]' must be a mapping in {path}")
        runtime = item.get("runtime")
        precision = item.get("precision")
        if not runtime or not precision:
            raise ConfigError(
                f"'backends[{index}]' requires 'runtime' and 'precision' in {path}"
            )
        backends.append(
            BackendSupport(
                runtime=str(runtime),
                precision=str(precision),
                device_target=_optional_str(item.get("device_target")),
                execution_provider=_optional_str(item.get("execution_provider")),
                threads=_optional_int(item.get("threads")),
            )
        )
    return tuple(backends)

