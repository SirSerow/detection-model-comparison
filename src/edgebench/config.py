"""YAML experiment / model / benchmark config loading.

Parsing only. This module does not run benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from edgebench.paths import CONFIGS_DIR, EXPERIMENTS_DIR, MODELS_DIR

if TYPE_CHECKING:
    from edgebench.devices.base import DeviceProfile


class ConfigError(ValueError):
    """Raised when a required YAML key is missing or invalid."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping in {path}")
    return data


def _require(data: dict[str, Any], key: str, source: Path | str) -> Any:
    if key not in data:
        raise ConfigError(f"Missing required key '{key}' in {source}")
    return data[key]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_size: tuple[int, int]
    framework: str = "pytorch"
    requires_nms: bool = True
    checkpoint: str | None = None
    upstream_config: str | None = None
    pytorch_supported: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: Path | str) -> ModelConfig:
        raw_size = _require(data, "input_size", source)
        if not (isinstance(raw_size, (list, tuple)) and len(raw_size) == 2):
            raise ConfigError(f"'input_size' must be [width, height] in {source}")
        return cls(
            name=str(_require(data, "name", source)),
            input_size=(int(raw_size[0]), int(raw_size[1])),
            framework=str(data.get("framework", "pytorch")),
            requires_nms=bool(data.get("requires_nms", True)),
            checkpoint=data.get("checkpoint"),
            upstream_config=data.get("upstream_config"),
            pytorch_supported=bool(data.get("pytorch_supported", True)),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    precision: str


@dataclass(frozen=True)
class BenchmarkSettings:
    batch_size: int = 1
    warmup: int = 50
    iterations: int = 500
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.65


@dataclass(frozen=True)
class ExperimentMeta:
    name: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentMeta
    device: str
    dataset: DatasetConfig
    model: ModelConfig
    runtime: RuntimeConfig
    benchmark: BenchmarkSettings
    metrics: list[str] = field(default_factory=list)
    device_profile: DeviceProfile | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    return _read_yaml(Path(path))


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.is_file():
        source = path
    else:
        source = MODELS_DIR / f"{name_or_path}.yaml"
    return ModelConfig.from_mapping(_read_yaml(source), source)


def load_benchmark_defaults(path: str | Path | None = None) -> BenchmarkSettings:
    source = Path(path) if path is not None else CONFIGS_DIR / "benchmark.yaml"
    data = _read_yaml(source)
    return BenchmarkSettings(
        batch_size=int(data.get("batch_size", 1)),
        warmup=int(data.get("warmup", 50)),
        iterations=int(data.get("iterations", 500)),
        input_width=int(data.get("input_width", 640)),
        input_height=int(data.get("input_height", 640)),
        confidence_threshold=float(data.get("confidence_threshold", 0.25)),
        iou_threshold=float(data.get("iou_threshold", 0.65)),
    )


def load_experiment(
    path: str | Path,
    *,
    devices_dir: str | Path | None = None,
    defaults_path: str | Path | None = None,
) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        candidate = EXPERIMENTS_DIR / source
        if candidate.is_file():
            source = candidate
    data = _read_yaml(source)

    experiment_data = _require(data, "experiment", source)
    dataset_data = _require(data, "dataset", source)
    model_data = _require(data, "model", source)
    runtime_data = _require(data, "runtime", source)
    benchmark_data = data.get("benchmark", {})
    if benchmark_data is None:
        benchmark_data = {}
    if not isinstance(benchmark_data, dict):
        raise ConfigError(f"'benchmark' must be a mapping in {source}")

    if isinstance(model_data, str):
        model = load_model_config(model_data)
    else:
        model = ModelConfig.from_mapping(model_data, source)

    device_name = str(_require(data, "device", source))
    from edgebench.devices.registry import DeviceRegistry

    profile = DeviceRegistry.load(devices_dir).get(device_name)
    defaults = load_benchmark_defaults(defaults_path)

    return ExperimentConfig(
        experiment=ExperimentMeta(name=str(_require(experiment_data, "name", source))),
        device=device_name,
        dataset=DatasetConfig(
            name=str(_require(dataset_data, "name", source)),
            split=str(_require(dataset_data, "split", source)),
        ),
        model=model,
        runtime=RuntimeConfig(
            name=str(_require(runtime_data, "name", source)),
            precision=str(_require(runtime_data, "precision", source)),
        ),
        benchmark=_merge_benchmark(defaults, profile, benchmark_data),
        metrics=list(data.get("metrics", [])),
        device_profile=profile,
    )


def _merge_benchmark(
    defaults: BenchmarkSettings,
    profile: DeviceProfile,
    experiment_data: dict[str, Any],
) -> BenchmarkSettings:
    """Merge global defaults ← device profile ← experiment YAML."""
    return BenchmarkSettings(
        batch_size=int(experiment_data.get("batch_size", defaults.batch_size)),
        warmup=int(experiment_data.get("warmup", profile.warmup)),
        iterations=int(experiment_data.get("iterations", profile.iterations)),
        input_width=int(experiment_data.get("input_width", defaults.input_width)),
        input_height=int(experiment_data.get("input_height", defaults.input_height)),
        confidence_threshold=float(
            experiment_data.get("confidence_threshold", defaults.confidence_threshold)
        ),
        iou_threshold=float(experiment_data.get("iou_threshold", defaults.iou_threshold)),
    )
