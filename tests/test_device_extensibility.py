"""Config-only device addition, metric providers, and runner capability gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgebench.benchmark import BenchmarkRunner
from edgebench.config import (
    BenchmarkSettings,
    ConfigError,
    DatasetConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    RuntimeConfig,
    load_experiment,
    load_model_config,
)
from edgebench.devices import DeviceProfile, DeviceRegistry
from edgebench.metrics.registry import collectors_for, get_collector
from edgebench.runtimes import get_runtime
from edgebench.types import SupportStatus

GENERIC_CUDA_YAML = """
name: generic_cuda
architecture: x86_64

capabilities:
  cuda: true
  gpu: true

backends:
  - runtime: pytorch
    precision: fp16
    device_target: cuda
  - runtime: onnxruntime
    precision: fp16
    device_target: cuda
    execution_provider: CUDAExecutionProvider
  - runtime: tensorrt
    precision: fp16
    device_target: cuda

benchmark:
  warmup: 40
  iterations: 200

metrics:
  temperature:
    provider: nvidia
  power:
    provider: nvidia_smi
  utilization:
    provider: nvidia

metadata:
  gpu_name: Generic CUDA GPU
"""


def _write_generic_cuda(directory: Path) -> Path:
    path = directory / "generic_cuda.yaml"
    path.write_text(GENERIC_CUDA_YAML, encoding="utf-8")
    return path


def test_extra_device_yaml_loads_without_code_changes(tmp_path: Path) -> None:
    _write_generic_cuda(tmp_path)
    registry = DeviceRegistry.load(tmp_path)
    profile = registry.get("generic_cuda")
    assert profile.supports("tensorrt", "fp16")
    assert not profile.supports("ncnn", "fp32")
    assert profile.extra["gpu_name"] == "Generic CUDA GPU"
    assert profile.warmup == 40

    collectors = collectors_for(
        profile, ["latency", "memory", "power", "temperature", "utilization"]
    )
    names = [collector.name for collector in collectors]
    assert names == [
        "latency",
        "memory",
        "nvidia_smi_power",
        "nvidia_temperature",
        "nvidia_utilization",
    ]


def test_extra_device_does_not_drop_shipped_profiles(tmp_path: Path) -> None:
    from edgebench.paths import DEVICES_DIR

    for path in DEVICES_DIR.glob("*.yaml"):
        (tmp_path / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_generic_cuda(tmp_path)

    registry = DeviceRegistry.load(tmp_path)
    assert "generic_cuda" in registry.names()
    assert "jetson_orin_nano_super" in registry.names()
    assert "raspberry_pi_4" in registry.names()
    assert "rtx_3060_laptop" in registry.names()


def test_unknown_metric_provider_raises() -> None:
    with pytest.raises(ConfigError, match="Unknown metric provider"):
        get_collector("power", "does_not_exist")


def test_experiment_inherits_device_warmup(tmp_path: Path) -> None:
    path = tmp_path / "pi.yaml"
    path.write_text(
        """
experiment:
  name: pi_ort
device: raspberry_pi_4
dataset:
  name: coco
  split: benchmark_500
model: yolox_tiny
runtime:
  name: onnxruntime
  precision: fp32
metrics:
  - latency
""",
        encoding="utf-8",
    )
    experiment = load_experiment(path)
    assert experiment.benchmark.warmup == 20
    assert experiment.benchmark.iterations == 500
    assert experiment.benchmark.input_width == 640
    assert experiment.device_profile is not None
    assert experiment.device_profile.name == "raspberry_pi_4"


def test_experiment_overrides_device_warmup(tmp_path: Path) -> None:
    path = tmp_path / "pi.yaml"
    path.write_text(
        """
experiment:
  name: pi_ort_custom
device: raspberry_pi_4
dataset:
  name: coco
  split: benchmark_500
model: yolox_tiny
runtime:
  name: onnxruntime
  precision: fp32
benchmark:
  warmup: 7
""",
        encoding="utf-8",
    )
    experiment = load_experiment(path)
    assert experiment.benchmark.warmup == 7
    assert experiment.benchmark.iterations == 500


def _experiment(
    *,
    device: str,
    runtime: str,
    precision: str,
    model: ModelConfig | None = None,
    profile: DeviceProfile | None = None,
    metrics: list[str] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMeta(name="test"),
        device=device,
        dataset=DatasetConfig(name="coco", split="benchmark_500"),
        model=model or load_model_config("yolox_tiny"),
        runtime=RuntimeConfig(name=runtime, precision=precision),
        benchmark=experiment_benchmark(profile),
        metrics=metrics or ["latency", "memory", "power"],
        device_profile=profile,
    )


def experiment_benchmark(profile: DeviceProfile | None) -> BenchmarkSettings:
    if profile is None:
        return BenchmarkSettings()
    return BenchmarkSettings(warmup=profile.warmup, iterations=profile.iterations)


def test_runner_records_unsupported_pi_tensorrt() -> None:
    profile = DeviceRegistry.load().get("raspberry_pi_4")
    runner = BenchmarkRunner(
        _experiment(device="raspberry_pi_4", runtime="tensorrt", precision="fp16", profile=profile)
    )
    result = runner.run()
    assert result.status is SupportStatus.UNSUPPORTED
    assert result.unsupported_reason is not None
    assert "tensorrt" in result.unsupported_reason


def test_runner_records_unsupported_picodet_pytorch() -> None:
    profile = DeviceRegistry.load().get("jetson_orin_nano_super")
    runner = BenchmarkRunner(
        _experiment(
            device="jetson_orin_nano_super",
            runtime="pytorch",
            precision="fp16",
            model=load_model_config("picodet_s"),
            profile=profile,
        )
    )
    result = runner.run()
    assert result.status is SupportStatus.UNSUPPORTED
    assert result.unsupported_reason is not None
    assert "picodet_s" in result.unsupported_reason


def test_runner_records_unsupported_precision() -> None:
    profile = DeviceRegistry.load().get("raspberry_pi_4")
    runner = BenchmarkRunner(
        _experiment(device="raspberry_pi_4", runtime="pytorch", precision="fp16", profile=profile)
    )
    result = runner.run()
    assert result.status is SupportStatus.UNSUPPORTED
    assert result.unsupported_reason is not None
    assert "fp16" in result.unsupported_reason


def test_supported_combination_is_not_implemented() -> None:
    profile = DeviceRegistry.load().get("jetson_orin_nano_super")
    runner = BenchmarkRunner(
        _experiment(
            device="jetson_orin_nano_super",
            runtime="tensorrt",
            precision="fp16",
            profile=profile,
        )
    )
    with pytest.raises(NotImplementedError, match="Phase 0"):
        runner.run()


def test_bind_uses_session_options_from_profile() -> None:
    profile = DeviceRegistry.load().get("raspberry_pi_4")
    runner = BenchmarkRunner(
        _experiment(
            device="raspberry_pi_4",
            runtime="onnxruntime",
            precision="fp32",
            profile=profile,
            metrics=["latency", "temperature", "power"],
        )
    )
    bound = runner.bind()
    assert bound.runtime.name == "onnxruntime"
    assert bound.session.precision == "fp32"
    assert bound.session.execution_provider == "CPUExecutionProvider"
    assert bound.session.device_target == "cpu"
    assert bound.session.threads == 4
    assert [collector.name for collector in bound.collectors] == [
        "latency",
        "linux_sysfs_temperature",
    ]


def test_bind_extra_cuda_device_reuses_existing_backends(tmp_path: Path) -> None:
    _write_generic_cuda(tmp_path)
    profile = DeviceRegistry.load(tmp_path).get("generic_cuda")
    runner = BenchmarkRunner(
        _experiment(
            device="generic_cuda",
            runtime="onnxruntime",
            precision="fp16",
            profile=profile,
            metrics=["power", "utilization"],
        )
    )
    bound = runner.bind()
    assert bound.runtime.name == "onnxruntime"
    assert bound.session.execution_provider == "CUDAExecutionProvider"
    assert [collector.name for collector in bound.collectors] == [
        "nvidia_smi_power",
        "nvidia_utilization",
    ]


def test_get_runtime_accepts_session() -> None:
    from edgebench.runtimes import RuntimeSessionConfig

    session = RuntimeSessionConfig(
        name="pytorch",
        precision="fp16",
        device_target="cuda",
    )
    runtime = get_runtime("pytorch", session)
    assert runtime.name == "pytorch"
    assert runtime.session is session
