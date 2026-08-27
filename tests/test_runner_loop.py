"""BenchmarkRunner.run loop tests with in-memory fakes. No datasets, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import edgebench.benchmark.runner as runner_module
from edgebench.benchmark import BenchmarkRunner
from edgebench.config import (
    BenchmarkSettings,
    DatasetConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    RuntimeConfig,
)
from edgebench.devices import DeviceRegistry
from edgebench.types import Detection, SupportStatus


class FakeAdapter:
    name = "fake_detector"

    def __init__(self) -> None:
        self.model_config: ModelConfig | None = None
        self.benchmark_settings: BenchmarkSettings | None = None
        self.preprocess_calls = 0

    def configure(self, model: ModelConfig, benchmark: BenchmarkSettings) -> None:
        self.model_config = model
        self.benchmark_settings = benchmark

    def preprocess(self, image: Any) -> tuple[np.ndarray, Any]:
        self.preprocess_calls += 1
        return np.zeros((1, 3, 640, 640), dtype=np.float32), {"image": image}

    def postprocess(self, raw_output: Any, metadata: Any) -> list[Detection]:
        return [Detection(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, class_id=1)]

    def load_pytorch(self) -> Any:
        raise NotImplementedError

    def export_onnx(self, output_path: str) -> None:
        raise NotImplementedError


class FakeRuntime:
    """Runtime stand-in: records call order, returns fixed outputs."""

    name = "fake_runtime"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None:
        self.calls.append("load")

    def warmup(self, input_data: Any) -> None:
        self.calls.append("warmup")

    def infer(self, input_data: Any) -> np.ndarray:
        self.calls.append("infer")
        return np.zeros((1, 85), dtype=np.float32)

    def synchronize(self) -> None:
        self.calls.append("synchronize")


class FakeDataset:
    def __init__(self, size: int = 4) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def image_id(self, index: int) -> int:
        return 1000 + index

    def get_image(self, index: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def evaluate(self, predictions: Any) -> dict[str, float]:
        assert set(predictions) == {1000 + i for i in range(self._size)}
        return {"map50": 0.5, "map50_95": 0.4}


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    adapter = FakeAdapter()
    runtime = FakeRuntime()
    dataset = FakeDataset(size=4)
    monkeypatch.setattr(runner_module, "get_detector", lambda name: adapter)
    monkeypatch.setattr(runner_module, "CocoDataset", lambda split: dataset)
    monkeypatch.setattr(
        runner_module, "get_runtime", lambda name, session=None: runtime
    )
    return {"adapter": adapter, "runtime": runtime, "dataset": dataset}


def _experiment(tmp_path: Path) -> ExperimentConfig:
    profile = DeviceRegistry.load().get("rtx_3060_laptop")
    return ExperimentConfig(
        experiment=ExperimentMeta(name="fake_run"),
        device="rtx_3060_laptop",
        dataset=DatasetConfig(name="coco", split="benchmark_500"),
        model=ModelConfig(
            name="fake_detector", input_size=(640, 640), pytorch_supported=True
        ),
        runtime=RuntimeConfig(name="pytorch", precision="fp32"),
        benchmark=BenchmarkSettings(warmup=3, iterations=10),
        metrics=["latency", "memory"],
        device_profile=profile,
    )


def test_run_executes_full_pipeline(
    fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "REPO_ROOT", tmp_path)
    result = BenchmarkRunner(_experiment(tmp_path)).run()
    runtime: FakeRuntime = fakes["runtime"]

    # Warm-up: exactly `warmup` calls, excluded from measurement; then the
    # dataset caps measured iterations at len(dataset) == 4.
    assert runtime.calls.count("warmup") == 3
    assert runtime.calls.count("infer") == 4
    assert runtime.calls[0] == "load"

    assert result.status is SupportStatus.OK
    assert result.latency_model_mean_ms is not None
    assert result.latency_e2e_mean_ms is not None
    assert result.latency_model_p99_ms is not None
    assert result.latency_e2e_mean_ms >= result.latency_model_mean_ms
    assert result.fps_model_derived == pytest.approx(1000.0 / result.latency_model_mean_ms)
    assert result.fps_e2e_measured is not None
    assert result.map50 == 0.5
    assert result.map50_95 == 0.4
    assert result.ram_peak_mb is not None and result.ram_peak_mb > 0
    assert result.device_metadata["images_measured"] == 4
    assert result.device_metadata["warmup_iterations"] == 3
    assert result.device_metadata["gpu_name"] == "NVIDIA GeForce RTX 3060 Laptop GPU"

    output = tmp_path / "results" / "raw" / "rtx_3060_laptop" / (
        "fake_detector_pytorch_fp32.json"
    )
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["input_size"] == [640, 640]
    assert payload["map50"] == 0.5


def test_run_unsupported_writes_nothing(
    fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "REPO_ROOT", tmp_path)
    profile = DeviceRegistry.load().get("raspberry_pi_4")
    experiment = ExperimentConfig(
        experiment=ExperimentMeta(name="blocked"),
        device="raspberry_pi_4",
        dataset=DatasetConfig(name="coco", split="benchmark_500"),
        model=ModelConfig(name="fake_detector", input_size=(640, 640)),
        runtime=RuntimeConfig(name="tensorrt", precision="fp16"),
        benchmark=BenchmarkSettings(),
        metrics=[],
        device_profile=profile,
    )
    result = BenchmarkRunner(experiment).run()
    assert result.status is SupportStatus.UNSUPPORTED
    assert not (tmp_path / "results").exists()


def test_run_empty_dataset_raises(
    fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner_module, "CocoDataset", lambda split: FakeDataset(size=0)
    )
    from edgebench.config import ConfigError

    with pytest.raises(ConfigError, match="empty"):
        BenchmarkRunner(_experiment(tmp_path)).run()
