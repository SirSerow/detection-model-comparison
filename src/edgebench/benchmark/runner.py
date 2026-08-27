"""Benchmark orchestrator.

Pipeline per measured image::

    image → adapter.preprocess → [sync] runtime.infer [sync]
          → adapter.postprocess → Detection (xyxy, COCO ids)

The runner measures model-only latency (synchronized inference window) and
end-to-end latency (full pipeline) itself, so both use one consistent
clock. ``LatencyCollector`` brackets the same window; when both produce
latency fields the runner's values win. Warm-up iterations are excluded
from every statistic. Unsupported combinations are recorded as
``BenchmarkResult.unsupported`` rather than raising.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from edgebench.config import ConfigError, ExperimentConfig
from edgebench.dataset_adapters import CocoDataset, DatasetAdapter
from edgebench.devices.base import DeviceProfile
from edgebench.devices.registry import DeviceRegistry
from edgebench.exporters import artifact_path_for
from edgebench.metrics.base import MetricCollector
from edgebench.metrics.registry import collectors_for
from edgebench.models import get_detector
from edgebench.paths import REPO_ROOT
from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig
from edgebench.runtimes.registry import get_runtime, list_runtimes
from edgebench.types import BenchmarkResult, Detection

if TYPE_CHECKING:
    pass

RESULT_FIELD_KEYS = frozenset(
    {
        "latency_model_mean_ms",
        "latency_model_p50_ms",
        "latency_model_p95_ms",
        "latency_model_p99_ms",
        "fps_model_derived",
        "latency_e2e_mean_ms",
        "latency_e2e_p50_ms",
        "latency_e2e_p95_ms",
        "fps_e2e_measured",
        "map50",
        "map50_95",
        "ram_peak_mb",
        "vram_peak_mb",
        "power_w",
    }
)


@dataclass(frozen=True)
class BoundBenchmark:
    """Resolved device/runtime/collector wiring for a supported experiment."""

    profile: DeviceProfile
    session: RuntimeSessionConfig
    runtime: RuntimeBackend
    collectors: tuple[MetricCollector, ...]


class BenchmarkRunner:
    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment

    def run(self) -> BenchmarkResult:
        """Execute the experiment, persist the raw result, and return it."""
        blocked = self.capability_result()
        if blocked is not None:
            return blocked

        from edgebench.benchmark.accuracy import evaluate_detections
        from edgebench.benchmark.latency import summarize_latencies
        from edgebench.results import ResultStore

        bound = self.bind()
        experiment = self.experiment
        settings = experiment.benchmark
        model = experiment.model
        profile = bound.profile
        runtime = bound.runtime
        collectors = list(bound.collectors)

        adapter = get_detector(model.name)
        adapter.configure(model, settings)
        dataset = self._dataset()
        if len(dataset) == 0:
            raise ConfigError(
                f"Dataset '{experiment.dataset.name}' split "
                f"'{experiment.dataset.split}' is empty"
            )

        # PyTorch-style backends execute the framework model object itself;
        # artifact-based backends load from session.artifact_path instead.
        attach_model = getattr(runtime, "attach_model", None)
        if callable(attach_model):
            attach_model(adapter.load_pytorch())

        load_start = time.perf_counter()
        runtime.load()
        load_seconds = time.perf_counter() - load_start

        first_input, _ = adapter.preprocess(dataset.get_image(0))
        warmup_start = time.perf_counter()
        for _ in range(settings.warmup):
            runtime.warmup(first_input)
        runtime.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start

        iterations = min(settings.iterations, len(dataset))
        for collector in collectors:
            collector.on_run_start()

        model_samples_ms: list[float] = []
        e2e_samples_ms: list[float] = []
        predictions: dict[int, list[Detection]] = {}
        wall_start = time.perf_counter()
        for index in range(iterations):
            e2e_start = time.perf_counter()
            image = dataset.get_image(index)
            input_data, meta = adapter.preprocess(image)

            for collector in collectors:
                collector.before_inference()
            runtime.synchronize()
            model_start = time.perf_counter()
            raw_output = runtime.infer(input_data)
            runtime.synchronize()
            model_end = time.perf_counter()
            for collector in collectors:
                collector.after_inference()

            detections = adapter.postprocess(raw_output, meta)
            e2e_end = time.perf_counter()

            model_samples_ms.append((model_end - model_start) * 1000.0)
            e2e_samples_ms.append((e2e_end - e2e_start) * 1000.0)
            image_id_getter = getattr(dataset, "image_id", None)
            image_id = image_id_getter(index) if callable(image_id_getter) else index
            predictions[int(image_id)] = detections
        wall_seconds = time.perf_counter() - wall_start

        for collector in collectors:
            collector.on_run_end()

        model_stats = summarize_latencies(model_samples_ms)
        e2e_stats = summarize_latencies(e2e_samples_ms)
        accuracy = evaluate_detections(predictions, dataset)

        fields: dict[str, Any] = {
            "latency_model_mean_ms": model_stats["mean_ms"],
            "latency_model_p50_ms": model_stats["p50_ms"],
            "latency_model_p95_ms": model_stats["p95_ms"],
            "latency_model_p99_ms": model_stats["p99_ms"],
            "fps_model_derived": 1000.0 / model_stats["mean_ms"],
            "latency_e2e_mean_ms": e2e_stats["mean_ms"],
            "latency_e2e_p50_ms": e2e_stats["p50_ms"],
            "latency_e2e_p95_ms": e2e_stats["p95_ms"],
            "fps_e2e_measured": iterations / wall_seconds,
            "map50": accuracy.get("map50"),
            "map50_95": accuracy.get("map50_95"),
        }
        collector_metrics: dict[str, Any] = {}
        for collector in collectors:
            for key, value in collector.result().items():
                if key in RESULT_FIELD_KEYS and value is not None and key not in fields:
                    fields[key] = value
                elif key not in RESULT_FIELD_KEYS:
                    collector_metrics[key] = value

        device_metadata: dict[str, Any] = {
            **profile.extra,
            "architecture": profile.architecture,
            "dataset": experiment.dataset.name,
            "split": experiment.dataset.split,
            "images_measured": iterations,
            "warmup_iterations": settings.warmup,
            "warmup_seconds": warmup_seconds,
            "model_load_seconds": load_seconds,
            "software": _software_versions(),
        }
        if collector_metrics:
            device_metadata["collector_metrics"] = collector_metrics

        result = BenchmarkResult(
            model=model.name,
            device=profile.name,
            runtime=experiment.runtime.name,
            precision=experiment.runtime.precision,
            input_size=model.input_size,
            batch_size=settings.batch_size,
            device_metadata=device_metadata,
            **fields,
        )
        store = ResultStore(REPO_ROOT / "results" / "raw")
        store.write(result)
        return result

    def _dataset(self) -> DatasetAdapter:
        dataset_config = self.experiment.dataset
        if dataset_config.name == "coco":
            return CocoDataset(split=dataset_config.split)
        raise ConfigError(
            f"Unknown dataset '{dataset_config.name}'; "
            "register a DatasetAdapter for it first"
        )

    def capability_result(self) -> BenchmarkResult | None:
        """Return an unsupported record, or None if the combination is allowed."""
        experiment = self.experiment
        profile = self._profile()
        runtime = experiment.runtime.name
        precision = experiment.runtime.precision
        model = experiment.model
        kwargs = {
            "model": model.name,
            "device": profile.name,
            "runtime": runtime,
            "precision": precision,
            "input_size": model.input_size,
            "batch_size": experiment.benchmark.batch_size,
        }
        if runtime == "pytorch" and not model.pytorch_supported:
            return BenchmarkResult.unsupported(
                reason=f"{model.name} does not support PyTorch",
                **kwargs,
            )
        if not profile.supports_runtime(runtime):
            return BenchmarkResult.unsupported(
                reason=f"{runtime} is not supported on {profile.name}",
                **kwargs,
            )
        if not profile.supports(runtime, precision):
            return BenchmarkResult.unsupported(
                reason=f"{runtime} {precision} is not supported on {profile.name}",
                **kwargs,
            )
        return None

    def bind(self) -> BoundBenchmark:
        """Resolve session options, runtime backend, and metric collectors."""
        blocked = self.capability_result()
        if blocked is not None:
            raise ConfigError(blocked.unsupported_reason or "unsupported combination")

        profile = self._profile()
        runtime_name = self.experiment.runtime.name
        if runtime_name not in list_runtimes():
            raise ConfigError(
                f"{runtime_name} is not a registered runtime backend; "
                "add a RuntimeBackend implementation"
            )

        backend = profile.backend(runtime_name, self.experiment.runtime.precision)
        if backend is None:
            raise ConfigError(
                f"{runtime_name} {self.experiment.runtime.precision} "
                f"is not supported on {profile.name}"
            )

        session = RuntimeSessionConfig(
            name=backend.runtime,
            precision=backend.precision,
            device_target=backend.device_target,
            execution_provider=backend.execution_provider,
            threads=(
                backend.threads if backend.threads is not None else profile.default_threads
            ),
        )
        if runtime_name != "pytorch":
            session = replace(
                session,
                artifact_path=str(
                    artifact_path_for(
                        self.experiment.model.name,
                        runtime_name,
                        backend.precision,
                        checkpoint=self.experiment.model.checkpoint,
                    )
                ),
            )
        runtime = get_runtime(session.name, session)
        collectors = tuple(collectors_for(profile, self.experiment.metrics))
        return BoundBenchmark(
            profile=profile,
            session=session,
            runtime=runtime,
            collectors=collectors,
        )

    def _profile(self) -> DeviceProfile:
        if self.experiment.device_profile is not None:
            return self.experiment.device_profile
        return DeviceRegistry.load().get(self.experiment.device)


def _software_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for package in ("torch", "onnxruntime", "tensorrt", "ncnn", "pycocotools", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions
