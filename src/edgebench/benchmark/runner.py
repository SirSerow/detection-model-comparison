"""Benchmark orchestrator stub.

Implemented in this slice:
    validate device/runtime/precision/model capability
    bind runtime session options and metric collectors

Not implemented:
    dataset load, warmup, timing, evaluation, result writing
"""

from __future__ import annotations

from dataclasses import dataclass

from edgebench.config import ConfigError, ExperimentConfig
from edgebench.devices.base import DeviceProfile
from edgebench.devices.registry import DeviceRegistry
from edgebench.metrics.base import MetricCollector
from edgebench.metrics.registry import collectors_for
from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig
from edgebench.runtimes.registry import get_runtime, list_runtimes
from edgebench.types import BenchmarkResult


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
        blocked = self.capability_result()
        if blocked is not None:
            return blocked
        raise NotImplementedError(
            "BenchmarkRunner.run is not implemented in the skeleton. "
            "Next slices: Phase 0 dataset, then YOLOX baseline."
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

