#!/usr/bin/env python3
"""Multi-device benchmark driver.

Runs every registered detector against every backend (runtime × precision)
in one device's profile. Unsupported combinations are recorded explicitly
as ``unsupported`` results (never silently omitted); failed runs are
reported and skipped so one broken model does not abort the matrix.

    python scripts/run_all.py --device jetson_orin_nano_super
    python scripts/run_all.py --device raspberry_pi_4 --split benchmark_500 \
        --backend onnxruntime:fp32 --backend ncnn:fp32
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

from edgebench.benchmark import BenchmarkRunner
from edgebench.config import (
    BenchmarkSettings,
    DatasetConfig,
    ExperimentConfig,
    ExperimentMeta,
    RuntimeConfig,
    load_benchmark_defaults,
    load_model_config,
)
from edgebench.devices import DeviceRegistry
from edgebench.paths import CONFIGS_DIR, REPO_ROOT
from edgebench.results import ResultStore
from edgebench.types import SupportStatus

DEFAULT_METRICS = ["latency", "memory", "power", "temperature", "utilization"]


def _parse_backend(value: str) -> tuple[str, str]:
    try:
        runtime, precision = value.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "backend must be written as <runtime>:<precision>"
        ) from exc
    if not runtime or not precision:
        raise argparse.ArgumentTypeError(
            "backend must be written as <runtime>:<precision>"
        )
    return runtime, precision


def experiments_for_device(
    device: str,
    split: str,
    *,
    backends: Iterable[tuple[str, str]] | None = None,
    models: Iterable[str] | None = None,
    warmup: int | None = None,
    iterations: int | None = None,
) -> list[ExperimentConfig]:
    profile = DeviceRegistry.load().get(device)
    defaults = load_benchmark_defaults()
    selected_backends = set(backends) if backends is not None else None
    selected_models = set(models) if models is not None else None
    model_names = sorted(path.stem for path in (CONFIGS_DIR / "models").glob("*.yaml"))
    if selected_models is not None:
        unknown_models = selected_models.difference(model_names)
        if unknown_models:
            raise ValueError(f"unknown model(s): {', '.join(sorted(unknown_models))}")
        model_names = [name for name in model_names if name in selected_models]

    available_backends = {(item.runtime, item.precision) for item in profile.backends}
    if selected_backends is not None:
        unknown_backends = selected_backends.difference(available_backends)
        if unknown_backends:
            labels = ", ".join(
                f"{runtime}:{precision}"
                for runtime, precision in sorted(unknown_backends)
            )
            raise ValueError(f"backend(s) not supported by {device}: {labels}")

    experiments: list[ExperimentConfig] = []
    for model_name in model_names:
        model = load_model_config(model_name)
        for backend in profile.backends:
            if (
                selected_backends is not None
                and (backend.runtime, backend.precision) not in selected_backends
            ):
                continue
            experiments.append(
                ExperimentConfig(
                    experiment=ExperimentMeta(
                        name=f"{model_name}_{device}_{backend.runtime}_{backend.precision}"
                    ),
                    device=device,
                    dataset=DatasetConfig(name="coco", split=split),
                    model=model,
                    runtime=RuntimeConfig(
                        name=backend.runtime, precision=backend.precision
                    ),
                    benchmark=BenchmarkSettings(
                        batch_size=defaults.batch_size,
                        warmup=profile.warmup if warmup is None else warmup,
                        iterations=(
                            profile.iterations if iterations is None else iterations
                        ),
                        input_width=defaults.input_width,
                        input_height=defaults.input_height,
                        confidence_threshold=defaults.confidence_threshold,
                        iou_threshold=defaults.iou_threshold,
                    ),
                    metrics=list(DEFAULT_METRICS),
                    device_profile=profile,
                )
            )
    return experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", default="benchmark_500")
    parser.add_argument(
        "--backend",
        action="append",
        type=_parse_backend,
        help="limit the run to a runtime:precision pair; repeat as needed",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="limit the run to one model; repeat as needed",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations must be positive")

    store = ResultStore(REPO_ROOT / "results" / "raw")
    failures = 0
    try:
        experiments = experiments_for_device(
            args.device,
            args.split,
            backends=args.backend,
            models=args.model,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{len(experiments)} combinations for {args.device}")
    for experiment in experiments:
        label = (
            f"{experiment.model.name} / {experiment.runtime.name} "
            f"/ {experiment.runtime.precision}"
        )
        try:
            result = BenchmarkRunner(experiment).run()
        except Exception as exc:
            failures += 1
            print(f"FAILED    {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if result.status is SupportStatus.UNSUPPORTED:
            store.write(result)  # recorded explicitly, never omitted
            print(f"N/A       {label}: {result.unsupported_reason}")
            continue
        if result.status is SupportStatus.INVALID:
            failures += 1
            print(f"INVALID   {label}: {result.invalid_reason}", file=sys.stderr)
            continue
        print(
            f"OK        {label}: "
            f"{result.latency_model_mean_ms:.2f} ms, "
            f"mAP50-95 {result.map50_95:.3f}"
        )
    print(f"done: {len(experiments) - failures - 0} attempted, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
