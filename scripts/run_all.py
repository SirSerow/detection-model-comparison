#!/usr/bin/env python3
"""Multi-device benchmark driver.

Runs every registered detector against every backend (runtime × precision)
in one device's profile. Unsupported combinations are recorded explicitly
as ``unsupported`` results (never silently omitted); failed runs are
reported and skipped so one broken model does not abort the matrix.

    python scripts/run_all.py --device jetson_orin_nano_super
    python scripts/run_all.py --device raspberry_pi_4 --split benchmark_500
"""

from __future__ import annotations

import argparse
import sys

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


def experiments_for_device(device: str, split: str) -> list[ExperimentConfig]:
    profile = DeviceRegistry.load().get(device)
    defaults = load_benchmark_defaults()
    model_names = sorted(path.stem for path in (CONFIGS_DIR / "models").glob("*.yaml"))
    experiments: list[ExperimentConfig] = []
    for model_name in model_names:
        model = load_model_config(model_name)
        for backend in profile.backends:
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
                        warmup=profile.warmup,
                        iterations=profile.iterations,
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
    args = parser.parse_args()

    store = ResultStore(REPO_ROOT / "results" / "raw")
    failures = 0
    experiments = experiments_for_device(args.device, args.split)
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
        print(
            f"OK        {label}: "
            f"{result.latency_model_mean_ms:.2f} ms, "
            f"mAP50-95 {result.map50_95:.3f}"
        )
    print(f"done: {len(experiments) - failures - 0} attempted, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
