"""Reporting tests: aggregation, tables, and the run_all matrix builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from edgebench.reporting.aggregate import aggregate_results
from edgebench.reporting.tables import render_tables


def _write_result(directory: Path, **overrides) -> None:
    payload = {
        "model": "yolox_tiny",
        "device": "jetson_orin_nano_super",
        "runtime": "tensorrt",
        "precision": "fp16",
        "input_size": [640, 640],
        "batch_size": 1,
        "status": "ok",
        "unsupported_reason": None,
        "latency_model_mean_ms": 12.34,
        "latency_model_p50_ms": 12.1,
        "latency_model_p95_ms": 14.2,
        "latency_model_p99_ms": 15.0,
        "fps_model_derived": 81.0,
        "latency_e2e_mean_ms": 20.5,
        "latency_e2e_p50_ms": 20.1,
        "latency_e2e_p95_ms": 22.0,
        "fps_e2e_measured": 48.7,
        "map50": 0.512,
        "map50_95": 0.337,
        "ram_peak_mb": 1024.0,
        "vram_peak_mb": 512.0,
        "power_w": 9.5,
        "device_metadata": {},
    }
    payload.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{payload['model']}_{payload['runtime']}_{payload['precision']}.json"
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_and_render_tables(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_result(raw / "jetson_orin_nano_super")
    _write_result(
        raw / "jetson_orin_nano_super",
        model="picodet_s",
        runtime="pytorch",
        precision="fp16",
        status="unsupported",
        unsupported_reason="picodet_s does not support PyTorch",
        latency_model_mean_ms=None,
        fps_model_derived=None,
        map50=None,
        map50_95=None,
        power_w=None,
    )
    _write_result(raw / "raspberry_pi_4", device="raspberry_pi_4", runtime="ncnn")

    rows = aggregate_results(str(raw))
    assert len(rows) == 3

    markdown = render_tables(rows)
    assert "## jetson_orin_nano_super" in markdown
    assert "## raspberry_pi_4" in markdown
    assert "| yolox_tiny | tensorrt | fp16 | 81.0 | 12.34 |" in markdown
    assert "| picodet_s | pytorch | N/A — unsupported |" in markdown


def test_render_tables_empty() -> None:
    assert render_tables([]) == "\n"


def test_run_all_matrix_matches_device_backends() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_all", Path(__file__).resolve().parent.parent / "scripts" / "run_all.py"
    )
    assert spec is not None and spec.loader is not None
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    experiments = run_all.experiments_for_device("raspberry_pi_4", "benchmark_500")
    combos = {(e.model.name, e.runtime.name, e.runtime.precision) for e in experiments}
    # 7 model configs × 4 Pi backends (pytorch fp32, onnxruntime fp32,
    # ncnn fp32, ncnn int8).
    assert len(experiments) == 28
    assert ("yolox_tiny", "ncnn", "int8") in combos
    assert ("picodet_s", "pytorch", "fp32") in combos  # recorded unsupported at runtime
    assert all(e.benchmark.warmup == 20 for e in experiments)
    assert all(e.device_profile is not None for e in experiments)
