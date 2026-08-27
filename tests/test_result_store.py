"""ResultStore persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

from edgebench.results import ResultStore
from edgebench.types import BenchmarkResult, SupportStatus


def test_write_round_trip(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    result = BenchmarkResult(
        model="yolox_tiny",
        device="jetson_orin_nano_super",
        runtime="tensorrt",
        precision="fp16",
        input_size=(640, 640),
        latency_model_mean_ms=12.5,
        fps_model_derived=80.0,
        map50=0.51,
        map50_95=0.34,
        device_metadata={"gpu_name": "NVIDIA Jetson Orin Nano Super"},
    )
    path = store.write(result)
    assert path == tmp_path / "jetson_orin_nano_super" / "yolox_tiny_tensorrt_fp16.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["input_size"] == [640, 640]
    assert payload["latency_model_mean_ms"] == 12.5
    assert payload["power_w"] is None


def test_write_unsupported_record(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    result = BenchmarkResult.unsupported(
        model="picodet_s",
        device="raspberry_pi_4",
        runtime="pytorch",
        precision="fp32",
        input_size=(640, 640),
        reason="picodet_s does not support PyTorch",
    )
    path = store.write(result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "unsupported"
    assert payload["unsupported_reason"] == "picodet_s does not support PyTorch"


def test_write_is_atomic_overwrite(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    kwargs = {
        "model": "m",
        "device": "d",
        "runtime": "r",
        "precision": "fp32",
        "input_size": (640, 640),
    }
    first = store.write(BenchmarkResult(latency_model_mean_ms=1.0, **kwargs))
    second = store.write(BenchmarkResult(latency_model_mean_ms=2.0, **kwargs))
    assert first == second
    payload = json.loads(second.read_text(encoding="utf-8"))
    assert payload["latency_model_mean_ms"] == 2.0
    assert SupportStatus.OK.value == payload["status"]
    assert not list(first.parent.glob("*.tmp"))
