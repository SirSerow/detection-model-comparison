"""Correctness gate for converted runtime artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from edgebench.benchmark.accuracy import evaluate_detections
from edgebench.config import load_benchmark_defaults, load_model_config
from edgebench.dataset_adapters import CocoDataset
from edgebench.exporters import artifact_path_for
from edgebench.models import get_detector
from edgebench.runtimes import get_runtime
from edgebench.runtimes.base import RuntimeSessionConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_marker_path(base: Path) -> Path:
    return base.with_suffix(".validated.json")


def validate_ncnn_export(
    model_name: str,
    *,
    samples: int = 20,
    max_map_delta: float = 0.005,
) -> tuple[bool, dict[str, Any]]:
    if samples <= 0 or samples > 500:
        raise ValueError("samples must be between 1 and 500")
    config = load_model_config(model_name)
    settings = load_benchmark_defaults()
    adapter = get_detector(model_name)
    adapter.configure(config, settings)
    dataset = CocoDataset(split="benchmark_500")

    reference_path = artifact_path_for(
        model_name,
        "onnxruntime",
        "fp32",
        checkpoint=config.checkpoint,
    )
    target_base = artifact_path_for(
        model_name, "ncnn", "fp32", checkpoint=config.checkpoint
    )
    target_param = target_base.with_suffix(".param")
    target_bin = target_base.with_suffix(".bin")
    for path in (reference_path, target_param, target_bin):
        if not path.is_file():
            raise FileNotFoundError(f"validation artifact missing: {path}")

    reference = get_runtime(
        "onnxruntime",
        RuntimeSessionConfig(
            name="onnxruntime",
            precision="fp32",
            device_target="cpu",
            execution_provider="CPUExecutionProvider",
            threads=4,
            artifact_path=str(reference_path),
        ),
    )
    target = get_runtime(
        "ncnn",
        RuntimeSessionConfig(
            name="ncnn",
            precision="fp32",
            device_target="cpu",
            threads=4,
            artifact_path=str(target_base),
        ),
    )
    reference.load()
    target.load()

    reference_predictions = {}
    target_predictions = {}
    for index in range(samples):
        input_data, image_meta = adapter.preprocess(dataset.get_image(index))
        image_id = dataset.image_id(index)
        reference_predictions[image_id] = adapter.postprocess(
            reference.infer(input_data), image_meta
        )
        target_predictions[image_id] = adapter.postprocess(
            target.infer(input_data), image_meta
        )

    reference_accuracy = evaluate_detections(reference_predictions, dataset)
    target_accuracy = evaluate_detections(target_predictions, dataset)
    reference_map = float(reference_accuracy["map50_95"])
    target_map = float(target_accuracy["map50_95"])
    map_delta = abs(reference_map - target_map)
    passed = map_delta <= max_map_delta
    payload: dict[str, Any] = {
        "model": model_name,
        "runtime": "ncnn",
        "precision": "fp32",
        "reference_runtime": "onnxruntime",
        "samples": samples,
        "reference_map50_95": reference_map,
        "target_map50_95": target_map,
        "map50_95_delta": map_delta,
        "maximum_allowed_delta": max_map_delta,
        "passed": passed,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            str(reference_path.name): _sha256(reference_path),
            str(target_param.name): _sha256(target_param),
            str(target_bin.name): _sha256(target_bin),
        },
        "software": {},
    }
    for package in ("onnxruntime", "ncnn", "pnnx"):
        try:
            payload["software"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    marker = validation_marker_path(target_base)
    if passed:
        marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    return passed, payload


def marker_matches_artifacts(base: Path, reference: Path) -> tuple[bool, str]:
    marker = validation_marker_path(base)
    if not marker.is_file():
        return False, f"missing {marker.name}; run edgebench validate-export"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid validation marker: {exc}"
    if payload.get("passed") is not True:
        return False, "validation marker does not record a passing result"
    expected = payload.get("artifacts", {})
    paths = (reference, base.with_suffix(".param"), base.with_suffix(".bin"))
    for path in paths:
        if not path.is_file() or expected.get(path.name) != _sha256(path):
            return False, f"artifact changed after validation: {path.name}"
    return True, f"validated on {payload.get('samples', '?')} samples"
