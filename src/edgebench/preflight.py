"""On-device benchmark readiness checks used by ``edgebench doctor``."""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass
from pathlib import Path

from edgebench.config import load_model_config
from edgebench.devices import DeviceRegistry
from edgebench.exporters import artifact_path_for
from edgebench.metrics.temperature.raspberry_pi import (
    parse_temperature,
    parse_throttled,
    read_vcgencmd,
)
from edgebench.paths import CONFIGS_DIR, REPO_ROOT
from edgebench.validation import marker_matches_artifacts


@dataclass(frozen=True)
class PreflightCheck:
    level: str
    label: str
    detail: str


def _check(level: str, label: str, detail: str) -> PreflightCheck:
    return PreflightCheck(level=level, label=label, detail=detail)


def _backend_label(value: str) -> tuple[str, str]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid backend '{value}'; expected runtime:precision")
    return parts[0], parts[1]


def _architecture(value: str) -> str:
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(value.lower(), value.lower())


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        result[key] = value.strip().strip('"')
    return result


def collect_preflight(
    device: str,
    *,
    backend_labels: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[PreflightCheck]:
    profile = DeviceRegistry.load().get(device)
    checks: list[PreflightCheck] = []

    actual_arch = _architecture(platform.machine())
    expected_arch = _architecture(profile.architecture)
    checks.append(
        _check(
            "PASS" if actual_arch == expected_arch else "FAIL",
            "architecture",
            f"expected {expected_arch}, found {actual_arch}",
        )
    )

    release = _os_release()
    os_detail = " ".join(
        item for item in (release.get("PRETTY_NAME"), platform.release()) if item
    )
    if device == "raspberry_pi_4" and release.get("VERSION_ID") != "12":
        checks.append(
            _check("WARN", "operating system", os_detail or "unable to identify")
        )
    else:
        checks.append(_check("PASS", "operating system", os_detail))

    imports = {
        "numpy": "numpy",
        "OpenCV": "cv2",
        "pycocotools": "pycocotools",
        "psutil": "psutil",
    }
    for label, module in imports.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            checks.append(_check("FAIL", label, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(_check("PASS", label, "imported"))

    requested = (
        [_backend_label(value) for value in backend_labels]
        if backend_labels
        else [(item.runtime, item.precision) for item in profile.backends]
    )
    available = {(item.runtime, item.precision) for item in profile.backends}
    for runtime, precision in requested:
        label = f"backend {runtime}:{precision}"
        if (runtime, precision) not in available:
            checks.append(_check("FAIL", label, f"not declared by {device}"))
            continue
        if runtime == "onnxruntime":
            try:
                onnxruntime = importlib.import_module("onnxruntime")
                providers = onnxruntime.get_available_providers()
            except Exception as exc:
                checks.append(_check("FAIL", label, f"{type(exc).__name__}: {exc}"))
            else:
                ok = "CPUExecutionProvider" in providers
                checks.append(_check("PASS" if ok else "FAIL", label, str(providers)))
        elif runtime == "ncnn":
            try:
                importlib.import_module("ncnn")
            except Exception as exc:
                checks.append(_check("FAIL", label, f"{type(exc).__name__}: {exc}"))
            else:
                checks.append(_check("PASS", label, "ncnn imported"))

    available_models = sorted(
        path.stem for path in (CONFIGS_DIR / "models").glob("*.yaml")
    )
    selected_models = model_names or available_models
    for model_name in selected_models:
        if model_name not in available_models:
            checks.append(_check("FAIL", f"model {model_name}", "unknown model"))
            continue
        config = load_model_config(model_name)
        for runtime, precision in requested:
            if runtime == "pytorch":
                continue
            base = artifact_path_for(
                model_name,
                runtime,
                precision,
                checkpoint=config.checkpoint,
            )
            paths = (
                [base.with_suffix(".param"), base.with_suffix(".bin")]
                if runtime == "ncnn"
                else [base]
            )
            missing = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.is_file()]
            detail = ", ".join(missing) if missing else "present"
            checks.append(
                _check(
                    "FAIL" if missing else "PASS",
                    f"artifact {model_name} {runtime}:{precision}",
                    detail,
                )
            )
            if runtime == "ncnn" and precision == "fp32" and not missing:
                reference = artifact_path_for(
                    model_name,
                    "onnxruntime",
                    "fp32",
                    checkpoint=config.checkpoint,
                )
                valid, validation_detail = marker_matches_artifacts(base, reference)
                checks.append(
                    _check(
                        "PASS" if valid else "FAIL",
                        f"validation {model_name} ncnn:fp32",
                        validation_detail,
                    )
                )

    annotations = REPO_ROOT / "datasets/coco/annotations/instances_val2017.json"
    split = REPO_ROOT / "datasets/splits/coco_benchmark_500.txt"
    image_dir = REPO_ROOT / "datasets/coco/val2017"
    checks.append(
        _check(
            "PASS" if annotations.is_file() else "FAIL",
            "COCO annotations",
            str(annotations.relative_to(REPO_ROOT)),
        )
    )
    try:
        image_ids = [
            line.strip()
            for line in split.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        checks.append(_check("FAIL", "COCO benchmark split", str(exc)))
    else:
        missing_images = [
            item
            for item in image_ids
            if not (image_dir / f"{int(item):012d}.jpg").is_file()
        ]
        ok = len(image_ids) == 500 and not missing_images
        detail = f"{len(image_ids)} IDs; {len(missing_images)} images missing"
        checks.append(_check("PASS" if ok else "FAIL", "COCO benchmark split", detail))

    if device == "raspberry_pi_4":
        throttled = parse_throttled(read_vcgencmd("get_throttled"))
        checks.append(
            _check(
                "PASS" if throttled == 0 else "FAIL",
                "throttling",
                "unavailable" if throttled is None else f"0x{throttled:x}",
            )
        )
        temperature = parse_temperature(read_vcgencmd("measure_temp"))
        checks.append(
            _check(
                "PASS" if temperature is not None and temperature <= 55.0 else "FAIL",
                "start temperature",
                "unavailable" if temperature is None else f"{temperature:.1f} C",
            )
        )
        governor_path = Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        )
        try:
            governor = governor_path.read_text(encoding="utf-8").strip()
        except OSError:
            governor = "unavailable"
        checks.append(
            _check(
                "PASS" if governor == "performance" else "WARN",
                "CPU governor",
                governor,
            )
        )
        try:
            psutil = importlib.import_module("psutil")
            swap_used = int(psutil.swap_memory().used)
        except Exception:
            swap_used = -1
        checks.append(
            _check(
                "PASS" if swap_used == 0 else "WARN",
                "swap usage",
                "unavailable" if swap_used < 0 else f"{swap_used} bytes used",
            )
        )
    return checks


def run_preflight(
    device: str,
    *,
    backend_labels: list[str] | None = None,
    model_names: list[str] | None = None,
) -> int:
    try:
        checks = collect_preflight(
            device, backend_labels=backend_labels, model_names=model_names
        )
    except (KeyError, ValueError) as exc:
        print(f"FAIL  configuration: {exc}")
        return 2
    for item in checks:
        print(f"{item.level:<5} {item.label}: {item.detail}")
    failures = sum(item.level == "FAIL" for item in checks)
    warnings = sum(item.level == "WARN" for item in checks)
    print(f"preflight: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0
