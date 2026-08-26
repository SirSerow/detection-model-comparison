"""Registry and capability tests. No datasets or GPUs required."""

from __future__ import annotations

import edgebench
from edgebench.config import load_experiment, load_model_config
from edgebench.devices import DeviceRegistry
from edgebench.models import get_detector, list_detectors
from edgebench.paths import EXPERIMENTS_DIR
from edgebench.runtimes import list_runtimes

EXPECTED_DETECTORS = [
    "yolox_tiny",
    "yolo26n",
    "rtmdet_tiny",
    "damo_yolo_t",
    "picodet_s",
    "rtdetrv2_s",
    "rfdetr_nano",
]

EXPECTED_RUNTIMES = ["pytorch", "onnxruntime", "tensorrt", "ncnn"]

EXPECTED_DEVICES = [
    "jetson_orin_nano_super",
    "raspberry_pi_4",
    "rtx_3060_laptop",
]


def test_package_importable() -> None:
    assert edgebench.__version__ == "0.1.0"


def test_detector_registry() -> None:
    names = list_detectors()
    assert names == EXPECTED_DETECTORS
    for name in names:
        assert get_detector(name).name == name


def test_runtime_registry() -> None:
    assert list_runtimes() == EXPECTED_RUNTIMES


def test_device_profiles_load() -> None:
    registry = DeviceRegistry.load()
    assert registry.names() == sorted(EXPECTED_DEVICES)
    for name in EXPECTED_DEVICES:
        profile = registry.get(name)
        assert profile.name == name
        assert profile.supported_runtimes
        assert profile.supported_precisions


def test_raspberry_pi_rejects_tensorrt() -> None:
    registry = DeviceRegistry.load()
    pi = registry.get("raspberry_pi_4")
    jetson = registry.get("jetson_orin_nano_super")
    rtx = registry.get("rtx_3060_laptop")

    assert not pi.supports_runtime("tensorrt")
    assert pi.supports_runtime("ncnn")
    assert jetson.supports_runtime("tensorrt")
    assert rtx.supports_runtime("tensorrt")
    assert not jetson.supports_runtime("ncnn")


def test_picodet_pytorch_is_unsupported() -> None:
    config = load_model_config("picodet_s")
    assert config.framework == "paddle"
    assert config.pytorch_supported is False

    yolox = load_model_config("yolox_tiny")
    assert yolox.pytorch_supported is True


def test_example_experiment_loads() -> None:
    experiment = load_experiment(EXPERIMENTS_DIR / "example_rtmdet_jetson_trt.yaml")
    assert experiment.experiment.name == "rtmdet_jetson_trt"
    assert experiment.device == "jetson_orin_nano_super"
    assert experiment.runtime.name == "tensorrt"
    assert experiment.model.name == "rtmdet_tiny"
    assert experiment.model.input_size == (640, 640)
