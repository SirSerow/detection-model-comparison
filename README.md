# Edge Object Detection Benchmark

Compare lightweight real-time detectors on Jetson Orin Nano Super, Raspberry Pi 4, and an RTX 3060 laptop GPU using the same input size, batch size, dataset split, and metrics.

This repository is a **scaffold**: package layout, YAML configs, registries, and abstract interfaces. Inference, export, dataset download, and evaluation are not implemented yet.

The full research spec lives in [edge_object_detection_benchmark_README_v4.md](edge_object_detection_benchmark_README_v4.md).

## Install

```text
pip install -e .
pip install -e ".[dev]"
```

Device-specific extras (`torch`, TensorRT, NCNN, and so on) are listed as comments in `requirements/` and are not required for the skeleton.

## Layout

```text
src/edgebench/          installable package
configs/devices/        hardware capability profiles
configs/models/         detector profiles
configs/experiments/    experiment YAML (example only)
datasets/               download instructions + split placeholders
results/                raw / processed / summaries / figures
scripts/                benchmark entry placeholders
tests/                  registry and capability tests
```

Composition rule: dataset, detector, runtime, device, and metrics stay independent. Do not add classes such as `YOLOXTensorRTRunner` or `JetsonTensorRTRunner`.

## Device extensibility

Adding a machine that reuses existing backends (`pytorch`, `onnxruntime`, `tensorrt`, `ncnn`) and metric providers (`nvidia_smi`, `tegrastats`, `linux_sysfs`, `nvidia`) requires only:

```text
configs/devices/new_device.yaml
```

The profile must declare a `backends` matrix (runtime × precision, plus execution provider / device target) rather than independent runtime and precision lists.

A new accelerator (RK3588 NPU, OpenVINO, Hailo) also needs:

```text
src/edgebench/runtimes/new_runtime.py
```

A new power/temperature API also needs a collector under `src/edgebench/metrics/`. Neither case should change `BenchmarkRunner`, detector adapters, COCO evaluation, or reporting.

## What works now

- `import edgebench`
- load device / model / experiment YAML (experiment inherits device warmup/iterations)
- list registered detectors and runtimes
- capability matrix checks (Raspberry Pi rejects TensorRT and PyTorch FP16)
- metric provider resolution from device YAML
- `BenchmarkRunner.run` records unsupported combinations as `BenchmarkResult.unsupported`

## What is stubbed

Supported `BenchmarkRunner.run` paths, dataset adapters, detector preprocess/postprocess, runtime `infer`, exporters, metric collectors, and result writing all raise `NotImplementedError`.

There is no working `python -m edgebench run` in this pass.

## Next implementation slices

1. Phase 0 — COCO `val2017` setup, deterministic 500-image split, `CocoDataset`
2. Phase 1 — shared preprocessing and result schema wiring
3. Phase 2 — YOLOX-Tiny baseline across runtimes
