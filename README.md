# Edge Object Detection Benchmark

Compare lightweight real-time detectors on Jetson Orin Nano Super, Raspberry Pi 4, and an RTX 3060 laptop GPU using the same input size, batch size, dataset split, and metrics.

The full research spec lives in [edge_object_detection_benchmark_README_v4.md](edge_object_detection_benchmark_README_v4.md).

## Install

```text
pip install -e .
pip install -e ".[dev]"
pip install -r requirements/common.txt   # numpy, opencv, pycocotools, psutil
```

Device-specific stacks (`torch`, `onnxruntime(-gpu)`, TensorRT, NCNN) are listed as comments in `requirements/` and installed per device.

Detector packages are optional and lazily imported. Put the official source
trees required by source-based adapters at the paths declared in
`configs/models/*.yaml`:

```text
third_party/
├── DAMO-YOLO/
├── PaddleDetection/
├── RT-DETR/
└── mmdetection/
```

Install current `ultralytics` and `rfdetr` packages for YOLO26n and
RF-DETR-Nano. Checkpoints remain local under `weights/`; the repository does
not download or redistribute model weights.

## Layout

```text
src/edgebench/          installable package
configs/devices/        hardware capability profiles
configs/models/         detector profiles
configs/experiments/    experiment YAML (example only)
datasets/               COCO annotations + deterministic split files (no images committed)
results/                raw / processed / summaries / figures
scripts/                split generation + benchmark drivers
tests/                  registry, capability, preprocessing, runner, reporting tests
```

Composition rule: dataset, detector, runtime, device, and metrics stay independent. Do not add classes such as `YOLOXTensorRTRunner` or `JetsonTensorRTRunner`.

## Usage

```bash
# one-time dataset setup (downloads are manual; see datasets/README.md)
python scripts/generate_coco_splits.py        # writes datasets/splits/*.txt, seed printed

# export artifacts on the target device (recorded deployment step)
python -m edgebench export yolox_tiny --to onnx
python -m edgebench export yolox_tiny --to tensorrt --precision fp16

# run one experiment
python -m edgebench run configs/experiments/example_rtmdet_jetson_trt.yaml

# run the full model × runtime matrix for a device
python scripts/run_all.py --device jetson_orin_nano_super

# aggregate raw results into the spec tables
python -m edgebench aggregate
python -m edgebench report
```

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

- `import edgebench`; registry, capability, and config loading
- COCO `val2017` adapter with deterministic splits (`coco_val2017_full.txt`, `coco_benchmark_500.txt`, seed `20240613`) and official pycocotools evaluation
- shared letterbox preprocessing and box rescaling with invertible metadata
- `BenchmarkRunner.run`: warm-up exclusion, synchronized model-only latency, end-to-end latency, accuracy evaluation, collector lifecycle, raw JSON persistence to `results/raw/<device>/`
- PyTorch, ONNX Runtime, TensorRT, and NCNN runtime backends (TensorRT/NCNN are on-device verified paths; lazy imports keep CPU-only machines usable)
- all seven detector adapters: model-specific preprocessing, decode/NMS,
  canonical COCO class mapping, lazy official-package checkpoint loading, and
  ONNX export; PicoDet uses PaddleDetection + `paddle2onnx` and is explicitly
  unsupported under PyTorch
- exporters: ONNX (`torch.onnx`), TensorRT (`tensorrt` API or `trtexec`), NCNN (`onnx2ncnn`, optional INT8 via `ncnn2table`/`ncnn2int8`)
- metric collectors: latency, memory (RSS/VRAM), nvidia-smi power/temperature/utilization, tegrastats power (Jetson), linux_sysfs temperature, CPU utilization
- CLI: `python -m edgebench run|export|aggregate|report`; `scripts/run_all.py` device matrix driver
- reporting: raw JSON aggregation → per-device spec tables with explicit `N/A — unsupported` rows, accuracy-vs-FPS figure

## External validation still required

The adapter contracts are covered by CPU-only synthetic tests, but real
checkpoint equivalence and speed must be correctness-gated on the target
devices. Install the optional upstream packages/source trees and official
checkpoints first, then export and compare PyTorch/ONNX/TensorRT/NCNN
detections model-by-model. `CustomIndustrialDataset` and the optional generic
`DeviceProbe` remain extension placeholders; neither is part of the COCO
benchmark matrix.

## Next implementation slices

1. Run YOLOX-Tiny across runtimes on each device and verify output equivalence
2. Validate the six new adapters in Phase 3 order, one official checkpoint at a time
3. Run the full device matrix and build analysis notebooks over `results/`
