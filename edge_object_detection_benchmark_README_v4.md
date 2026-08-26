# Edge Object Detection Benchmark

## Project Goal

Compare modern real-time object detection models across two edge-computing platforms using a consistent evaluation procedure.

The benchmark focuses on:

- inference speed;
- latency;
- accuracy;
- memory usage;
- power consumption;
- deployment complexity;
- portability between runtimes.

The same dataset, preprocessing pipeline, input resolution, batch size, and evaluation metrics should be used on both devices wherever possible.

---

## Target Devices

### 1. NVIDIA Jetson Orin Nano Super

Primary GPU-accelerated edge platform.

Target runtimes:

- PyTorch FP16;
- ONNX Runtime FP16 with CUDA Execution Provider;
- TensorRT FP16.

### 2. Raspberry Pi 4

CPU-only low-cost edge platform.

Target runtimes:

- PyTorch FP32;
- ONNX Runtime FP32 with CPU Execution Provider;
- NCNN FP32;
- optional NCNN INT8.

TensorRT is not available on Raspberry Pi 4 because it requires NVIDIA GPU hardware.

### 3. NVIDIA RTX 3060 Laptop GPU

Use the RTX 3060 laptop GPU as a higher-performance CUDA reference platform.

Primary runtimes:

- PyTorch FP16 with CUDA;
- ONNX Runtime FP16 with CUDA Execution Provider;
- TensorRT FP16.

Optional reference:

- PyTorch FP32.

Keep the common benchmark settings:

```text
Input resolution: 640 × 640
Batch size: 1
```

For reproducibility, the laptop must be plugged into AC power and the GPU power/performance mode must be documented. Laptop RTX 3060 variants can differ substantially in TGP, cooling, and boost behavior.


PyTorch FP16 is not used as the main Raspberry Pi runtime because the Pi 4 CPU is better suited to FP32 inference for this benchmark.

---

## Models

The benchmark is intentionally weighted toward lightweight CNN-based detectors, with two transformer-based models retained as secondary architectural references.

The initial model set is:

1. YOLOX-Tiny
2. YOLO26n
3. RTMDet-Tiny
4. DAMO-YOLO-T
5. PP-PicoDet-S
6. RT-DETRv2-S
7. RF-DETR-Nano

### Purpose of each model

| Model | Architecture focus | Role in benchmark |
|---|---|---|
| YOLOX-Tiny | CNN | Existing baseline and older real-time detector |
| YOLO26n | CNN-based real-time detector | Modern YOLO speed-oriented reference |
| RTMDet-Tiny | CNN | Primary modern lightweight CNN alternative |
| DAMO-YOLO-T | CNN | Lightweight NAS/reparameterization-based detector designed for efficient deployment |
| PP-PicoDet-S | Ultra-light CNN | CPU/mobile-focused detector; especially relevant to Raspberry Pi |
| RT-DETRv2-S | CNN + Transformer | Secondary modern NMS-free architecture reference |
| RF-DETR-Nano | Transformer | Secondary accuracy-oriented modern architecture reference |

### CNN-focused core comparison

The primary research comparison should emphasize:

```text
YOLOX-Tiny
YOLO26n
RTMDet-Tiny
DAMO-YOLO-T
PP-PicoDet-S
```

RT-DETRv2-S and RF-DETR-Nano should remain in the project, but can be treated as a separate transformer/hybrid comparison rather than the central focus.

### Licensing note

The benchmark prioritizes models with permissive/open-source implementations where possible.

- YOLOX: Apache 2.0
- RTMDet / MMDetection: Apache 2.0
- DAMO-YOLO: Apache 2.0
- PaddleDetection / PP-PicoDet implementation: Apache 2.0
- RT-DETRv2: Apache 2.0
- RF-DETR core models: permissive/open-source; verify the exact checkpoint/model variant before redistribution
- YOLO26: include as a performance reference, but review its current Ultralytics licensing separately before commercial use

For any commercial redistribution of pretrained weights, verify the license attached to the exact checkpoint in addition to the source-code repository license.

---

## Input Configuration

The image size must be identical across both devices and across all models for the main benchmark.

### Primary configuration

```text
Input resolution: 640 × 640
Batch size: 1
```

All models should receive the same preprocessed image dimensions.

If a model normally uses a different native resolution, it should still be evaluated at 640 × 640 for the normalized comparison where technically supported.

Optional native-resolution results may be reported separately, but must not replace the normalized 640 × 640 comparison.

---

## Dataset

### Primary dataset: MS COCO 2017

Use **MS COCO 2017 validation (`val2017`)** as the standard benchmark dataset.

Reason for choosing COCO:

- strong existing labels;
- no manual annotation is required;
- almost all selected detectors have COCO-pretrained checkpoints;
- standardized object-detection metrics are already well established;
- wide variety of object sizes, categories, backgrounds, occlusion, and scene types;
- results are easier to compare with published detector benchmarks.

The benchmark should not retrain the models initially. Use official or well-established COCO-pretrained checkpoints wherever possible.

### Accuracy set

Use the complete:

```text
COCO val2017
5,000 images
80 classes
```

for final accuracy evaluation.

This set is used for:

- mAP@0.50:0.95;
- AP50;
- AP75;
- AP-small;
- AP-medium;
- AP-large;
- precision and recall where useful.

### Performance subset

For latency and FPS measurements, use a deterministic subset of **500 images** sampled from `val2017`.

Do not randomly resample this set for every benchmark run.

Create the image list once using a fixed seed and reuse exactly the same image IDs on:

- Jetson Orin Nano Super;
- NVIDIA RTX 3060 Laptop GPU;
- Raspberry Pi 4;
- all models;
- all runtimes.

Suggested layout:

```text
datasets/
├── coco/
│   ├── annotations/
│   │   └── instances_val2017.json
│   └── val2017/
│
└── splits/
    ├── coco_val2017_full.txt
    └── coco_benchmark_500.txt
```

### Warm-up

Warm-up samples must not be included in measured timing statistics.

Recommended:

```text
Jetson:
  warm-up: 50 iterations
  measured: 500 images

Raspberry Pi:
  warm-up: 20 iterations
  measured: 500 images
```

If a model is prohibitively slow on Raspberry Pi 4, keep the same image resolution and batch size, but the number of measured images may be reduced only if clearly documented. Prefer keeping 500 images for consistency.

### Dataset fairness rules

Use:

- identical source images;
- identical annotations;
- identical 640 × 640 input size;
- identical batch size = 1;
- identical benchmark image IDs;
- identical confidence threshold where applicable;
- identical IoU threshold where applicable;
- deterministic preprocessing where possible.

Do not compare models using different validation subsets.

### Native versus normalized input

The primary benchmark is normalized:

```text
all models → 640 × 640
```

Some models may have a different recommended native inference resolution.

Native-resolution results may be added as a secondary experiment, but they must be reported separately from the normalized benchmark.

Do not mix native-resolution and normalized-resolution results in the same comparison table.

### Dataset licensing note

COCO annotations are openly distributed, but the source images originate from Flickr and retain their original image licenses.

For this project:

- use COCO for benchmarking and evaluation;
- do not redistribute the complete image dataset inside the repository;
- provide download/setup instructions instead;
- store only image IDs, split files, scripts, and derived benchmark results in the project repository.

### Dataset abstraction

The benchmark code should not be tightly coupled to COCO.

Use a dataset adapter/interface so that a custom industrial dataset can be added later without changing the model/runtime benchmark code.

Suggested design:

```text
DatasetAdapter
├── CocoDataset
└── CustomIndustrialDataset
```

The dataset adapter should provide at minimum:

```python
class DatasetAdapter:
    def __len__(self):
        ...

    def get_image(self, index):
        ...

    def get_annotations(self, index):
        ...

    def evaluate(self, predictions):
        ...
```

This allows the same benchmark framework to be reused later with proprietary or industrial datasets.

---

## Preprocessing

Create one shared preprocessing specification.

Recommended pipeline:

```text
image
  ↓
decode
  ↓
resize / letterbox to 640 × 640
  ↓
color conversion if required
  ↓
normalization
  ↓
tensor/runtime-specific conversion
```

Record any model-specific preprocessing differences.

Where model architectures require different normalization constants or tensor layouts, document them explicitly.

---

# Benchmark 1: Jetson Orin Nano Super

## Runtime Matrix

| Model | PyTorch FP16 | ONNX Runtime FP16 | TensorRT FP16 |
|---|---:|---:|---:|
| YOLOX-Tiny | | | |
| YOLO26n | | | |
| RTMDet-Tiny | | | |
| DAMO-YOLO-T | | | |
| PP-PicoDet-S | N/A* | | |
| RT-DETRv2-S | | | |
| RF-DETR-Nano | | | |

\* PP-PicoDet is Paddle-native. Do not create an artificial PyTorch benchmark solely for symmetry. Benchmark its exported ONNX/TensorRT path instead.

### PyTorch

Recommended configuration:

```text
device = CUDA
precision = FP16
batch = 1
input = 640 × 640
```

### ONNX Runtime

Recommended configuration:

```text
Execution Provider = CUDAExecutionProvider
precision = FP16
batch = 1
input = 640 × 640
```

### TensorRT

Recommended configuration:

```text
precision = FP16
batch = 1
input = 640 × 640
```

Prefer static input shapes for the main benchmark.

---

# Benchmark 3: NVIDIA RTX 3060 Laptop GPU

## Runtime Matrix

| Model | PyTorch FP16 | ONNX Runtime FP16 | TensorRT FP16 | Optional PyTorch FP32 |
|---|---:|---:|---:|---:|
| YOLOX-Tiny | | | | |
| YOLO26n | | | | |
| RTMDet-Tiny | | | | |
| DAMO-YOLO-T | | | | |
| PP-PicoDet-S | N/A* | | | N/A* |
| RT-DETRv2-S | | | | |
| RF-DETR-Nano | | | | |

\* PP-PicoDet is Paddle-native. Benchmark it through ONNX/TensorRT rather than forcing an artificial PyTorch implementation.

### Recommended configuration

```text
PyTorch:
  device = CUDA
  precision = FP16

ONNX Runtime:
  provider = CUDAExecutionProvider
  precision = FP16

TensorRT:
  precision = FP16
  static input = 640 × 640
```

### Laptop-specific controls

Record and control:

- laptop model;
- GPU name;
- VRAM size;
- RTX 3060 laptop TGP/power limit if available;
- NVIDIA driver version;
- CUDA version;
- cuDNN version;
- TensorRT version;
- ONNX Runtime version;
- OS;
- power profile;
- thermal/performance mode;
- GPU temperature;
- GPU clocks.

Use AC power and a high-performance profile during all measurements.

---

# Benchmark 2: Raspberry Pi 4

## Runtime Matrix

| Model | PyTorch FP32 | ONNX Runtime FP32 | NCNN FP32 | NCNN INT8 |
|---|---:|---:|---:|---:|
| YOLOX-Tiny | | | | |
| YOLO26n | | | | |
| RTMDet-Tiny | | | | |
| DAMO-YOLO-T | | | | |
| PP-PicoDet-S | N/A* | | | |
| RT-DETRv2-S | | | | |
| RF-DETR-Nano | | | | |

\* PP-PicoDet is Paddle-native. Its Raspberry Pi comparison should focus on ONNX Runtime and especially NCNN rather than PyTorch.

### PyTorch

Recommended configuration:

```text
device = CPU
precision = FP32
batch = 1
input = 640 × 640
```

### ONNX Runtime

Recommended configuration:

```text
Execution Provider = CPUExecutionProvider
precision = FP32
batch = 1
input = 640 × 640
```

### NCNN

Recommended configurations:

```text
FP32
batch = 1
input = 640 × 640
```

and optionally:

```text
INT8
batch = 1
input = 640 × 640
```

INT8 results should be reported separately because quantization may affect accuracy.

---

# Metrics

Collect the same metrics wherever technically possible.

## Performance

For each model/runtime combination record:

- average inference latency;
- median latency (P50);
- P95 latency;
- FPS;
- warm-up time;
- model load time.

Use synchronized timing for GPU runtimes.

---

## Accuracy

Recommended metrics:

- mAP@0.50;
- mAP@0.50:0.95;
- precision;
- recall;
- per-class AP.

Accuracy should be measured using the same dataset and post-processing thresholds.

---

## Resource Usage

Record:

- peak RAM;
- peak GPU memory where applicable;
- CPU utilization;
- GPU utilization where applicable;
- device temperature;
- power consumption where measurable.

---

## Deployment Metrics

Also record qualitative deployment information:

- export difficulty;
- unsupported operators;
- custom plugins required;
- conversion errors;
- runtime-specific modifications;
- dependency complexity;
- build time;
- model file size.

Suggested rating:

```text
1 = very easy
2 = easy
3 = moderate
4 = difficult
5 = very difficult
```

---

# Latency Definitions

Two latency measurements should be kept separate.

## 1. Model-only latency

Measure only the neural network execution.

```text
preprocessed input
      ↓
model inference
      ↓
raw model output
```

This is useful for comparing architecture/runtime efficiency.

---

## 2. End-to-end latency

Measure the complete detection pipeline.

```text
input image
    ↓
decode
    ↓
resize / letterbox
    ↓
normalization
    ↓
tensor conversion
    ↓
device transfer
    ↓
model inference
    ↓
post-processing
    ↓
NMS if required
    ↓
final detections
```

This is the more important metric for real-world deployment.

---

# FPS Measurement

Do not calculate FPS from a single inference.

Recommended process:

1. Load model.
2. Run warm-up iterations.
3. Run at least 100 measured iterations.
4. Synchronize the GPU before and after timing on CUDA/TensorRT.
5. Record individual inference times.
6. Calculate:
   - mean;
   - median;
   - P95;
   - standard deviation;
   - FPS.

Suggested:

```text
warm-up iterations: 20–50
measured iterations: >= 100
```

For more stable results, use 500–1000 iterations.

---

# Fairness Rules

To keep the comparison valid:

1. Use the same 640 × 640 input resolution.
2. Use batch size 1.
3. Use the same dataset.
4. Use identical confidence thresholds where possible.
5. Use identical IoU thresholds where possible.
6. Include post-processing in end-to-end measurements.
7. Report whether NMS is required.
8. Use the same device power/performance mode for all models.
9. Run models after warm-up.
10. Avoid background processes during benchmarking.
11. Record software versions.
12. Record model commit/version/checkpoint.
13. Do not mix native-resolution results with normalized-resolution results.

---

# Device Metadata

Store metadata for every benchmark run.

Example:

```yaml
device:
  name: Jetson Orin Nano Super
  os:
  kernel:
  jetpack:
  cuda:
  cudnn:
  tensorrt:
  python:

benchmark:
  input_width: 640
  input_height: 640
  batch_size: 1
  warmup_iterations: 50
  benchmark_iterations: 500
```

Raspberry Pi example:

```yaml
device:
  name: Raspberry Pi 4
  ram:
  os:
  kernel:
  architecture:
  python:
  onnxruntime:
  ncnn:

benchmark:
  input_width: 640
  input_height: 640
  batch_size: 1
  warmup_iterations: 20
  benchmark_iterations: 200
```

---

# Dataset Summary

```text
Dataset: MS COCO 2017

Accuracy evaluation:
  split: val2017
  images: 5,000

Performance evaluation:
  split: deterministic subset of val2017
  images: 500
  same image IDs for all devices/runtimes/models

Input:
  640 × 640

Batch:
  1

Primary detection metric:
  mAP@[0.50:0.95]

Additional metrics:
  AP50
  AP75
  AP-small
  AP-medium
  AP-large
```

---

# Result Schema

Store individual benchmark runs in machine-readable format.

Example:

```json
{
  "model": "rtmdet_tiny",
  "device": "jetson_orin_nano_super",
  "runtime": "tensorrt",
  "precision": "fp16",
  "input_size": [640, 640],
  "batch_size": 1,
  "latency_model_mean_ms": null,
  "latency_model_p50_ms": null,
  "latency_model_p95_ms": null,
  "latency_model_p99_ms": null,
  "fps_model_derived": null,
  "latency_e2e_mean_ms": null,
  "latency_e2e_p50_ms": null,
  "latency_e2e_p95_ms": null,
  "fps_e2e_measured": null,
  "map50": null,
  "map50_95": null,
  "ram_peak_mb": null,
  "vram_peak_mb": null,
  "power_w": null
}
```

---

# Suggested Project Structure

```text
edge-object-detection-benchmark/
│
├── README.md
├── requirements/
│   ├── common.txt
│   ├── jetson.txt
│   └── raspberry_pi.txt
│
├── configs/
│   ├── benchmark.yaml
│   ├── jetson_orin_nano.yaml
│   └── raspberry_pi4.yaml
│
├── models/
│   ├── yolox/
│   ├── yolo26/
│   ├── rtmdet/
│   ├── damo_yolo/
│   ├── picodet/
│   ├── rtdetrv2/
│   └── rfdetr/
│
├── exporters/
│   ├── export_onnx.py
│   ├── export_tensorrt.py
│   └── export_ncnn.py
│
├── runtimes/
│   ├── pytorch_runner.py
│   ├── onnx_runner.py
│   ├── tensorrt_runner.py
│   └── ncnn_runner.py
│
├── benchmark/
│   ├── benchmark_model.py
│   ├── latency.py
│   ├── accuracy.py
│   ├── memory.py
│   └── power.py
│
├── preprocessing/
│   ├── preprocess.py
│   └── postprocess.py
│
├── datasets/
│   ├── README.md
│   ├── coco/
│   │   ├── annotations/
│   │   └── val2017/
│   └── splits/
│       ├── coco_val2017_full.txt
│       └── coco_benchmark_500.txt
│
├── dataset_adapters/
│   ├── base.py
│   ├── coco.py
│   └── custom.py
│
├── scripts/
│   ├── run_jetson_benchmark.sh
│   ├── run_rpi_benchmark.sh
│   └── run_all.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── notebooks/
│   └── analyze_results.ipynb
│
└── tests/
    ├── test_preprocessing.py
    ├── test_postprocessing.py
    └── test_runtime_outputs.py
```

---


# Software Architecture

The project should be modular and configuration-driven so new models, runtimes, metrics, and datasets can be added without rewriting the benchmark core.

The main design principle is to keep these concepts separate:

```text
dataset
model
runtime
benchmark orchestration
metrics
evaluation
result storage/reporting
```

Avoid combining model-specific code and runtime-specific code into classes such as:

```text
YOLOXTensorRTRunner
YOLOXONNXRunner
RTMDetTensorRTRunner
RTMDetONNXRunner
```

Instead, use independent model adapters and runtime backends.

## High-Level Architecture

```text
                    BenchmarkRunner
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
     Dataset          Detector         Runtime
     Adapter          Adapter          Backend
        │                │                │
        ↓                ↓                ↓
      COCO         YOLOX / RTMDet    PyTorch / ONNX
                  YOLO26 / etc.      TensorRT / NCNN

                         │
                         ↓
                    Evaluators
                 /       |        \
             latency   accuracy   resources
                         │
                         ↓
                    Result Store
                         │
                         ↓
                   Report/Analysis
```

The `BenchmarkRunner` should remain stable while models, runtimes, and metric collectors are plugged into it.


## Device Profiles

Devices should be represented as **configuration + capabilities**, not as hard-coded branches inside the benchmark runner.

Avoid logic such as:

```python
if device == "raspberry_pi":
    ...
elif device == "jetson":
    ...
```

Instead, define a `DeviceProfile` abstraction:

```python
from dataclasses import dataclass

@dataclass
class DeviceProfile:
    name: str
    architecture: str
    has_cuda: bool
    has_gpu: bool
    supported_runtimes: list[str]
    supported_precisions: list[str]
    default_threads: int | None = None
    power_monitor: str | None = None
```

The device profile describes hardware capabilities while runtime implementations remain device-independent.

For example:

```text
Device
  └── Jetson Orin Nano Super

Runtime
  └── TensorRT
```

Do not create tightly coupled classes such as `JetsonTensorRTRunner` or `RaspberryPiNCNNRunner`. Keep `TensorRTRuntime`, `ONNXRuntimeBackend`, `NCNNRuntime`, and other backends independent.

### Device configuration files

Suggested layout:

```text
configs/
└── devices/
    ├── jetson_orin_nano_super.yaml
    ├── raspberry_pi_4.yaml
    ├── rk3588.yaml
    └── intel_nuc.yaml
```

Jetson example:

```yaml
name: jetson_orin_nano_super
architecture: aarch64

capabilities:
  cuda: true
  gpu: true

supported_runtimes:
  - pytorch
  - onnxruntime
  - tensorrt

supported_precisions:
  - fp32
  - fp16
  - int8

benchmark:
  warmup: 50
  iterations: 500

metrics:
  temperature:
    provider: linux_sysfs
  power:
    provider: tegrastats
```

RTX 3060 laptop example:

```yaml
name: rtx_3060_laptop
architecture: x86_64

capabilities:
  cuda: true
  gpu: true

supported_runtimes:
  - pytorch
  - onnxruntime
  - tensorrt

supported_precisions:
  - fp32
  - fp16
  - int8

benchmark:
  warmup: 50
  iterations: 500

metrics:
  temperature:
    provider: nvidia
  power:
    provider: nvidia_smi
  utilization:
    provider: nvidia
```

Important metadata:

```text
laptop_model
gpu_name
gpu_vram_gb
gpu_tgp_w
driver_version
cuda_version
cudnn_version
tensorrt_version
power_profile
thermal_mode
```

Raspberry Pi 4 example:

```yaml
name: raspberry_pi_4
architecture: aarch64

capabilities:
  cuda: false
  gpu: false

supported_runtimes:
  - pytorch
  - onnxruntime
  - ncnn

supported_precisions:
  - fp32
  - int8

benchmark:
  warmup: 20
  iterations: 500

metrics:
  temperature:
    provider: linux_sysfs
```

### Capability-based runtime selection

The benchmark should validate combinations before execution:

```python
device = DeviceRegistry.load(config.device)

if not device.supports_runtime(runtime.name):
    return BenchmarkResult.unsupported(
        reason=f"{runtime.name} is not supported on {device.name}"
    )
```

Example capability matrix:

| Device | PyTorch | ONNX Runtime | TensorRT | NCNN |
|---|---:|---:|---:|---:|
| Jetson Orin Nano Super | ✓ | ✓ | ✓ | optional |
| NVIDIA RTX 3060 Laptop GPU | ✓ | ✓ | ✓ | optional |
| Raspberry Pi 4 | ✓ | ✓ | — | ✓ |
| RK3588 SBC | ✓ | ✓ | — | ✓ |
| NVIDIA x86 system | ✓ | ✓ | ✓ | optional |

Unsupported combinations should be recorded explicitly as `N/A — unsupported by device/runtime` rather than silently omitted.

### Device registry

Suggested structure:

```text
src/edgebench/devices/
├── __init__.py
├── base.py
├── registry.py
└── probes.py
```

Responsibilities:

```text
base.py      DeviceProfile definition
registry.py  load profiles and validate capabilities
probes.py    optional automatic hardware/software discovery
```

The registry must not contain inference code.

### Device-specific metric providers

Power, temperature, clocks, utilization, and thermal throttling are often device-specific. Keep those implementations outside `BenchmarkRunner`.

Suggested layout:

```text
metrics/
├── power/
│   ├── base.py
│   ├── jetson.py
│   ├── raspberry_pi.py
│   └── external_meter.py
├── temperature/
│   ├── base.py
│   ├── linux_sysfs.py
│   └── jetson.py
└── utilization/
    ├── base.py
    ├── cpu.py
    └── nvidia.py
```

The device profile selects the relevant provider.

### Device metadata in every result

Every benchmark run should include reproducibility metadata.

Common example:

```json
{
  "device": {
    "name": "raspberry_pi_4",
    "architecture": "aarch64",
    "cpu": "Cortex-A72",
    "ram_gb": 8,
    "os": "Raspberry Pi OS",
    "kernel": "..."
  }
}
```

Jetson-specific fields should additionally include:

```json
{
  "device": {
    "name": "jetson_orin_nano_super",
    "jetpack": "...",
    "cuda": "...",
    "cudnn": "...",
    "tensorrt": "...",
    "power_mode": "...",
    "clock_mode": "...",
    "cooling": "..."
  }
}
```

### Optional automatic device probing

A future `DeviceProbe` may discover:

```text
CPU architecture
CPU model
RAM
CUDA availability/version
TensorRT version
ONNX Runtime providers
CPU thread count
JetPack version
OS/kernel
```

Manual configuration should take precedence so benchmark conditions remain reproducible.

### Device extensibility rule

Adding a new device should normally require only:

```text
configs/devices/new_device.yaml
```

and, if necessary, a new hardware metric provider such as:

```text
metrics/power/new_device.py
```

It should not require changes to `BenchmarkRunner`, detector adapters, COCO evaluation, result aggregation, or reporting.

The intended composition is:

```text
ExperimentConfig
      │
      ├── DeviceProfile
      ├── DatasetAdapter
      ├── DetectorAdapter
      ├── RuntimeBackend
      └── MetricCollectors
               │
               ↓
         BenchmarkRunner
               │
               ↓
       Standardized Result
```

## 1. Dataset Adapters

Dataset adapters expose standardized samples and annotations. The dataset layer must not contain model-specific or runtime-specific logic.

```python
class DatasetAdapter(ABC):
    def __len__(self) -> int: ...
    def get_sample(self, index: int): ...
```

Initial implementations:

```text
CocoDataset
CustomIndustrialDataset
```

Future extensions may include `DirectoryDataset`, `VideoDataset`, and `CameraDataset`.

## 2. Model Adapters

Model adapters contain model-specific behavior:

- preprocessing and normalization;
- RGB/BGR conversion;
- letterboxing;
- output decoding;
- confidence processing;
- NMS if required;
- conversion to original-image coordinates;
- COCO class mapping;
- model export helpers.

```python
class DetectorAdapter(ABC):
    @property
    def name(self) -> str: ...
    def preprocess(self, image): ...
    def postprocess(self, raw_output, metadata): ...
    def load_pytorch(self): ...
    def export_onnx(self, output_path): ...
```

Implementations:

```text
models/
├── base.py
├── yolox.py
├── yolo26.py
├── rtmdet.py
├── damo_yolo.py
├── picodet.py
├── rtdetr.py
└── rfdetr.py
```

## 3. Runtime Backends

Runtime backends should answer only one question: **given model input, execute the network**.

```python
class RuntimeBackend(ABC):
    def load(self): ...
    def warmup(self, input_data): ...
    def infer(self, input_data): ...
    def synchronize(self): ...
```

Implementations:

```text
runtimes/
├── base.py
├── pytorch.py
├── onnxruntime.py
├── tensorrt.py
└── ncnn.py
```

The runtime layer must not contain model-specific postprocessing.

## 4. Benchmark Orchestrator

The benchmark orchestrator controls:

- model/runtime loading;
- dataset iteration;
- warm-up;
- measured iterations;
- preprocessing;
- inference;
- postprocessing;
- model-only latency;
- end-to-end latency;
- metric collection;
- prediction collection;
- evaluation;
- result writing.

The benchmark runner should not contain detector-specific decoding logic.

## 5. Metric Collectors

Execution/resource measurements should be independent collectors.

```python
class MetricCollector(ABC):
    def on_run_start(self): ...
    def before_inference(self): ...
    def after_inference(self): ...
    def on_run_end(self): ...
    def result(self) -> dict: ...
```

Initial collectors:

```text
latency
memory
utilization
temperature
power
```

## 6. Evaluation

Keep dataset-level evaluation separate from runtime measurements.

Execution metrics:

```text
latency
FPS
RAM
VRAM
CPU/GPU utilization
temperature
power
```

Dataset evaluation:

```text
mAP@0.50:0.95
AP50
AP75
AP-small
AP-medium
AP-large
precision
recall
```

## 7. Standard Detection Format

All models must convert outputs into one common format.

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
```

Convention:

```text
bbox: xyxy
coordinates: original image coordinates
class_id: canonical COCO class ID
```

## 8. Result Storage and Reporting

Store raw results first, then generate summaries separately.

```text
results/
├── raw/
│   ├── jetson/
│   ├── rtx_3060_laptop/
│   └── raspberry_pi/
├── processed/
├── summaries/
└── figures/
```

Reporting modules:

```text
reporting/
├── aggregate.py
├── tables.py
└── plots.py
```

# FPS and Latency

FPS can be estimated from mean model latency:

```text
FPS ≈ 1000 / latency_ms
```

For this benchmark, keep both **derived model FPS** and **directly measured end-to-end FPS**.

## Model FPS

```text
fps_model_derived = 1000 / latency_model_mean_ms
```

Use mean model-only latency. Do not derive FPS from P50, P95, or P99 latency.

## End-to-End FPS

Measure throughput directly over the complete pipeline:

```text
fps_e2e_measured = number_of_measured_images / total_elapsed_seconds
```

Store these timing fields for every run:

```text
latency_model_mean_ms
latency_model_p50_ms
latency_model_p95_ms
latency_model_p99_ms
fps_model_derived
latency_e2e_mean_ms
latency_e2e_p50_ms
latency_e2e_p95_ms
fps_e2e_measured
```

# Configuration-Driven Experiments

Experiment definitions should be configuration files instead of separate scripts for every model/runtime pair.

```yaml
experiment:
  name: rtmdet_jetson_trt

device: jetson_orin_nano_super

dataset:
  name: coco
  split: benchmark_500

model:
  name: rtmdet_tiny
  input_size: [640, 640]

runtime:
  name: tensorrt
  precision: fp16

benchmark:
  batch_size: 1
  warmup: 50
  iterations: 500

metrics:
  - latency
  - memory
  - power
  - utilization
```

Recommended CLI:

```bash
python -m edgebench run configs/rtmdet_jetson_trt.yaml
```

# Architectural Extension Rule

Adding a new device should ideally require only:

```text
configs/devices/new_device.yaml
```

and only when required, a device-specific metric provider.

Adding a new detector should ideally require only:

```text
models/new_detector.py
configs/models/new_detector.yaml
```

Adding a new runtime should ideally require only:

```text
runtimes/new_runtime.py
```

Adding a new metric should ideally require only:

```text
metrics/new_metric.py
```

The `BenchmarkRunner` should not need to be rewritten for these additions.


---

# Recommended Implementation Strategy

## Phase 0 — Dataset setup

Implement the COCO benchmark setup first:

1. download COCO `val2017`;
2. download `instances_val2017.json`;
3. generate `coco_val2017_full.txt`;
4. generate a deterministic `coco_benchmark_500.txt` using a fixed seed;
5. verify that the same 500 image IDs are used on both devices;
6. implement the `CocoDataset` adapter;
7. validate COCO-format prediction export and evaluation.

Do not copy the entire COCO dataset into version control.

---

## Phase 1 — Common benchmark framework

Implement:

```text
common preprocessing
common result schema
common timing interface
common accuracy evaluation
```

Create one abstract runner interface such as:

```python
class DetectorRunner:
    def load(self):
        ...

    def preprocess(self, image):
        ...

    def infer(self, input_tensor):
        ...

    def postprocess(self, outputs):
        ...
```

Runtime implementations can then share the benchmark logic.

---

## Phase 2 — YOLOX baseline

Start with the existing YOLOX model.

Implement:

1. PyTorch runner;
2. ONNX export;
3. ONNX Runtime runner;
4. TensorRT export on Jetson;
5. NCNN export on Raspberry Pi.

Validate that all runtimes produce approximately equivalent detections before benchmarking speed.

---

## Phase 3 — Add modern detectors

Add models one at a time:

```text
YOLOX-Tiny
    ↓
RTMDet-Tiny
    ↓
DAMO-YOLO-T
    ↓
YOLO26n
    ↓
PP-PicoDet-S
    ↓
RT-DETRv2-S
    ↓
RF-DETR-Nano
```

Do not implement every model/runtime combination simultaneously.

Verify output correctness after each addition.

---

## Phase 4 — Automated benchmarking

Create one command interface such as:

```bash
python scripts/run_all.py \
    --device jetson \
    --input-size 640 \
    --batch-size 1
```

or:

```bash
python benchmark/benchmark_model.py \
    --model rtmdet_tiny \
    --runtime onnx \
    --precision fp16 \
    --input-size 640 \
    --iterations 500
```

---

## Phase 5 — Analysis

Generate plots such as:

### Accuracy vs FPS

```text
mAP
 ↑
 │        ● RF-DETR
 │
 │     ● RT-DETR
 │
 │             ● RTMDet
 │                  ● YOLO26
 │             ● YOLOX
 └────────────────────────→ FPS
```

### Latency by runtime

Compare:

```text
PyTorch
ONNX Runtime
TensorRT
NCNN
```

### FPS per watt

Especially useful for Jetson.

### Accuracy loss from INT8

Especially useful for Raspberry Pi / NCNN.

---

# Primary Result Tables

## Jetson Orin Nano Super

**Input: 640 × 640, batch=1**

| Model | Runtime | Precision | FPS | Mean latency | P50 | P95 | mAP50-95 | RAM | VRAM | Power |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOX-Tiny | PyTorch | FP16 | | | | | | | | |
| YOLOX-Tiny | ONNX Runtime | FP16 | | | | | | | | |
| YOLOX-Tiny | TensorRT | FP16 | | | | | | | | |
| YOLO26n | PyTorch | FP16 | | | | | | | | |
| YOLO26n | ONNX Runtime | FP16 | | | | | | | | |
| YOLO26n | TensorRT | FP16 | | | | | | | | |
| RTMDet-Tiny | PyTorch | FP16 | | | | | | | | |
| RTMDet-Tiny | ONNX Runtime | FP16 | | | | | | | | |
| RTMDet-Tiny | TensorRT | FP16 | | | | | | | | |
| DAMO-YOLO-T | PyTorch | FP16 | | | | | | | | |
| DAMO-YOLO-T | ONNX Runtime | FP16 | | | | | | | | |
| DAMO-YOLO-T | TensorRT | FP16 | | | | | | | | |
| PP-PicoDet-S | PyTorch | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| PP-PicoDet-S | ONNX Runtime | FP16 | | | | | | | | |
| PP-PicoDet-S | TensorRT | FP16 | | | | | | | | |
| RT-DETRv2-S | PyTorch | FP16 | | | | | | | | |
| RT-DETRv2-S | ONNX Runtime | FP16 | | | | | | | | |
| RT-DETRv2-S | TensorRT | FP16 | | | | | | | | |
| RF-DETR-Nano | PyTorch | FP16 | | | | | | | | |
| RF-DETR-Nano | ONNX Runtime | FP16 | | | | | | | | |
| RF-DETR-Nano | TensorRT | FP16 | | | | | | | | |

---

## NVIDIA RTX 3060 Laptop GPU

**Input: 640 × 640, batch=1**

| Model | Runtime | Precision | FPS | Mean latency | P50 | P95 | mAP50-95 | RAM | VRAM | Power |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOX-Tiny | PyTorch | FP16 | | | | | | | | |
| YOLOX-Tiny | ONNX Runtime | FP16 | | | | | | | | |
| YOLOX-Tiny | TensorRT | FP16 | | | | | | | | |
| YOLO26n | PyTorch | FP16 | | | | | | | | |
| YOLO26n | ONNX Runtime | FP16 | | | | | | | | |
| YOLO26n | TensorRT | FP16 | | | | | | | | |
| RTMDet-Tiny | PyTorch | FP16 | | | | | | | | |
| RTMDet-Tiny | ONNX Runtime | FP16 | | | | | | | | |
| RTMDet-Tiny | TensorRT | FP16 | | | | | | | | |
| DAMO-YOLO-T | PyTorch | FP16 | | | | | | | | |
| DAMO-YOLO-T | ONNX Runtime | FP16 | | | | | | | | |
| DAMO-YOLO-T | TensorRT | FP16 | | | | | | | | |
| PP-PicoDet-S | ONNX Runtime | FP16 | | | | | | | | |
| PP-PicoDet-S | TensorRT | FP16 | | | | | | | | |
| RT-DETRv2-S | PyTorch | FP16 | | | | | | | | |
| RT-DETRv2-S | ONNX Runtime | FP16 | | | | | | | | |
| RT-DETRv2-S | TensorRT | FP16 | | | | | | | | |
| RF-DETR-Nano | PyTorch | FP16 | | | | | | | | |
| RF-DETR-Nano | ONNX Runtime | FP16 | | | | | | | | |
| RF-DETR-Nano | TensorRT | FP16 | | | | | | | | |

---

## Raspberry Pi 4

**Input: 640 × 640, batch=1**

| Model | Runtime | Precision | FPS | Mean latency | P50 | P95 | mAP50-95 | RAM | CPU | Power |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOX-Tiny | PyTorch | FP32 | | | | | | | | |
| YOLOX-Tiny | ONNX Runtime | FP32 | | | | | | | | |
| YOLOX-Tiny | NCNN | FP32 | | | | | | | | |
| YOLOX-Tiny | NCNN | INT8 | | | | | | | | |
| YOLO26n | PyTorch | FP32 | | | | | | | | |
| YOLO26n | ONNX Runtime | FP32 | | | | | | | | |
| YOLO26n | NCNN | FP32 | | | | | | | | |
| YOLO26n | NCNN | INT8 | | | | | | | | |
| RTMDet-Tiny | PyTorch | FP32 | | | | | | | | |
| RTMDet-Tiny | ONNX Runtime | FP32 | | | | | | | | |
| RTMDet-Tiny | NCNN | FP32 | | | | | | | | |
| RTMDet-Tiny | NCNN | INT8 | | | | | | | | |
| DAMO-YOLO-T | PyTorch | FP32 | | | | | | | | |
| DAMO-YOLO-T | ONNX Runtime | FP32 | | | | | | | | |
| DAMO-YOLO-T | NCNN | FP32 | | | | | | | | |
| DAMO-YOLO-T | NCNN | INT8 | | | | | | | | |
| PP-PicoDet-S | PyTorch | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| PP-PicoDet-S | ONNX Runtime | FP32 | | | | | | | | |
| PP-PicoDet-S | NCNN | FP32 | | | | | | | | |
| PP-PicoDet-S | NCNN | INT8 | | | | | | | | |
| RT-DETRv2-S | PyTorch | FP32 | | | | | | | | |
| RT-DETRv2-S | ONNX Runtime | FP32 | | | | | | | | |
| RT-DETRv2-S | NCNN | FP32 | | | | | | | | |
| RF-DETR-Nano | PyTorch | FP32 | | | | | | | | |
| RF-DETR-Nano | ONNX Runtime | FP32 | | | | | | | | |

Unsupported model/runtime combinations should be explicitly marked as:

```text
N/A — unsupported
```

rather than omitted.

---

# Main Research Questions

The benchmark should answer:

1. How much faster are modern lightweight CNN detectors than YOLOX on current edge hardware?
2. Does RTMDet provide a better accuracy/speed tradeoff than YOLOX?
3. Does DAMO-YOLO-T improve efficiency through NAS/reparameterized CNN design?
4. Is PP-PicoDet-S significantly better suited to CPU-only edge inference than the larger CNN detectors?
5. How competitive is YOLO26n against permissively licensed CNN alternatives?
6. How large is the gap between lightweight CNNs and RT-DETRv2/RF-DETR on edge hardware?
7. Can RT-DETRv2-S maintain real-time performance on Jetson?
8. Is RF-DETR-Nano practical for low-power edge deployment?
9. How much performance is gained by moving from PyTorch to ONNX Runtime?
10. How much additional performance does TensorRT provide on Jetson?
11. How much does NCNN improve CPU inference on Raspberry Pi?
12. How much accuracy is lost with INT8 quantization?
13. Which CNN model provides the best accuracy/FPS/power tradeoff?
14. Which architecture degrades most gracefully when moving from Jetson GPU inference to Raspberry Pi CPU inference?
15. How do the same models scale from Raspberry Pi 4 to Jetson Orin Nano Super to RTX 3060 laptop GPU?

---

# Initial Success Criteria

A useful final comparison should make it possible to identify:

```text
Best raw accuracy
Best raw FPS
Best latency
Best FPS/W
Best accuracy/FPS
Best CPU detector
Best Jetson detector
Best RTX 3060 detector
Easiest model to deploy
Best permissively licensed alternative to YOLO
```

The benchmark should prioritize reproducibility over achieving the highest possible headline FPS.
