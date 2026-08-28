# Raspberry Pi 4 benchmark setup

This is the reproducible phase-1 setup for ONNX Runtime FP32 and NCNN FP32.
PyTorch and NCNN INT8 are deliberately excluded until these paths pass their
accuracy gates.

## 1. Flash and secure the operating system

Use Raspberry Pi Imager to install **Raspberry Pi OS Legacy Lite (64-bit,
Bookworm)**. In Imager's customisation screen set:

- hostname: `modeltest` (the network address becomes `modeltest.local`);
- username: `user`;
- the temporary password supplied separately by the operator;
- SSH with password authentication enabled;
- timezone: `Asia/Tokyo`.

Prefer wired Ethernet. From the workstation, verify the new SSH fingerprint
against the fingerprint shown locally on the Pi before accepting it:

```bash
ssh user@modeltest.local
```

Install the workstation's public key interactively. Never put the password in
a command, environment variable, script, or repository file.

```bash
ssh-copy-id user@modeltest.local
ssh user@modeltest.local
```

Keep the first session open, verify key-only access in a second session, then
run `passwd` to replace the temporary password. Password-based SSH may be
disabled only after key login has been verified.

## 2. Install the host dependencies

Run on the Pi:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install git python3-venv python3-dev build-essential \
  libopenblas-dev libgl1 libglib2.0-0 avahi-daemon cpufrequtils
sudo reboot
```

Transfer or clone the repository into
`/home/user/detection-model-comparison`, then run from that directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/raspberry_pi.txt
.venv/bin/python -m edgebench doctor --device raspberry_pi_4 \
  --backend onnxruntime:fp32 --backend ncnn:fp32
```

The doctor command is expected to report missing artifacts until the next
step is complete.

## 3. Transfer data and build NCNN artifacts

Copy these platform-independent inputs from the workstation while preserving
their repository-relative paths:

- `datasets/coco/val2017/`;
- `datasets/coco/annotations/instances_val2017.json`;
- `datasets/splits/`;
- every `weights/*/*_onnxruntime_fp32.onnx` file.

Do not transfer TensorRT engines or use FP16 ONNX files on the Pi. Record and
compare SHA-256 hashes on both machines after transfer.

Install the pinned converter on the build machine:

```bash
python -m pip install pnnx==20260526
python -m edgebench export yolox_tiny --to ncnn --precision fp32
```

Repeat the export for each model whose ONNX graph passes conversion, then copy
the generated `*_ncnn_fp32.param` and `*_ncnn_fp32.bin` pairs to the matching
weights directories on the Pi. Use the same converter version for every
model. A failed conversion is recorded as unsupported; it must not be silently
omitted or mislabeled as an INT8 artifact.

Gate every converted artifact before copying it:

```bash
python -m edgebench validate-export yolox_tiny --runtime ncnn --samples 20
```

This creates a checksum-bound validation marker only when the converted mAP is
within 0.005 of ONNX Runtime FP32. Copy the marker with the NCNN pair. The Pi
preflight rejects missing, failed, stale, or modified validation markers.

## 4. Prepare a measured run

Use stock clocks and no `force_turbo` or overclock settings. With the passive
heatsink exposed and the Pi mounted vertically, run:

```bash
for cpu in 0 1 2 3; do sudo cpufreq-set -c "$cpu" -g performance; done
sudo swapoff -a
vcgencmd measure_temp
vcgencmd get_throttled
```

Allow the Pi to cool to 55 C or below. `get_throttled` must return `0x0`.
The benchmark records maximum temperature and post-run flags; a run reaching
80 C or reporting any throttling/undervoltage flag is stored as invalid.

Pilot ONNX Runtime before starting the matrix. This path remains runnable even
while NCNN conversion work is still accuracy-gated:

```bash
.venv/bin/python -m edgebench doctor --device raspberry_pi_4 \
  --backend onnxruntime:fp32 --model yolox_tiny
.venv/bin/python scripts/run_all.py --device raspberry_pi_4 \
  --backend onnxruntime:fp32 --model yolox_tiny --warmup 2 --iterations 10
```

After every NCNN artifact has a passing validation marker, run the combined
phase-1 500-image matrix:

```bash
./scripts/run_rpi_benchmark.sh
```

Restore swap after testing if desired:

```bash
sudo swapon -a
```

For final COCO accuracy, use the full split in a separate run with 5,000
iterations. Do not use its timings in the 500-image performance table.
