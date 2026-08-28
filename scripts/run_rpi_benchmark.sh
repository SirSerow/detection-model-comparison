#!/usr/bin/env bash
# Raspberry Pi 4 benchmark driver. Run on-device inside the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m edgebench doctor --device raspberry_pi_4 \
  --backend onnxruntime:fp32 --backend ncnn:fp32
python scripts/run_all.py --device raspberry_pi_4 \
  --backend onnxruntime:fp32 --backend ncnn:fp32 "$@"
python -m edgebench report
