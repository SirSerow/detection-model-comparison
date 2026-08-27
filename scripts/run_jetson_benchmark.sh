#!/usr/bin/env bash
# Jetson Orin Nano Super benchmark driver. Run on-device inside the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/run_all.py --device jetson_orin_nano_super "$@"
python -m edgebench report
