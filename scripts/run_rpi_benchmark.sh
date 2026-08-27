#!/usr/bin/env bash
# Raspberry Pi 4 benchmark driver. Run on-device inside the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/run_all.py --device raspberry_pi_4 "$@"
python -m edgebench report
