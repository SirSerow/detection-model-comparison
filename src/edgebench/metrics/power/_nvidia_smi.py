"""Shared nvidia-smi query helper."""

from __future__ import annotations

import shutil
import subprocess


def query_nvidia_smi(fields: list[str]) -> dict[str, float]:
    """Query numeric GPU fields; returns only fields that parsed as floats."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    output = subprocess.run(
        [
            binary,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = output.splitlines()[0].split(",")
    parsed: dict[str, float] = {}
    for field, raw in zip(fields, values):
        try:
            parsed[field] = float(raw.strip())
        except ValueError:
            continue  # e.g. "[Not Supported]" on some GPUs
    return parsed
