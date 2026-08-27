"""Jetson tegrastats power provider.

Parses power rail readings (e.g. ``POM_5V_IN 4962mW/4962mW``) from the
``tegrastats`` stream and reports mean total draw in watts. tegrastats
ships with JetPack on the device; this collector is validated on Jetson
hardware, not in CI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Any

from edgebench.metrics.base import MetricCollector

_RAIL_RE = re.compile(r"\b[A-Z0-9_]*(?:VDD|POM|VIN|SOC|GPU|CPU|CV|VDDRQ|SYS5V)[A-Z0-9_]* (\d+)mW")


class JetsonPowerCollector(MetricCollector):
    name = "jetson_power"

    def __init__(self, interval_ms: int = 500) -> None:
        self.interval_ms = interval_ms
        self._samples_w: list[float] = []
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._error: str | None = None

    def on_run_start(self) -> None:
        tegrastats = shutil.which("tegrastats")
        if tegrastats is None:
            self._error = "tegrastats not found on PATH"
            return
        self._samples_w = []
        self._process = subprocess.Popen(
            [tegrastats, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._reader = threading.Thread(
            target=self._read_stream, name="edgebench-tegrastats", daemon=True
        )
        self._reader.start()

    def before_inference(self) -> None:
        return None

    def after_inference(self) -> None:
        return None

    def on_run_end(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._reader is not None:
            self._reader.join(timeout=5.0)
            self._reader = None

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._samples_w:
            result["power_w"] = sum(self._samples_w) / len(self._samples_w)
        else:
            result["power_w"] = None
        if self._error:
            result["jetson_power_error"] = self._error
        return result

    def _read_stream(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            watts = sum(int(match) for match in _RAIL_RE.findall(line)) / 1000.0
            if watts > 0:
                self._samples_w.append(watts)
