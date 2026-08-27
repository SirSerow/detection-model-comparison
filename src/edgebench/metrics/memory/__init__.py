"""Per-process RAM/VRAM peak collector.

RAM uses process RSS. VRAM uses NVML's per-process allocation when available,
which keeps the measurement comparable across PyTorch, ONNX Runtime, and
TensorRT. PyTorch allocator statistics remain a fallback for non-NVML setups.
"""

from __future__ import annotations

from typing import Any

import psutil

from edgebench.metrics.base import MetricCollector


class MemoryCollector(MetricCollector):
    name = "memory"

    def __init__(self) -> None:
        self._process: psutil.Process | None = None
        self._peak_rss_bytes = 0
        self._peak_vram_bytes = 0
        self._nvml: Any = None
        self._nvml_handle: Any = None

    def on_run_start(self) -> None:
        self._process = psutil.Process()
        self._peak_rss_bytes = self._rss_bytes()
        self._peak_vram_bytes = 0
        self._start_nvml()
        self._sample()

    def before_inference(self) -> None:
        self._sample()

    def after_inference(self) -> None:
        self._sample()

    def on_run_end(self) -> None:
        self._sample()

    def result(self) -> dict[str, Any]:
        return {
            "ram_peak_mb": self._peak_rss_bytes / (1024.0 * 1024.0),
            "vram_peak_mb": self._vram_peak_mb(),
        }

    def _rss_bytes(self) -> int:
        if self._process is None:
            return 0
        return int(self._process.memory_info().rss)

    def _sample(self) -> None:
        if self._process is None:
            return
        self._peak_rss_bytes = max(self._peak_rss_bytes, self._rss_bytes())
        self._peak_vram_bytes = max(self._peak_vram_bytes, self._nvml_vram_bytes())

    def _start_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def _nvml_vram_bytes(self) -> int:
        if self._nvml is None or self._nvml_handle is None or self._process is None:
            return 0
        try:
            processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(
                self._nvml_handle
            )
        except Exception:
            return 0
        used = sum(
            int(process.usedGpuMemory)
            for process in processes
            if process.pid == self._process.pid
            and isinstance(process.usedGpuMemory, int)
            and 0 < process.usedGpuMemory < (1 << 60)
        )
        return used

    def _vram_peak_mb(self) -> float | None:
        if self._peak_vram_bytes > 0:
            return self._peak_vram_bytes / (1024.0 * 1024.0)
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.max_memory_allocated()
        if allocated <= 0:
            return None
        return float(allocated) / (1024.0 * 1024.0)
