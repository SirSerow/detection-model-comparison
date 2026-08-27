"""RAM/VRAM peak collector.

RAM: peak RSS of this process, sampled around every inference and at run
end. VRAM: ``torch.cuda.max_memory_allocated`` when torch with CUDA is in
use; ``None`` otherwise (never fabricated).
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

    def on_run_start(self) -> None:
        self._process = psutil.Process()
        self._peak_rss_bytes = self._rss_bytes()

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

    @staticmethod
    def _vram_peak_mb() -> float | None:
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
