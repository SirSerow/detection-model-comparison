"""Background-sampling collector base.

Hardware metrics (power, temperature, utilization) are polled on a daemon
thread owned by the collector: started in ``on_run_start``, stopped and
joined in ``on_run_end``. The ``before_inference``/``after_inference``
hooks are no-ops — sampling is periodic, not per-inference. All provider
failures are collected into ``errors`` in the result instead of crashing
the measured run.
"""

from __future__ import annotations

import threading
from typing import Any

from edgebench.metrics.base import MetricCollector


class SamplingCollector(MetricCollector):
    """MetricCollector that polls ``sample()`` on a background thread."""

    name = "metric"
    interval_s = 0.5

    def __init__(self, interval_s: float | None = None) -> None:
        if interval_s is not None:
            self.interval_s = interval_s
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def on_run_start(self) -> None:
        self._samples = []
        self._errors = []
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"edgebench-{self.name}", daemon=True
        )
        self._thread.start()

    def before_inference(self) -> None:
        return None

    def after_inference(self) -> None:
        return None

    def on_run_end(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def sample(self) -> dict[str, float] | None:
        """Return one measurement (e.g. ``{"power_w": 12.3}``)."""
        raise NotImplementedError

    def aggregate(
        self, samples: list[dict[str, float]]
    ) -> dict[str, float | None]:
        """Reduce samples to result fields; mean per key by default."""
        keys = {key for sample in samples for key in sample}
        return {
            key: (
                sum(sample[key] for sample in samples if key in sample)
                / max(sum(1 for sample in samples if key in sample), 1)
            )
            for key in sorted(keys)
        }

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = dict(self.aggregate(self._samples))
        if self._errors:
            result[f"{self.name}_errors"] = len(self._errors)
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self.sample()
            except Exception as exc:  # keep the measured run alive
                self._errors.append(f"{type(exc).__name__}: {exc}")
            else:
                if sample:
                    self._samples.append(sample)
            self._stop.wait(self.interval_s)
