"""Raspberry Pi power provider.

The Pi 4 exposes no power telemetry. This collector reports
``power_w: None`` — never a fabricated number. For real measurements use
the ``external_meter`` provider with a USB power meter.
"""

from __future__ import annotations

from typing import Any

from edgebench.metrics.base import MetricCollector


class RaspberryPiPowerCollector(MetricCollector):
    name = "raspberry_pi_power"

    def on_run_start(self) -> None:
        return None

    def before_inference(self) -> None:
        return None

    def after_inference(self) -> None:
        return None

    def on_run_end(self) -> None:
        return None

    def result(self) -> dict[str, Any]:
        return {
            "power_w": None,
            "raspberry_pi_power_note": "no on-board power telemetry on Pi 4",
        }
