"""External watt-meter provider.

Readings are entered manually after a run; this collector intentionally
returns ``power_w: None`` so automated results never mix measured and
estimated power. Record meter readings in the experiment notes / summary
layer.
"""

from __future__ import annotations

from typing import Any

from edgebench.metrics.base import MetricCollector


class ExternalMeterPowerCollector(MetricCollector):
    name = "external_meter_power"

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
            "external_meter_power_note": "manual entry required; see experiment notes",
        }
