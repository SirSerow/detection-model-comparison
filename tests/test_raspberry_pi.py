"""Raspberry Pi telemetry, thermal validity, and NCNN export safeguards."""

from __future__ import annotations

from edgebench.benchmark.runner import _environment_invalid_reason
from edgebench.metrics.temperature.raspberry_pi import (
    RaspberryPiTemperatureCollector,
    parse_temperature,
    parse_throttled,
)


def test_vcgencmd_parsers() -> None:
    assert parse_temperature("temp=47.8'C") == 47.8
    assert parse_temperature("not a temperature") is None
    assert parse_throttled("throttled=0x0") == 0
    assert parse_throttled("throttled=0x50005") == 0x50005
    assert parse_throttled(None) is None


def test_pi_collector_aggregates_maximum_and_flags() -> None:
    collector = RaspberryPiTemperatureCollector()
    result = collector.aggregate(
        [
            {"temperature_c": 50.0, "throttled_flags": 0.0},
            {"temperature_c": 62.0, "throttled_flags": 4.0},
        ]
    )
    assert result["temperature_c"] == 56.0
    assert result["temperature_c_max"] == 62.0
    assert result["throttled_flags"] == 4


def test_pi_run_is_invalidated_by_throttling_or_heat() -> None:
    assert (
        _environment_invalid_reason(
            "raspberry_pi_4",
            {
                "post_run": {"throttled_flags": "0x50005"},
                "collector_metrics": {"temperature_c_max": 70.0},
            },
        )
        is not None
    )
    assert (
        _environment_invalid_reason(
            "raspberry_pi_4",
            {
                "post_run": {"throttled_flags": "0x0"},
                "collector_metrics": {"temperature_c_max": 80.0},
            },
        )
        is not None
    )
    assert (
        _environment_invalid_reason(
            "raspberry_pi_4",
            {
                "post_run": {"throttled_flags": "0x0"},
                "collector_metrics": {"temperature_c_max": 79.9},
            },
        )
        is None
    )
