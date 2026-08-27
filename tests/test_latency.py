"""Latency summarization tests."""

from __future__ import annotations

import pytest

from edgebench.benchmark.latency import summarize_latencies


def test_summarize_basic_statistics() -> None:
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert stats["std_ms"] == pytest.approx(1.118033988749895)
    assert stats["p95_ms"] <= 4.0
    assert stats["p99_ms"] <= 4.0


def test_summarize_single_sample() -> None:
    stats = summarize_latencies([7.5])
    assert stats["mean_ms"] == 7.5
    assert stats["p50_ms"] == 7.5
    assert stats["p99_ms"] == 7.5
    assert stats["std_ms"] == 0.0


def test_summarize_monotonic_percentiles() -> None:
    samples = [float(i) for i in range(1, 501)]
    stats = summarize_latencies(samples)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert stats["mean_ms"] == pytest.approx(250.5)


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize_latencies([])
