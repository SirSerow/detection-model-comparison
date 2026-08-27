"""Latency helpers. Model-only vs end-to-end stay separate.

FPS derivation follows the research spec exactly:
``fps_model_derived = 1000 / latency_model_mean_ms`` (never from percentiles),
while end-to-end FPS is measured directly by the runner over the full loop.
"""

from __future__ import annotations


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    """Return mean/p50/p95/p99/std over latency samples in milliseconds.

    Percentiles use linear interpolation on the sorted samples.
    """
    if not samples_ms:
        raise ValueError("No latency samples to summarize")
    ordered = sorted(float(sample) for sample in samples_ms)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((sample - mean) ** 2 for sample in ordered) / count
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(ordered, 50.0),
        "p95_ms": _percentile(ordered, 95.0),
        "p99_ms": _percentile(ordered, 99.0),
        "std_ms": variance**0.5,
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
