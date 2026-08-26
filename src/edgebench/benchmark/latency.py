"""Latency helpers. Model-only vs end-to-end stay separate."""

from __future__ import annotations


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    raise NotImplementedError("Latency summarization is not implemented yet")
