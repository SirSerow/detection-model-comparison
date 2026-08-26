"""Aggregate raw result JSON files."""

from __future__ import annotations

from typing import Any


def aggregate_results(raw_dir: str) -> list[dict[str, Any]]:
    raise NotImplementedError("Result aggregation is not implemented yet")
