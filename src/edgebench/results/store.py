"""Persist ``BenchmarkResult`` records. Writing is not implemented yet."""

from __future__ import annotations

from pathlib import Path

from edgebench.types import BenchmarkResult


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, result: BenchmarkResult) -> Path:
        raise NotImplementedError("ResultStore.write is not implemented yet")
