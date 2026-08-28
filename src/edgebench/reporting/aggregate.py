"""Aggregate raw result JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgebench.paths import REPO_ROOT


def aggregate_results(raw_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load every raw BenchmarkResult JSON into row dicts.

    Rows keep their JSON fields; ``status`` records successful, unsupported,
    or environmentally invalid runs for reporting.
    """
    root = Path(raw_dir) if raw_dir is not None else REPO_ROOT / "results" / "raw"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["_source"] = str(path)
        rows.append(row)
    return rows
