"""Persist ``BenchmarkResult`` records as raw JSON.

Layout: ``<root>/<device>/<model>_<runtime>_<precision>.json``. Writes are
atomic (temporary file + rename) so interrupted runs never leave truncated
JSON behind.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from edgebench.types import BenchmarkResult


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, result: BenchmarkResult) -> Path:
        device_dir = self.root / result.device
        device_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.model}_{result.runtime}_{result.precision}.json"
        target = device_dir / filename
        payload = self._to_json_dict(result)
        temp_path = target.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, target)
        return target

    @staticmethod
    def _to_json_dict(result: BenchmarkResult) -> dict[str, Any]:
        payload = dataclasses.asdict(result)
        payload["status"] = result.status.value
        payload["input_size"] = list(result.input_size)
        return payload
