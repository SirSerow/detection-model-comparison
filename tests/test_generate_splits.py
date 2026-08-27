"""Deterministic split generation tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "generate_coco_splits",
    Path(__file__).resolve().parent.parent / "scripts" / "generate_coco_splits.py",
)
assert SPEC is not None and SPEC.loader is not None
generate_coco_splits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_coco_splits)


def _write_annotations(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 640, "height": 480}
            for image_id in range(1, count + 1)
        ],
        "annotations": [],
        "categories": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_splits_are_deterministic_and_disjoint_from_seed(tmp_path: Path) -> None:
    annotations = tmp_path / "instances_val2017.json"
    _write_annotations(annotations, 1000)

    full_a, subset_a = generate_coco_splits.generate_splits(
        annotations, tmp_path / "a", seed=42, subset_size=100
    )
    full_b, subset_b = generate_coco_splits.generate_splits(
        annotations, tmp_path / "b", seed=42, subset_size=100
    )
    assert subset_a.read_text() == subset_b.read_text()
    subset_ids = subset_a.read_text().split()
    assert len(subset_ids) == 100
    assert len(set(subset_ids)) == 100
    assert subset_ids == sorted(subset_ids, key=int)
    assert len(full_a.read_text().split()) == 1000

    _, subset_c = generate_coco_splits.generate_splits(
        annotations, tmp_path / "c", seed=43, subset_size=100
    )
    assert subset_c.read_text() != subset_a.read_text()


def test_oversized_subset_raises(tmp_path: Path) -> None:
    annotations = tmp_path / "instances_val2017.json"
    _write_annotations(annotations, 10)
    import pytest

    with pytest.raises(ValueError, match="exceeds"):
        generate_coco_splits.generate_splits(
            annotations, tmp_path / "out", subset_size=11
        )
