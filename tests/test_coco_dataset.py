"""CocoDataset tests over a tiny synthetic COCO fixture. No downloads required."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgebench.dataset_adapters import CocoDataset
from edgebench.types import Detection

IMAGE_IDS = [11, 22, 33]


def _write_fixture(root: Path) -> Path:
    coco = root / "coco"
    images_dir = coco / "val2017"
    annotations_dir = coco / "annotations"
    splits_dir = root / "splits"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    images = []
    annotations = []
    for index, image_id in enumerate(IMAGE_IDS):
        file_name = f"{image_id:012d}.jpg"
        image = np.full((80, 120, 3), fill_value=40 * (index + 1), dtype=np.uint8)
        assert cv2.imwrite(str(images_dir / file_name), image)
        images.append(
            {"id": image_id, "file_name": file_name, "width": 120, "height": 80}
        )
        annotations.append(
            {
                "id": index + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [10.0, 10.0, 30.0, 40.0],
                "area": 1200.0,
                "iscrowd": 0,
            }
        )
    payload = {
        "info": {"description": "synthetic fixture"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        ],
    }
    (annotations_dir / "instances_val2017.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (splits_dir / "coco_val2017_full.txt").write_text(
        "".join(f"{image_id}\n" for image_id in IMAGE_IDS), encoding="utf-8"
    )
    (splits_dir / "coco_benchmark_500.txt").write_text("22\n33\n", encoding="utf-8")
    return root


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    return _write_fixture(tmp_path)


def test_dataset_length_and_samples(dataset_root: Path) -> None:
    dataset = CocoDataset(dataset_root, split="val2017_full")
    assert len(dataset) == 3
    sample = dataset.get_sample(1)
    assert sample.image_id == 22
    assert (sample.width, sample.height) == (120, 80)


def test_split_filters_images(dataset_root: Path) -> None:
    dataset = CocoDataset(dataset_root, split="benchmark_500")
    assert len(dataset) == 2
    assert dataset.image_id(0) == 22
    assert dataset.get_annotations(0)[0]["image_id"] == 22


def test_get_image_decodes_bgr(dataset_root: Path) -> None:
    dataset = CocoDataset(dataset_root, split="val2017_full")
    image = dataset.get_image(0)
    assert image.shape == (80, 120, 3)
    assert image.dtype == np.uint8
    assert int(image[0, 0, 0]) == 40


def test_unknown_split_id_raises(dataset_root: Path) -> None:
    (dataset_root / "splits" / "coco_val2017_full.txt").write_text(
        "11\n999\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing"):
        CocoDataset(dataset_root, split="val2017_full")


def test_missing_files_raise(dataset_root: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotations"):
        CocoDataset(tmp_path / "empty", split="val2017_full")
    (dataset_root / "splits" / "coco_missing.txt").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="Split file"):
        CocoDataset(dataset_root, split="missing")


def test_evaluate_perfect_predictions(dataset_root: Path) -> None:
    dataset = CocoDataset(dataset_root, split="val2017_full")
    predictions = {
        image_id: [Detection(bbox=(10.0, 10.0, 40.0, 50.0), score=0.99, class_id=1)]
        for image_id in IMAGE_IDS
    }
    metrics = dataset.evaluate(predictions)
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["map50_95"] == pytest.approx(1.0)
    assert "ap75" in metrics


def test_evaluate_empty_predictions_raises(dataset_root: Path) -> None:
    dataset = CocoDataset(dataset_root, split="val2017_full")
    with pytest.raises(ValueError, match="No predictions"):
        dataset.evaluate({})
