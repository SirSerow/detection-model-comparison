"""MS COCO 2017 adapter.

Loads ``instances_val2017.json`` and a deterministic split file from
``datasets/splits/coco_<split>.txt`` (image ids, one per line). Images are
decoded lazily as BGR ndarrays via OpenCV; evaluation uses pycocotools and
expects predictions in the canonical :class:`~edgebench.types.Detection`
format (xyxy, original-image coordinates, canonical COCO category ids).

Expected dataset layout (see ``datasets/README.md``)::

    datasets/
    ├── coco/
    │   ├── annotations/instances_val2017.json
    │   └── val2017/
    └── splits/
        ├── coco_val2017_full.txt
        └── coco_benchmark_500.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.dataset_adapters.base import DatasetAdapter
from edgebench.paths import REPO_ROOT

if TYPE_CHECKING:
    import numpy as np

    from edgebench.types import Detection

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets"


@dataclass(frozen=True)
class CocoSample:
    """Metadata for one image in the selected split."""

    image_id: int
    file_name: str
    width: int
    height: int


class CocoDataset(DatasetAdapter):
    """COCO val2017 adapter over a deterministic image-id split."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "benchmark_500",
        *,
        annotation_file: str = "annotations/instances_val2017.json",
        image_dir: str = "val2017",
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_DATASET_ROOT
        self.split = split
        self.image_dir = self.root / "coco" / image_dir
        annotations_path = self.root / "coco" / annotation_file
        split_path = self.root / "splits" / f"coco_{split}.txt"

        if not annotations_path.is_file():
            raise FileNotFoundError(
                f"COCO annotations not found: {annotations_path}. "
                "Download instances_val2017.json (see datasets/README.md)."
            )
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. "
                "Generate it with scripts/generate_coco_splits.py."
            )

        with annotations_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self._annotations_path = annotations_path
        self._images: dict[int, CocoSample] = {
            int(image["id"]): CocoSample(
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
            )
            for image in data["images"]
        }
        self._annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in data["annotations"]:
            self._annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
                annotation
            )

        with split_path.open(encoding="utf-8") as handle:
            split_ids = [int(line) for line in handle if line.strip()]
        unknown = [image_id for image_id in split_ids if image_id not in self._images]
        if unknown:
            raise ValueError(
                f"Split {split_path} references {len(unknown)} image ids missing "
                f"from {annotations_path} (first: {unknown[0]})"
            )
        self._split_ids = split_ids

    def __len__(self) -> int:
        return len(self._split_ids)

    def image_id(self, index: int) -> int:
        return self._split_ids[index]

    def get_sample(self, index: int) -> CocoSample:
        return self._images[self._split_ids[index]]

    def image_path(self, index: int) -> Path:
        return self.image_dir / self.get_sample(index).file_name

    def get_image(self, index: int) -> np.ndarray:
        """Decode one image as a BGR ndarray (HWC, uint8)."""
        import cv2

        path = self.image_path(index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        return image

    def get_annotations(self, index: int) -> list[dict[str, Any]]:
        """Raw COCO annotation dicts for one image (xywh, canonical category ids)."""
        return list(self._annotations_by_image.get(self._split_ids[index], []))

    def evaluate(
        self, predictions: dict[int, list[Detection]], *, verbose: bool = False
    ) -> dict[str, float]:
        """Run official COCO evaluation.

        Args:
            predictions: mapping of image id to detections in canonical format
                (xyxy, original-image coordinates, canonical COCO category ids).
            verbose: print the full pycocotools summary table.

        Returns:
            Dict with map50, map50_95, ap75, ap_small, ap_medium, ap_large.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(str(self._annotations_path))
        image_ids = [image_id for image_id in self._split_ids if image_id in predictions]
        results = []
        for image_id in image_ids:
            for detection in predictions[image_id]:
                x1, y1, x2, y2 = detection.bbox
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": detection.class_id,
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(max(x2 - x1, 0.0)),
                            float(max(y2 - y1, 0.0)),
                        ],
                        "score": float(detection.score),
                    }
                )
        if not results:
            raise ValueError(
                "No predictions to evaluate; refusing to report a misleading zero mAP"
            )

        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        if verbose:
            evaluator.summarize()
        else:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.summarize()
        stats = evaluator.stats
        return {
            "map50_95": float(stats[0]),
            "map50": float(stats[1]),
            "ap75": float(stats[2]),
            "ap_small": float(stats[3]),
            "ap_medium": float(stats[4]),
            "ap_large": float(stats[5]),
        }
