#!/usr/bin/env python3
"""Generate deterministic COCO val2017 split files.

Reads ``datasets/coco/annotations/instances_val2017.json`` and writes:

- ``datasets/splits/coco_val2017_full.txt`` — every image id, sorted.
- ``datasets/splits/coco_benchmark_500.txt`` — a fixed-seed sample of
  ``--subset-size`` image ids, sorted for stable iteration order.

Split files list image ids only (one per line). Regenerating with the same
seed reproduces identical ids, so every device, model, and runtime measures
the same images. Record the seed in benchmark metadata; the default is the
canonical project seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from edgebench.paths import REPO_ROOT

DEFAULT_SEED = 20240613
DEFAULT_SUBSET_SIZE = 500


def generate_splits(
    annotations_path: Path,
    splits_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    subset_size: int = DEFAULT_SUBSET_SIZE,
) -> tuple[Path, Path]:
    with annotations_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    image_ids = sorted(int(image["id"]) for image in data["images"])
    if not image_ids:
        raise ValueError(f"No images found in {annotations_path}")
    if subset_size > len(image_ids):
        raise ValueError(
            f"Subset size {subset_size} exceeds available images ({len(image_ids)})"
        )

    rng = random.Random(seed)
    subset_ids = sorted(rng.sample(image_ids, subset_size))

    splits_dir.mkdir(parents=True, exist_ok=True)
    full_path = splits_dir / "coco_val2017_full.txt"
    subset_path = splits_dir / f"coco_benchmark_{subset_size}.txt"
    full_path.write_text(
        "".join(f"{image_id}\n" for image_id in image_ids), encoding="utf-8"
    )
    subset_path.write_text(
        "".join(f"{image_id}\n" for image_id in subset_ids), encoding="utf-8"
    )
    return full_path, subset_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json",
        help="Path to instances_val2017.json",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "splits",
        help="Output directory for split files",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    args = parser.parse_args()

    if not args.annotations.is_file():
        parser.error(
            f"Annotations not found: {args.annotations}\n"
            "Download instances_val2017.json first (see datasets/README.md)."
        )

    full_path, subset_path = generate_splits(
        args.annotations,
        args.splits_dir,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    print(f"seed={args.seed}")
    print(f"wrote {full_path}")
    print(f"wrote {subset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
