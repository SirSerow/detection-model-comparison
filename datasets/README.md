# Datasets

Do not commit COCO images or annotation JSON to this repository.

## MS COCO 2017

Download locally:

1. `val2017` images
2. `annotations/instances_val2017.json`

Expected layout after download:

```text
datasets/
├── coco/
│   ├── annotations/
│   │   └── instances_val2017.json
│   └── val2017/
└── splits/
    ├── coco_val2017_full.txt
    └── coco_benchmark_500.txt
```

Split files list image IDs only. Run `python scripts/generate_coco_splits.py`
to regenerate the full and deterministic 500-image splits (seed 20240613),
then reuse the same IDs on every device, model, and runtime.

COCO source images retain their original Flickr licenses. Use them for evaluation; do not redistribute the image set from this repo.

A custom industrial dataset can be added later through `edgebench.dataset_adapters` without changing runtime or detector code.
