# Datasets

Do not commit COCO images or annotation JSON to this repository.

## MS COCO 2017 (planned)

Download locally (not implemented by the skeleton):

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

Split files should list image IDs only. Generate `coco_benchmark_500.txt` once with a fixed seed and reuse the same IDs on every device, model, and runtime.

COCO source images retain their original Flickr licenses. Use them for evaluation; do not redistribute the image set from this repo.

A custom industrial dataset can be added later through `edgebench.dataset_adapters` without changing runtime or detector code.
