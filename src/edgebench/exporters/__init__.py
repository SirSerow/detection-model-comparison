"""Model export helpers and the exported-artifact path convention.

Artifacts live next to the model checkpoint under ``weights/<family>/`` and
are named ``<model>_<runtime>_<precision><ext>``. NCNN artifacts are a
``.param`` / ``.bin`` pair sharing one base path (no extension). Exported
binaries are git-ignored; produce them on the target device via
``python -m edgebench export`` so export conditions stay part of the
deployment record.
"""

from __future__ import annotations

from pathlib import Path

from edgebench.paths import REPO_ROOT

ARTIFACT_SUFFIXES: dict[str, str] = {
    "onnxruntime": ".onnx",
    "tensorrt": ".engine",
    "ncnn": "",
}


def default_weights_root() -> Path:
    return REPO_ROOT / "weights"


def artifact_path_for(
    model: str,
    runtime: str,
    precision: str,
    *,
    checkpoint: str | Path | None = None,
    weights_root: str | Path | None = None,
) -> Path:
    """Expected exported-artifact path for a model/runtime/precision.

    The artifact sits beside the model checkpoint when one is configured
    (e.g. ``weights/yolox/yolox_tiny_tensorrt_fp16.engine``). For NCNN the
    returned path is the shared base; append ``.param`` and ``.bin``.
    PyTorch models load checkpoints directly and have no artifact.
    """
    try:
        suffix = ARTIFACT_SUFFIXES[runtime]
    except KeyError as exc:
        raise ValueError(f"No artifact convention for runtime '{runtime}'") from exc
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        directory = checkpoint_path.parent
    else:
        root = Path(weights_root) if weights_root is not None else default_weights_root()
        directory = root / model
    return directory / f"{model}_{runtime}_{precision}{suffix}"
