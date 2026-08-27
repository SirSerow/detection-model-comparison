"""NCNN export.

Converts an ONNX model through the ``onnx2ncnn`` CLI (from the ncnn build)
into a ``.param`` / ``.bin`` pair. Optional INT8 quantization runs
``ncnn2int8`` after calibrating with ``ncnn2table`` over the deterministic
``benchmark_500`` image list. Both tools are part of an ncnn build and are
typically run on (or cross-compiled for) the Raspberry Pi.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def export_ncnn(onnx_path: str, output_dir: str, *, model_stem: str) -> tuple[Path, Path]:
    """Convert ONNX to an NCNN ``.param`` / ``.bin`` pair.

    Returns ``(param_path, bin_path)`` where both share
    ``<output_dir>/<model_stem>`` as their base.
    """
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {source}. Export ONNX first with "
            f"`python -m edgebench export <model> --to onnx`."
        )
    onnx2ncnn = shutil.which("onnx2ncnn")
    if onnx2ncnn is None:
        raise RuntimeError(
            "onnx2ncnn not found on PATH; build or install ncnn tools first"
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    param_path = directory / f"{model_stem}.param"
    bin_path = directory / f"{model_stem}.bin"
    subprocess.run(
        [onnx2ncnn, str(source), str(param_path), str(bin_path)], check=True
    )
    return param_path, bin_path


def quantize_ncnn_int8(param_path: str, bin_path: str, table_path: str) -> Path:
    """Quantize an FP32 NCNN pair to INT8 using a precomputed scale table.

    The table is produced by ``ncnn2table`` over the deterministic
    benchmark_500 calibration images; generating it is a deliberate,
    recorded deployment step.
    """
    ncnn2int8 = shutil.which("ncnn2int8")
    if ncnn2int8 is None:
        raise RuntimeError("ncnn2int8 not found on PATH; build ncnn tools first")
    param = Path(param_path)
    binary = Path(bin_path)
    table = Path(table_path)
    for path in (param, binary, table):
        if not path.is_file():
            raise FileNotFoundError(f"Missing INT8 quantization input: {path}")
    out_param = param.with_name(param.stem + "_int8.param")
    out_bin = binary.with_name(binary.stem + "_int8.bin")
    subprocess.run(
        [ncnn2int8, str(param), str(binary), str(out_param), str(out_bin), str(table)],
        check=True,
    )
    return out_param
