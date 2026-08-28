"""NCNN export.

Converts an ONNX model through the current ``pnnx`` CLI into a ``.param`` /
``.bin`` pair. Optional INT8 quantization runs
``ncnn2int8`` after calibrating with ``ncnn2table`` over the deterministic
``benchmark_500`` image list. Both tools are part of an ncnn build and are
typically run on (or cross-compiled for) the Raspberry Pi.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
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
    pnnx = _find_pnnx()
    if pnnx is None:
        raise RuntimeError("pnnx not found on PATH; install the pinned pnnx release")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    param_path = (directory / f"{model_stem}.param").resolve()
    bin_path = (directory / f"{model_stem}.bin").resolve()
    # pnnx writes several conversion sidecars beside its input. Isolate those
    # in a temporary directory and request only the canonical ncnn outputs in
    # the benchmark's weights directory.
    with tempfile.TemporaryDirectory(prefix="edgebench-pnnx-") as temp_dir:
        temp_source = Path(temp_dir) / source.name
        shutil.copy2(source, temp_source)
        subprocess.run(
            [
                pnnx,
                str(temp_source),
                f"ncnnparam={param_path}",
                f"ncnnbin={bin_path}",
                "fp16=0",
            ],
            check=True,
            cwd=temp_dir,
        )
    for path in (param_path, bin_path):
        if not path.is_file():
            raise RuntimeError(f"pnnx did not produce expected NCNN artifact: {path}")
    return param_path, bin_path


def _find_pnnx() -> str | None:
    executable = shutil.which("pnnx")
    if executable is not None:
        return executable
    sibling = Path(sys.executable).with_name("pnnx")
    if sibling.is_file():
        return str(sibling)
    try:
        import pnnx
    except ImportError:
        return None
    packaged = Path(pnnx.EXEC_PATH)
    return str(packaged) if packaged.is_file() else None


def quantize_ncnn_int8(
    param_path: str,
    bin_path: str,
    table_path: str,
    *,
    output_base: str | Path | None = None,
) -> tuple[Path, Path]:
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
    if output_base is None:
        out_param = param.with_name(param.stem + "_int8.param")
        out_bin = binary.with_name(binary.stem + "_int8.bin")
    else:
        base = Path(output_base)
        out_param = base.with_suffix(".param")
        out_bin = base.with_suffix(".bin")
        out_param.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ncnn2int8, str(param), str(binary), str(out_param), str(out_bin), str(table)],
        check=True,
    )
    return out_param, out_bin
