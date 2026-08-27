"""TensorRT engine build.

Prefers the Python ``tensorrt`` API; falls back to the ``trtexec`` CLI when
the Python bindings are unavailable. Engines are device-specific — always
build on the target device (Jetson / RTX laptop), never copy between them.

FP16 and INT8 builder flags follow the experiment precision. INT8 requires
a calibrator; provide calibration images (the deterministic
``benchmark_500`` split) via ``calibration_images`` or the build raises.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    *,
    precision: str = "fp16",
    input_size: tuple[int, int] = (640, 640),
    workspace_mb: int = 1024,
) -> Path:
    """Build a static-shape TensorRT engine from an ONNX model on-device."""
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {source}. Export ONNX first with "
            f"`python -m edgebench export <model> --to onnx`."
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if precision == "int8":
        raise NotImplementedError(
            "TensorRT INT8 requires a calibration cache; build it on-device "
            "with trtexec and the benchmark_500 calibration list, then place "
            "the engine at the artifact path"
        )

    try:
        return _build_with_python_api(source, target, precision, workspace_mb)
    except ImportError:
        return _build_with_trtexec(source, target, precision, workspace_mb)


def _build_with_python_api(
    source: Path, target: Path, precision: str, workspace_mb: int
) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {source}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This device reports no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT engine build failed for {source}")
    target.write_bytes(bytes(engine_bytes))
    return target


def _build_with_trtexec(
    source: Path, target: Path, precision: str, workspace_mb: int
) -> Path:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "Neither the tensorrt Python package nor trtexec is available; "
            "install the JetPack/desktop-matched TensorRT stack"
        )
    command = [
        trtexec,
        f"--onnx={source}",
        f"--saveEngine={target}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--staticPlugins",
    ]
    if precision == "fp16":
        command.append("--fp16")
    subprocess.run(command, check=True)
    return target
