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
        return _build_with_python_api(
            source, target, precision, input_size, workspace_mb
        )
    except ImportError:
        return _build_with_trtexec(source, target, precision, workspace_mb)


def _build_with_python_api(
    source: Path,
    target: Path,
    precision: str,
    input_size: tuple[int, int],
    workspace_mb: int,
) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Explicit batch became mandatory and its creation flag was removed in
    # TensorRT 11. Keep the flag for TensorRT 8-10 / JetPack compatibility.
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 1 << int(explicit_batch) if explicit_batch is not None else 0
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if precision == "fp16" and fp16_flag is None:
        strongly_typed = getattr(
            trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None
        )
        if strongly_typed is None:
            raise RuntimeError("TensorRT has neither FP16 nor strongly-typed support")
        network_flags |= 1 << int(strongly_typed)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {source}: {errors}")
    if (
        precision == "fp16"
        and fp16_flag is None
        and network.get_input(0).dtype != trt.DataType.HALF
    ):
        raise RuntimeError(
            "TensorRT 11 requires a true FP16 ONNX graph for an FP16 engine; "
            "export it first with `python -m edgebench export <model> --to onnx "
            "--precision fp16`"
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    dynamic_inputs = [
        network.get_input(index)
        for index in range(network.num_inputs)
        if any(dimension < 0 for dimension in network.get_input(index).shape)
    ]
    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        width, height = input_size
        for tensor in dynamic_inputs:
            shape = list(tensor.shape)
            shape = [1 if dimension < 0 else dimension for dimension in shape]
            if len(shape) == 4:
                shape[2], shape[3] = height, width
            static_shape = tuple(shape)
            profile.set_shape(tensor.name, static_shape, static_shape, static_shape)
        config.add_optimization_profile(profile)
    if precision == "fp16":
        # TensorRT 11 removed the capability property; unsupported precision
        # is reported by the builder when setting/building the flagged graph.
        if not getattr(builder, "platform_has_fast_fp16", True):
            raise RuntimeError("This device reports no fast FP16 support")
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
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
