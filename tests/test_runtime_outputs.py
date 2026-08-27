"""Runtime backend output tests.

The ONNX Runtime test builds a tiny handcrafted ONNX graph so no model
download or torch install is required. Skipped automatically when
onnxruntime / onnx are not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgebench.runtimes.base import RuntimeSessionConfig
from edgebench.runtimes.onnxruntime import ONNXRuntimeBackend


def _write_scale_model(path: Path, scale: float = 2.0) -> None:
    onnx = pytest.importorskip("onnx")

    scale_tensor = onnx.helper.make_tensor(
        "scale", onnx.TensorProto.FLOAT, [], [scale]
    )
    node = onnx.helper.make_node("Mul", ["input", "scale"], ["output"])
    graph = onnx.helper.make_graph(
        [node],
        "scale_graph",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "output", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
            )
        ],
        initializer=[scale_tensor],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
    model.ir_version = 8  # stay within onnxruntime's supported IR range
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _write_add_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    inputs = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        )
        for name in ("left", "right")
    ]
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["left", "right"], ["output"])],
        "add_graph",
        inputs,
        [output],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_onnxruntime_backend_executes_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    artifact = tmp_path / "scale.onnx"
    _write_scale_model(artifact)

    session = RuntimeSessionConfig(
        name="onnxruntime",
        precision="fp32",
        execution_provider="CPUExecutionProvider",
        threads=2,
        artifact_path=str(artifact),
    )
    backend = ONNXRuntimeBackend(session)
    backend.load()
    backend.warmup(np.ones((1, 3, 8, 8), dtype=np.float32))
    output = backend.infer(np.full((1, 3, 8, 8), 3.0, dtype=np.float32))
    assert output.shape == (1, 3, 8, 8)
    assert np.allclose(output, 6.0)
    backend.synchronize()  # no-op for ORT; must not raise


def test_onnxruntime_missing_artifact_raises(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    session = RuntimeSessionConfig(
        name="onnxruntime",
        precision="fp32",
        artifact_path=str(tmp_path / "missing.onnx"),
    )
    backend = ONNXRuntimeBackend(session)
    with pytest.raises(FileNotFoundError, match="Export it first"):
        backend.load()


def test_onnxruntime_rejects_unavailable_provider(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    artifact = tmp_path / "scale.onnx"
    _write_scale_model(artifact)
    backend = ONNXRuntimeBackend(
        RuntimeSessionConfig(
            name="onnxruntime",
            precision="fp32",
            execution_provider="MissingExecutionProvider",
            artifact_path=str(artifact),
        )
    )
    with pytest.raises(RuntimeError, match="is not available"):
        backend.load()


def test_onnxruntime_supports_named_multi_input(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    artifact = tmp_path / "add.onnx"
    _write_add_model(artifact)
    backend = ONNXRuntimeBackend(
        RuntimeSessionConfig(
            name="onnxruntime",
            precision="fp32",
            execution_provider="CPUExecutionProvider",
            artifact_path=str(artifact),
        )
    )
    backend.load()
    output = backend.infer(
        {
            "left": np.ones((1, 3, 8, 8), dtype=np.float32),
            "right": np.full((1, 3, 8, 8), 2.0, dtype=np.float32),
        }
    )
    assert np.allclose(output, 3.0)
    with pytest.raises(ValueError, match="Missing ONNX inputs"):
        backend.infer({"left": np.ones((1, 3, 8, 8), dtype=np.float32)})


def test_fp16_conversion_uses_float16_io(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from edgebench.exporters.onnx import convert_onnx_to_fp16

    artifact = tmp_path / "scale.onnx"
    _write_scale_model(artifact)
    convert_onnx_to_fp16(artifact)

    model = onnx.load(str(artifact))
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert model.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16

    backend = ONNXRuntimeBackend(
        RuntimeSessionConfig(
            name="onnxruntime",
            precision="fp16",
            execution_provider="CPUExecutionProvider",
            artifact_path=str(artifact),
        )
    )
    backend.load()
    output = backend.infer(np.full((1, 3, 8, 8), 3.0, dtype=np.float32))
    assert output.dtype == np.float16
    assert np.allclose(output, 6.0)
