"""ONNX Runtime backend stub."""

from edgebench.runtimes._stub import StubRuntime


class ONNXRuntimeBackend(StubRuntime):
    @property
    def name(self) -> str:
        return "onnxruntime"
