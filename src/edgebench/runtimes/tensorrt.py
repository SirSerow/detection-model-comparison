"""TensorRT runtime stub."""

from edgebench.runtimes._stub import StubRuntime


class TensorRTRuntime(StubRuntime):
    @property
    def name(self) -> str:
        return "tensorrt"
