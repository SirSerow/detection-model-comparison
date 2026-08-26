"""PyTorch runtime stub."""

from edgebench.runtimes._stub import StubRuntime


class PyTorchRuntime(StubRuntime):
    @property
    def name(self) -> str:
        return "pytorch"
