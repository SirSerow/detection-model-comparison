"""NCNN runtime stub."""

from edgebench.runtimes._stub import StubRuntime


class NCNNRuntime(StubRuntime):
    @property
    def name(self) -> str:
        return "ncnn"
