"""Map runtime names to backend classes."""

from __future__ import annotations

from edgebench.runtimes.base import RuntimeBackend
from edgebench.runtimes.ncnn import NCNNRuntime
from edgebench.runtimes.onnxruntime import ONNXRuntimeBackend
from edgebench.runtimes.pytorch import PyTorchRuntime
from edgebench.runtimes.tensorrt import TensorRTRuntime

RUNTIMES: dict[str, type[RuntimeBackend]] = {
    "pytorch": PyTorchRuntime,
    "onnxruntime": ONNXRuntimeBackend,
    "tensorrt": TensorRTRuntime,
    "ncnn": NCNNRuntime,
}


class RuntimeRegistry:
    def names(self) -> list[str]:
        return list(RUNTIMES)

    def get(self, name: str) -> RuntimeBackend:
        try:
            return RUNTIMES[name]()
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(f"Unknown runtime '{name}'. Known: {known}") from exc


def list_runtimes() -> list[str]:
    return RuntimeRegistry().names()


def get_runtime(name: str) -> RuntimeBackend:
    return RuntimeRegistry().get(name)
