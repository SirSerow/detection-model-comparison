"""Map runtime names to backend classes."""

from __future__ import annotations

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig
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

    def get(
        self,
        name: str,
        session: RuntimeSessionConfig | None = None,
    ) -> RuntimeBackend:
        try:
            cls = RUNTIMES[name]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(f"Unknown runtime '{name}'. Known: {known}") from exc
        return cls(session)


def list_runtimes() -> list[str]:
    return RuntimeRegistry().names()


def get_runtime(
    name: str,
    session: RuntimeSessionConfig | None = None,
) -> RuntimeBackend:
    return RuntimeRegistry().get(name, session)
