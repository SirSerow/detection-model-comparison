"""PyTorch runtime.

Executes the framework model returned by ``DetectorAdapter.load_pytorch``.
Preprocessing and decode stay on the adapter; this backend only moves the
model and inputs to the configured device/precision and runs the forward
pass under ``torch.no_grad``.

Device and precision come from ``RuntimeSessionConfig`` (populated from the
device YAML backends matrix). FP16 is applied only on CUDA; the device
capability matrix already rejects FP16 on CPU-only profiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig

if TYPE_CHECKING:
    import numpy as np


class PyTorchRuntime(RuntimeBackend):
    def __init__(self, session: RuntimeSessionConfig | None = None) -> None:
        self.session = session
        self._model: Any = None
        self._device: Any = None
        self._fp16 = False

    @property
    def name(self) -> str:
        return "pytorch"

    def attach_model(self, model: Any) -> None:
        """Provide the framework model before :meth:`load`."""
        self._model = model

    def load(self) -> None:
        import torch

        if self._model is None:
            raise RuntimeError(
                "PyTorchRuntime requires a model; call attach_model() first "
                "(the benchmark runner passes adapter.load_pytorch())"
            )
        session = self.session or RuntimeSessionConfig(name="pytorch", precision="fp32")
        device_target = session.device_target or "cpu"
        if device_target == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Device profile targets CUDA but torch.cuda.is_available() is False; "
                "fix the installation or the device YAML"
            )
        self._device = torch.device(device_target)
        self._fp16 = session.precision == "fp16"
        self._model.eval()
        self._model.to(self._device)
        if self._fp16:
            self._model.half()

    def warmup(self, input_data: np.ndarray) -> None:
        """One untimed forward pass; the runner drives the warm-up count."""
        self.infer(input_data)
        self.synchronize()

    def infer(self, input_data: np.ndarray) -> Any:
        import numpy as np
        import torch

        if self._model is None or self._device is None:
            raise RuntimeError("PyTorchRuntime.load() must run before infer()")
        array = np.ascontiguousarray(input_data, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(array).to(self._device)
            if self._fp16:
                tensor = tensor.half()
            output = self._model(tensor)
        return _to_numpy(output)

    def synchronize(self) -> None:
        if self._device is None or self._device.type != "cuda":
            return
        import torch

        torch.cuda.synchronize(self._device)


def _to_numpy(output: Any) -> Any:
    import numpy as np

    if isinstance(output, dict):
        return {key: _to_numpy(value) for key, value in output.items()}
    if isinstance(output, (tuple, list)):
        return type(output)(_to_numpy(item) for item in output)
    if hasattr(output, "detach"):
        return output.detach().float().cpu().numpy()
    return np.asarray(output)
