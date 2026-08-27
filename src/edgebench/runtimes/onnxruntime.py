"""ONNX Runtime backend.

Loads the exported ONNX artifact from ``session.artifact_path`` and honors
the device profile's execution provider and thread count. ORT ``run`` is
blocking, so ``synchronize`` is a no-op by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig

if TYPE_CHECKING:
    import numpy as np


class ONNXRuntimeBackend(RuntimeBackend):
    def __init__(self, session: RuntimeSessionConfig | None = None) -> None:
        self.session = session
        self._session: Any = None
        self._input_names: list[str] = []

    @property
    def name(self) -> str:
        return "onnxruntime"

    def load(self) -> None:
        import onnxruntime

        session = self.session or RuntimeSessionConfig(
            name="onnxruntime", precision="fp32"
        )
        if not session.artifact_path:
            raise RuntimeError(
                "ONNXRuntimeBackend requires session.artifact_path; "
                "the benchmark runner resolves it from the export convention"
            )
        artifact = Path(session.artifact_path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"ONNX artifact not found: {artifact}. Export it first with "
                f"`python -m edgebench export <model> --to onnx`."
            )
        options = onnxruntime.SessionOptions()
        if session.threads:
            options.intra_op_num_threads = int(session.threads)
        providers = [session.execution_provider] if session.execution_provider else None
        self._session = onnxruntime.InferenceSession(
            str(artifact), sess_options=options, providers=providers
        )
        self._input_names = [item.name for item in self._session.get_inputs()]

    def warmup(self, input_data: np.ndarray) -> None:
        """One untimed inference; the runner drives the warm-up count."""
        self.infer(input_data)

    def infer(self, input_data: Any) -> Any:
        import numpy as np

        if self._session is None:
            raise RuntimeError("ONNXRuntimeBackend.load() must run before infer()")
        if isinstance(input_data, dict):
            missing = [name for name in self._input_names if name not in input_data]
            if missing:
                raise ValueError(f"Missing ONNX inputs: {', '.join(missing)}")
            feed = {
                name: np.ascontiguousarray(input_data[name], dtype=np.float32)
                for name in self._input_names
            }
        else:
            if len(self._input_names) != 1:
                raise ValueError(
                    f"ONNX model requires inputs {self._input_names}; provide a mapping"
                )
            feed = {
                self._input_names[0]: np.ascontiguousarray(
                    input_data, dtype=np.float32
                )
            }
        outputs = self._session.run(None, feed)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def synchronize(self) -> None:
        # onnxruntime InferenceSession.run blocks until completion.
        return None
