"""NCNN runtime.

Loads the ``.param`` / ``.bin`` pair sharing ``session.artifact_path`` as
its base (no extension). Thread count comes from the device profile.

Blob naming convention: ``onnx2ncnn`` names the single input ``in0`` and
outputs ``out0``, ``out1``, ... Output names are discovered from the
``.param`` file's final layer so multi-head exports work.

Device-verified path: validated on Raspberry Pi 4, not in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig

if TYPE_CHECKING:
    import numpy as np


class NCNNRuntime(RuntimeBackend):
    def __init__(self, session: RuntimeSessionConfig | None = None) -> None:
        self.session = session
        self._net: Any = None
        self._param_path: Path | None = None
        self._input_names = ["in0"]
        self._output_names: list[str] = []

    @property
    def name(self) -> str:
        return "ncnn"

    def load(self) -> None:
        import ncnn

        session = self.session or RuntimeSessionConfig(name="ncnn", precision="fp32")
        if not session.artifact_path:
            raise RuntimeError(
                "NCNNRuntime requires session.artifact_path; "
                "the benchmark runner resolves it from the export convention"
            )
        base = Path(session.artifact_path)
        param_path = base.with_suffix(".param")
        bin_path = base.with_suffix(".bin")
        for path in (param_path, bin_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"NCNN artifact not found: {path}. Export it first with "
                    f"`python -m edgebench export <model> --to ncnn`."
                )
        self._param_path = param_path
        self._input_names, self._output_names = _read_blob_names(param_path)

        self._net = ncnn.Net()
        if session.threads:
            self._net.opt.num_threads = int(session.threads)
        self._net.load_param(str(param_path))
        self._net.load_model(str(bin_path))

    def warmup(self, input_data: np.ndarray) -> None:
        """One untimed inference; the runner drives the warm-up count."""
        self.infer(input_data)

    def infer(self, input_data: Any) -> Any:
        import numpy as np

        if self._net is None:
            raise RuntimeError("NCNNRuntime.load() must run before infer()")
        extractor = self._net.create_extractor()
        if isinstance(input_data, dict):
            missing = [name for name in self._input_names if name not in input_data]
            if missing:
                raise ValueError(f"Missing NCNN inputs: {', '.join(missing)}")
            inputs = input_data
        else:
            if len(self._input_names) != 1:
                raise ValueError(
                    f"NCNN model requires inputs {self._input_names}; provide a mapping"
                )
            inputs = {self._input_names[0]: input_data}
        for name in self._input_names:
            array = np.ascontiguousarray(inputs[name], dtype=np.float32)
            # The binding accepts CHW directly; strip only a leading batch=1.
            value = array[0] if array.ndim >= 2 and array.shape[0] == 1 else array
            extractor.input(name, ncnn_mat(value))
        outputs = []
        for name in self._output_names:
            result, mat = extractor.extract(name)
            if result != 0:
                raise RuntimeError(f"ncnn extract failed for blob '{name}' ({result})")
            outputs.append(np.asarray(mat))
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def synchronize(self) -> None:
        # ncnn extractor.extract blocks until the network finishes.
        return None


def ncnn_mat(chw_array: np.ndarray) -> Any:
    import ncnn

    return ncnn.Mat(chw_array)


def _read_blob_names(param_path: Path) -> tuple[list[str], list[str]]:
    """Best-effort parse of input/output blob names from a .param file.

    Falls back to the onnx2ncnn convention (``in0`` / ``out0``) when the
    file cannot be interpreted.
    """
    input_names: list[str] = []
    output_names: list[str] = []
    try:
        lines = param_path.read_text(encoding="utf-8").splitlines()
        for line in lines[2:]:  # skip magic and layer/blob counts
            parts = line.split()
            if len(parts) < 4:
                continue
            layer_type = parts[0]
            bottom_count = int(parts[2])
            top_count = int(parts[3])
            tops = parts[4 + bottom_count : 4 + bottom_count + top_count]
            if layer_type == "Input" and tops:
                input_names.extend(tops)
            if top_count == 0:
                output_names.extend(parts[4 : 4 + bottom_count])
    except (ValueError, IndexError):
        return ["in0"], ["out0"]
    return input_names or ["in0"], output_names or ["out0"]
