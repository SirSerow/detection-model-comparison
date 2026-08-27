"""TensorRT runtime.

Deserializes a prebuilt engine from ``session.artifact_path`` and executes
it with static input shapes (the benchmark fixes 1×3×640×640). Uses the
TensorRT V3 execution API (``set_tensor_address`` / ``execute_async_v3``),
available in TensorRT ≥ 8.5 (JetPack 5/6 and current desktop releases).

CUDA memory management uses the ``cuda-python`` bindings when present and
falls back to ``pycuda``; both ship with typical JetPack/desktop setups.

Device-verified path: engine builds and timing are validated on Jetson /
RTX hardware, not in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.runtimes.base import RuntimeBackend, RuntimeSessionConfig

if TYPE_CHECKING:
    import numpy as np


class TensorRTRuntime(RuntimeBackend):
    def __init__(self, session: RuntimeSessionConfig | None = None) -> None:
        self.session = session
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._cuda: Any = None
        self._buffers: dict[str, tuple[Any, Any]] = {}
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    @property
    def name(self) -> str:
        return "tensorrt"

    def load(self) -> None:
        import numpy as np
        import tensorrt as trt

        session = self.session or RuntimeSessionConfig(name="tensorrt", precision="fp16")
        if not session.artifact_path:
            raise RuntimeError(
                "TensorRTRuntime requires session.artifact_path; "
                "the benchmark runner resolves it from the export convention"
            )
        artifact = Path(session.artifact_path)
        if not artifact.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {artifact}. Build it on-device with "
                f"`python -m edgebench export <model> --to tensorrt`."
            )

        self._cuda = _CudaBindings()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(artifact.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {artifact}")
        self._context = self._engine.create_execution_context()
        self._stream = self._cuda.create_stream()

        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(self._engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Engine tensor '{name}' has dynamic shape {shape}; the "
                    "benchmark requires static input shapes"
                )
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            device_buffer = self._cuda.malloc(nbytes)
            host_buffer = np.empty(shape, dtype=dtype)
            self._buffers[name] = (host_buffer, device_buffer)
            self._context.set_tensor_address(name, device_buffer)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def warmup(self, input_data: np.ndarray) -> None:
        """One untimed inference; the runner drives the warm-up count."""
        self.infer(input_data)
        self.synchronize()

    def infer(self, input_data: Any) -> Any:
        import numpy as np

        if self._context is None:
            raise RuntimeError("TensorRTRuntime.load() must run before infer()")
        if isinstance(input_data, dict):
            missing = [name for name in self._input_names if name not in input_data]
            if missing:
                raise ValueError(f"Missing TensorRT inputs: {', '.join(missing)}")
            inputs = input_data
        else:
            if len(self._input_names) != 1:
                raise ValueError(
                    f"TensorRT engine requires inputs {self._input_names}; provide a mapping"
                )
            inputs = {self._input_names[0]: input_data}
        for name in self._input_names:
            host_input, device_input = self._buffers[name]
            array = np.ascontiguousarray(inputs[name], dtype=host_input.dtype)
            if array.shape != host_input.shape:
                raise ValueError(
                    f"Input '{name}' shape {array.shape} does not match engine "
                    f"input {host_input.shape}"
                )
            self._cuda.copy_h2d(device_input, array, self._stream)
        self._context.execute_async_v3(self._stream)
        outputs = []
        for name in self._output_names:
            host_output, device_output = self._buffers[name]
            self._cuda.copy_d2h(host_output, device_output, self._stream)
            outputs.append(host_output.copy())
        self.synchronize()
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def synchronize(self) -> None:
        if self._stream is not None:
            self._cuda.synchronize(self._stream)


class _CudaBindings:
    """Thin wrapper over cuda-python (preferred) or pycuda."""

    def __init__(self) -> None:
        try:
            from cuda import cuda as cuda_driver

            self._kind = "cuda-python"
            self._drv = cuda_driver
            self._drv.cuInit(0)
        except ImportError:
            import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
            import pycuda.driver as cuda_driver

            self._kind = "pycuda"
            self._drv = cuda_driver

    def create_stream(self) -> Any:
        if self._kind == "cuda-python":
            _, stream = self._drv.cuStreamCreate(0)
            return stream
        return self._drv.Stream()

    def malloc(self, nbytes: int) -> Any:
        if self._kind == "cuda-python":
            _, pointer = self._drv.cuMemAlloc(nbytes)
            return pointer
        return self._drv.mem_alloc(nbytes)

    def copy_h2d(self, device: Any, array: Any, stream: Any) -> None:
        if self._kind == "cuda-python":
            self._drv.cuMemcpyHtoDAsync(device, array.ctypes.data, array.nbytes, stream)
        else:
            self._drv.memcpy_htod_async(device, array, stream)

    def copy_d2h(self, array: Any, device: Any, stream: Any) -> None:
        if self._kind == "cuda-python":
            self._drv.cuMemcpyDtoHAsync(array.ctypes.data, device, array.nbytes, stream)
        else:
            self._drv.memcpy_dtoh_async(array, device, stream)

    def synchronize(self, stream: Any) -> None:
        if self._kind == "cuda-python":
            self._drv.cuStreamSynchronize(stream)
        else:
            stream.synchronize()
