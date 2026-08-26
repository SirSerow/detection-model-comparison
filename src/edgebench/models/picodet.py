"""PP-PicoDet-S adapter stub.

PicoDet is Paddle-native. Do not create an artificial PyTorch benchmark
solely for symmetry. Compare exported ONNX / TensorRT / NCNN paths.
"""

from __future__ import annotations

from typing import Any

from edgebench.models._stub import StubDetector


class PicoDetSAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "picodet_s"

    def load_pytorch(self) -> Any:
        raise NotImplementedError(
            "PP-PicoDet-S is Paddle-native; PyTorch is N/A — unsupported"
        )
