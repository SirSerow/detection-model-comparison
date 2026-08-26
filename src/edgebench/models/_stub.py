"""Shared NotImplemented detector body used by all model stubs."""

from __future__ import annotations

from typing import Any

from edgebench.models.base import DetectorAdapter
from edgebench.types import Detection


class StubDetector(DetectorAdapter):
    """Concrete detectors only override ``name`` (and PicoDet PyTorch)."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def preprocess(self, image: Any) -> Any:
        raise NotImplementedError(f"{self.name} preprocess is not implemented yet")

    def postprocess(self, raw_output: Any, metadata: Any) -> list[Detection]:
        raise NotImplementedError(f"{self.name} postprocess is not implemented yet")

    def load_pytorch(self) -> Any:
        raise NotImplementedError(f"{self.name} load_pytorch is not implemented yet")

    def export_onnx(self, output_path: str) -> None:
        raise NotImplementedError(f"{self.name} export_onnx is not implemented yet")
