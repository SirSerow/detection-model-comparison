"""Detector adapter interface.

Model-specific preprocess, decode, NMS, and export helpers live here.
Runtime backends must not contain this logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from edgebench.types import Detection


class DetectorAdapter(ABC):
    """Model-specific preprocess / postprocess / export hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_output: Any, metadata: Any) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def load_pytorch(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def export_onnx(self, output_path: str) -> None:
        raise NotImplementedError
