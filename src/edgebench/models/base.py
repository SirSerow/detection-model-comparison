"""Detector adapter interface.

Model-specific preprocess, decode, NMS, and export helpers live here.
Runtime backends must not contain this logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from edgebench.types import Detection

if TYPE_CHECKING:
    from edgebench.config import BenchmarkSettings, ModelConfig


class DetectorAdapter(ABC):
    """Model-specific preprocess / postprocess / export hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def configure(
        self, model: ModelConfig, benchmark: BenchmarkSettings
    ) -> None:
        """Attach experiment context before use.

        The runner calls this once after construction. Adapters read the
        checkpoint path, input size, and confidence/IoU thresholds from
        here; the registry still constructs adapters with no arguments.
        """
        self.model_config: ModelConfig | None = model
        self.benchmark_settings: BenchmarkSettings | None = benchmark

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
