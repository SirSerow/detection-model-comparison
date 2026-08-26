"""Shared decode → letterbox 640×640 → normalize pipeline.

Model-specific mean/std and RGB/BGR differences belong on the detector
adapter and must be documented when implemented.
"""

from __future__ import annotations

from typing import Any


def letterbox(image: Any, size: tuple[int, int] = (640, 640)) -> Any:
    raise NotImplementedError("Shared letterbox preprocess is not implemented yet")
