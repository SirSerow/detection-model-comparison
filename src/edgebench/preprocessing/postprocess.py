"""Shared helpers to map boxes back to original-image coordinates."""

from __future__ import annotations

from typing import Any


def rescale_boxes(boxes: Any, metadata: Any) -> Any:
    raise NotImplementedError("Box rescaling is not implemented yet")
