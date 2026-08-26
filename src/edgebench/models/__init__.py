"""Detector adapters and name registry."""

from edgebench.models.base import DetectorAdapter
from edgebench.models.registry import DetectorRegistry, get_detector, list_detectors

__all__ = ["DetectorAdapter", "DetectorRegistry", "get_detector", "list_detectors"]
