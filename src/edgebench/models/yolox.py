"""YOLOX-Tiny adapter stub."""

from edgebench.models._stub import StubDetector


class YOLOXTinyAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "yolox_tiny"
