"""YOLO26n adapter stub."""

from edgebench.models._stub import StubDetector


class YOLO26nAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "yolo26n"
