"""DAMO-YOLO-T adapter stub."""

from edgebench.models._stub import StubDetector


class DAMOYOLOTAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "damo_yolo_t"
