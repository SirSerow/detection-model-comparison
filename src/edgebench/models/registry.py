"""Map stable detector names to adapter classes."""

from __future__ import annotations

from edgebench.models.base import DetectorAdapter
from edgebench.models.damo_yolo import DAMOYOLOTAdapter
from edgebench.models.picodet import PicoDetSAdapter
from edgebench.models.rfdetr import RFDETRNanoAdapter
from edgebench.models.rtdetr import RTDETRv2SAdapter
from edgebench.models.rtmdet import RTMDetTinyAdapter
from edgebench.models.yolo26 import YOLO26nAdapter
from edgebench.models.yolox import YOLOXTinyAdapter

DETECTORS: dict[str, type[DetectorAdapter]] = {
    "yolox_tiny": YOLOXTinyAdapter,
    "yolo26n": YOLO26nAdapter,
    "rtmdet_tiny": RTMDetTinyAdapter,
    "damo_yolo_t": DAMOYOLOTAdapter,
    "picodet_s": PicoDetSAdapter,
    "rtdetrv2_s": RTDETRv2SAdapter,
    "rfdetr_nano": RFDETRNanoAdapter,
}


class DetectorRegistry:
    def names(self) -> list[str]:
        return list(DETECTORS)

    def get(self, name: str) -> DetectorAdapter:
        try:
            return DETECTORS[name]()
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(f"Unknown detector '{name}'. Known: {known}") from exc


def list_detectors() -> list[str]:
    return DetectorRegistry().names()


def get_detector(name: str) -> DetectorAdapter:
    return DetectorRegistry().get(name)
