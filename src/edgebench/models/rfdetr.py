"""RF-DETR-Nano adapter stub."""

from edgebench.models._stub import StubDetector


class RFDETRNanoAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "rfdetr_nano"
