"""RT-DETRv2-S adapter stub."""

from edgebench.models._stub import StubDetector


class RTDETRv2SAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "rtdetrv2_s"
