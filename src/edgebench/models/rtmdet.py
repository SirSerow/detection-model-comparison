"""RTMDet-Tiny adapter stub."""

from edgebench.models._stub import StubDetector


class RTMDetTinyAdapter(StubDetector):
    @property
    def name(self) -> str:
        return "rtmdet_tiny"
