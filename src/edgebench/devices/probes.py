"""Optional automatic hardware discovery.

Manual device YAML is the reproducibility source of truth. Probing is
intentionally unimplemented in the skeleton.
"""

from __future__ import annotations

from typing import Any


class DeviceProbe:
    """Discover CPU, CUDA, and runtime versions from the host."""

    def discover(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Automatic device probing is not implemented; use configs/devices/*.yaml"
        )
