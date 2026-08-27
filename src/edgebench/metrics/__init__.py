"""Independent metric collectors."""

from edgebench.metrics.base import MetricCollector
from edgebench.metrics.registry import collectors_for, get_collector

__all__ = ["MetricCollector", "collectors_for", "get_collector"]
