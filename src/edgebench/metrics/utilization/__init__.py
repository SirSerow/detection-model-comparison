"""Utilization collectors."""

from edgebench.metrics.utilization.cpu import CPUUtilizationCollector
from edgebench.metrics.utilization.nvidia import NvidiaUtilizationCollector

__all__ = ["CPUUtilizationCollector", "NvidiaUtilizationCollector"]
