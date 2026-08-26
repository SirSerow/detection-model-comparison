"""Dataset adapters. No model- or runtime-specific logic belongs here."""

from edgebench.dataset_adapters.base import DatasetAdapter
from edgebench.dataset_adapters.coco import CocoDataset
from edgebench.dataset_adapters.custom import CustomIndustrialDataset

__all__ = ["CocoDataset", "CustomIndustrialDataset", "DatasetAdapter"]
