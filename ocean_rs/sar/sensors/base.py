"""
Base sensor adapter for SAR preprocessing.

All sensor adapters inherit from SensorAdapter and produce OceanImage instances.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..core.data_models import OceanImage


class SensorAdapter(ABC):
    """Abstract base class for SAR sensor adapters."""

    @property
    @abstractmethod
    def sensor_name(self) -> str:
        """Human-readable sensor name."""
        ...

    @abstractmethod
    def preprocess(self, input_path: Path, output_dir: Path,
                   snap_gpt_path: Optional[str] = None) -> OceanImage:
        """Preprocess raw SAR data to calibrated, geocoded OceanImage."""
        ...

    @abstractmethod
    def can_process(self, input_path: Path) -> bool:
        """Check if this adapter can process the given input."""
        ...
