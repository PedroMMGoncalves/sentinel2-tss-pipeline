"""
Base sensor adapter for SAR preprocessing.

All sensor adapters inherit from SensorAdapter and produce OceanImage instances
(for bathymetry) or SLCImage instances (for InSAR).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..core.data_models import OceanImage, SLCImage


class SensorAdapter(ABC):
    """Abstract base class for SAR sensor adapters.

    Subclasses must implement:
        - sensor_name: human-readable name
        - preprocess(): raw SAR → calibrated OceanImage (for bathymetry)
        - can_process(): input file detection

    Optional overrides for InSAR:
        - read_slc(): raw SAR → complex SLCImage
        - deburst_slc(): TOPS deburst (Sentinel-1 IW only)
    """

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

    def read_slc(self, input_path: Path, output_dir: Path,
                 polarization: str = "VV") -> SLCImage:
        """Read SLC data for InSAR processing.

        Override in subclasses that support InSAR. For Sentinel-1 IW,
        this automatically debursts TOPS data before reading.

        Args:
            input_path: Path to SLC product (.SAFE, .zip, .h5, CEOS dir)
            output_dir: Working directory for intermediate files
            polarization: Polarization channel (default: VV)

        Returns:
            SLCImage with complex data and orbit metadata.

        Raises:
            NotImplementedError: If sensor does not support SLC reading.
        """
        raise NotImplementedError(
            f"{self.sensor_name} does not support SLC reading"
        )

    def deburst_slc(self, input_path: Path, output_dir: Path,
                    swaths: Optional[list] = None) -> Path:
        """Deburst TOPS SLC data (Sentinel-1 IW only).

        Args:
            input_path: Path to SLC product
            output_dir: Working directory for debursted output
            swaths: List of swaths to process (e.g. ['IW1', 'IW2', 'IW3']).
                    None = all swaths.

        Returns:
            Path to debursted product.

        Raises:
            NotImplementedError: If sensor does not use TOPS acquisition.
        """
        raise NotImplementedError(
            f"{self.sensor_name} does not require debursting"
        )
