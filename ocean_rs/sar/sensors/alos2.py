"""
ALOS-2 PALSAR-2 sensor adapter.

Reads JAXA ALOS-2 PALSAR-2 SLC products in CEOS format via GDAL.
ALOS-2 operates in L-band (1236.5 MHz, λ=0.2424m).

Supported modes:
    - FBS (Fine Beam Single polarization)
    - FBD (Fine Beam Dual polarization)
    - PLR (Polarimetric / quad-pol)
    - SM  (Stripmap)
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.data_models import (
    OceanImage, SLCImage, ImageType, GeoTransform, OrbitStateVector,
)
from .base import SensorAdapter

logger = logging.getLogger('ocean_rs')

# ALOS-2 PALSAR-2 L-band wavelength (m) — 1236.5 MHz
ALOS2_WAVELENGTH_M = 0.2424


class ALOS2Adapter(SensorAdapter):
    """Adapter for JAXA ALOS-2 PALSAR-2 SLC products."""

    def __init__(self, pixel_spacing_m: float = 10.0):
        self.pixel_spacing_m = pixel_spacing_m

    @property
    def sensor_name(self) -> str:
        return "ALOS-2"

    def can_process(self, input_path: Path) -> bool:
        """Check if file is an ALOS-2 PALSAR-2 product.

        ALOS-2 CEOS products are typically distributed as directories
        containing IMG-* and LED-* files, or as zip archives.
        """
        name = input_path.name.upper()
        # ALOS-2 filename patterns
        if 'ALOS2' in name or 'PALSAR' in name:
            return True
        # CEOS leader file
        if name.startswith('LED-ALOS2'):
            return True
        return False

    def preprocess(self, input_path: Path, output_dir: Path,
                   snap_gpt_path: Optional[str] = None) -> OceanImage:
        """Extract amplitude image from ALOS-2 SLC for bathymetry."""
        slc = self.read_slc(input_path, output_dir)
        amplitude = np.abs(slc.data).astype(np.float32)

        return OceanImage(
            data=amplitude,
            image_type=ImageType.SIGMA0,
            geo=slc.geo,
            metadata=slc.metadata,
            pixel_spacing_m=slc.pixel_spacing_range,
        )

    def read_slc(self, input_path: Path, output_dir: Path,
                 polarization: str = "HH") -> SLCImage:
        """Read ALOS-2 PALSAR-2 SLC via GDAL.

        GDAL reads CEOS format natively using the SAR_CEOS driver.
        The data directory should contain IMG-* (image) and LED-* (leader) files.

        Args:
            input_path: Path to ALOS-2 product directory or leader file
            output_dir: Working directory (unused for ALOS-2)
            polarization: Polarization channel (default: HH)

        Returns:
            SLCImage with complex data and orbit metadata.
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise ImportError("GDAL is required for ALOS-2 data reading")

        input_path = Path(input_path)
        logger.info(f"Reading ALOS-2 SLC: {input_path.name}")

        # Find the image file to open
        img_path = self._find_image_file(input_path, polarization)
        if img_path is None:
            raise FileNotFoundError(
                f"No ALOS-2 image file found for polarization '{polarization}' "
                f"in: {input_path}"
            )

        ds = gdal.Open(str(img_path))
        if ds is None:
            raise RuntimeError(f"GDAL failed to open ALOS-2 data: {img_path}")

        try:
            # Read complex data
            band = ds.GetRasterBand(1)
            data_type = band.DataType

            # Check memory before loading large SLC
            from ocean_rs.shared.raster_io import check_memory_for_array
            check_memory_for_array(
                ds.RasterYSize, ds.RasterXSize,
                bytes_per_pixel=8, description="ALOS-2 SLC"
            )

            # CEOS SLC is typically complex int16 or complex float32
            raw_data = band.ReadAsArray()
            if np.issubdtype(raw_data.dtype, np.complexfloating):
                complex_data = raw_data.astype(np.complex64)
            else:
                # Real-valued: treat as amplitude (no phase info)
                complex_data = raw_data.astype(np.complex64)
                logger.warning(
                    "ALOS-2 data is not complex. Phase information unavailable."
                )

            gt = ds.GetGeoTransform()
            crs_wkt = ds.GetProjection() or ''
            rows, cols = ds.RasterYSize, ds.RasterXSize

            # Read metadata
            md = ds.GetMetadata()
        finally:
            ds = None

        geo = GeoTransform(
            origin_x=gt[0], origin_y=gt[3],
            pixel_size_x=gt[1], pixel_size_y=gt[5],
            crs_wkt=crs_wkt, rows=rows, cols=cols,
        )

        # Parse pixel spacing from metadata or filename
        spacing_range, spacing_azimuth = self._parse_pixel_spacing(
            md, input_path
        )

        # Parse orbit from leader file
        orbit_vectors = self._parse_orbit_from_leader(input_path)

        # Parse acquisition time
        acq_time = self._parse_acquisition_time(md, input_path)

        # Detect beam mode
        beam_mode = self._detect_beam_mode(input_path)

        return SLCImage(
            data=complex_data,
            geo=geo,
            metadata={
                'sensor': 'ALOS-2',
                'beam_mode': beam_mode,
                'acquisition_time': acq_time,
                'source_file': str(input_path),
                'orbit_state_vectors': orbit_vectors,
                'gdal_metadata': md,
            },
            wavelength_m=ALOS2_WAVELENGTH_M,
            pixel_spacing_range=spacing_range,
            pixel_spacing_azimuth=spacing_azimuth,
            is_debursted=True,  # ALOS-2 SLC is already focused
        )

    def _find_image_file(self, input_path: Path,
                          polarization: str) -> Optional[Path]:
        """Find the image file within an ALOS-2 product.

        Searches for IMG-{pol}-* files in the product directory.
        Falls back to any IMG-* file if specific polarization not found.
        """
        input_path = Path(input_path)

        # If input is a file (leader or image), find the directory
        if input_path.is_file():
            search_dir = input_path.parent
        else:
            search_dir = input_path

        # Try exact polarization match
        pol_upper = polarization.upper()
        img_files = sorted(search_dir.glob(f"IMG-{pol_upper}-*"))
        if img_files:
            return img_files[0]

        # Try any image file
        img_files = sorted(search_dir.glob("IMG-*"))
        if img_files:
            logger.warning(
                f"Polarization '{polarization}' not found. "
                f"Using: {img_files[0].name}"
            )
            return img_files[0]

        # Try GDAL-readable files in directory
        for ext in ('*.tif', '*.tiff', '*.img'):
            files = sorted(search_dir.glob(ext))
            if files:
                return files[0]

        return None

    def _parse_pixel_spacing(self, metadata: dict,
                              input_path: Path) -> tuple:
        """Parse pixel spacing from GDAL metadata or defaults.

        ALOS-2 typical spacings:
            FBS: ~1.4m range × ~2.1m azimuth
            FBD: ~4.3m range × ~3.2m azimuth
            PLR: ~5.3m range × ~3.6m azimuth
        """
        range_spacing = 4.3   # Default FBD
        azimuth_spacing = 3.2

        # Try GDAL metadata keys
        for key in ('RANGE_PIXEL_SPACING', 'PIXEL_SPACING_RANGE'):
            if key in metadata:
                try:
                    range_spacing = float(metadata[key])
                except (ValueError, TypeError):
                    pass

        for key in ('AZIMUTH_PIXEL_SPACING', 'PIXEL_SPACING_AZIMUTH'):
            if key in metadata:
                try:
                    azimuth_spacing = float(metadata[key])
                except (ValueError, TypeError):
                    pass

        return range_spacing, azimuth_spacing

    def _parse_orbit_from_leader(self, input_path: Path) -> list:
        """Parse orbit state vectors from ALOS-2 CEOS leader file.

        The LED-* file contains orbit information in the platform
        position data record. This is a binary format.

        Returns:
            List of OrbitStateVector objects, or empty list.
        """
        input_path = Path(input_path)

        # Find leader file
        if input_path.is_file():
            search_dir = input_path.parent
        else:
            search_dir = input_path

        leader_files = sorted(search_dir.glob("LED-*"))
        if not leader_files:
            logger.warning("No ALOS-2 leader file (LED-*) found for orbit data")
            return []

        # CEOS leader file orbit parsing would require binary format reading.
        # For now, return empty — orbit will be read from external source.
        logger.warning(
            f"ALOS-2 leader file found: {leader_files[0].name}. "
            f"Orbit parsing from CEOS binary is not yet implemented. "
            f"Returning empty orbit state vectors — baseline computation "
            f"will require an external orbit source."
        )
        return []

    def _parse_acquisition_time(self, metadata: dict,
                                 input_path: Path) -> str:
        """Parse acquisition time from metadata or filename."""
        # Try GDAL metadata
        for key in ('ACQUISITION_START_TIME', 'SCENE_CENTER_TIME'):
            if key in metadata:
                return metadata[key]

        # Try filename pattern: first YYYYMMDD, then ALOS2XXXXXXX-YYMMDD-...
        name = input_path.name
        # Prefer 8-digit YYYYMMDD date
        match = re.search(r'(\d{8})', name)
        if match:
            raw = match.group(1)
            year = int(raw[0:4])
            month = int(raw[4:6])
            day = int(raw[6:8])
            if 1990 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T00:00:00Z"

        # Fall back to hyphen-delimited 6-digit YYMMDD
        match = re.search(r'-(\d{6})-', name)
        if match:
            raw = match.group(1)
            return f"20{raw[0:2]}-{raw[2:4]}-{raw[4:6]}T00:00:00Z"

        return ""

    def _detect_beam_mode(self, input_path: Path) -> str:
        """Detect ALOS-2 beam mode from filename or directory name."""
        name = str(input_path).upper()
        for mode in ('FBS', 'FBD', 'PLR', 'SM1', 'SM2', 'SM3'):
            if mode in name:
                return mode
        return "FBD"  # Default
