"""
NISAR sensor adapter.

Reads NASA-ISRO NISAR L1 SLC products in HDF5 format.
NISAR operates in L-band (1257.5 MHz, λ=0.2384m) and S-band (3.2 GHz, λ=0.0938m).

Data is distributed via ASF DAAC in HDF5 format with the structure:
    /science/LSAR/{RSLC,SLC}/swaths/frequency{A,B}/
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.data_models import (
    OceanImage, SLCImage, ImageType, GeoTransform, OrbitStateVector,
)
from .base import SensorAdapter
from ocean_rs.shared.raster_io import check_memory_for_array

logger = logging.getLogger('ocean_rs')


# NISAR radar wavelengths (m) — read from metadata when available
NISAR_L_BAND_WAVELENGTH = 0.2384   # 1257.5 MHz
NISAR_S_BAND_WAVELENGTH = 0.0938   # 3.2 GHz


class NISARAdapter(SensorAdapter):
    """Adapter for NASA-ISRO NISAR L1 SLC products."""

    def __init__(self, pixel_spacing_m: float = 10.0):
        self.pixel_spacing_m = pixel_spacing_m

    @property
    def sensor_name(self) -> str:
        return "NISAR"

    def can_process(self, input_path: Path) -> bool:
        """Check if file is a NISAR product."""
        name = input_path.name.upper()
        return 'NISAR' in name and name.endswith('.H5')

    def preprocess(self, input_path: Path, output_dir: Path,
                   snap_gpt_path: Optional[str] = None) -> OceanImage:
        """Extract amplitude image from NISAR SLC for bathymetry.

        Reads the complex SLC and returns the amplitude (|z|) as an
        OceanImage suitable for the FFT bathymetry pipeline.
        """
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
        """Read NISAR L1 SLC from HDF5.

        NISAR HDF5 structure:
            /science/LSAR/SLC/swaths/frequencyA/HH (complex data)
            /science/LSAR/identification/radarWavelength
            /science/LSAR/SLC/metadata/orbit/

        Args:
            input_path: Path to NISAR .h5 file
            output_dir: Working directory (unused for NISAR)
            polarization: Polarization channel (default: HH for NISAR)

        Returns:
            SLCImage with complex data and orbit metadata.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for NISAR data.\n"
                "Install with: conda install -c conda-forge h5py"
            )

        input_path = Path(input_path)
        logger.info(f"Reading NISAR SLC: {input_path.name}")

        with h5py.File(str(input_path), 'r') as f:
            # Determine which SAR band (LSAR or SSAR)
            sar_band = self._detect_sar_band(f)

            # Try RSLC (standard L1 product) first, then SLC
            for product_type in ['RSLC', 'SLC']:
                swaths_path = f'/science/{sar_band}/{product_type}/swaths/frequencyA'
                if swaths_path in f:
                    break
            else:
                raise FileNotFoundError(
                    f"No {sar_band} SLC/RSLC group found in {input_path.name}"
                )

            swath_group = f[swaths_path]
            if polarization in swath_group:
                # M15: Check memory before reading large SLC
                ds_shape = swath_group[polarization].shape
                check_memory_for_array(
                    ds_shape[0], ds_shape[1] if len(ds_shape) > 1 else 1,
                    bytes_per_pixel=8, description="NISAR SLC"
                )
                complex_data = swath_group[polarization][()].astype(np.complex64)
            else:
                available = list(swath_group.keys())
                pol_keys = [k for k in available if len(k) == 2 and k.isalpha()]
                if pol_keys:
                    logger.warning(
                        f"Polarization '{polarization}' not found. "
                        f"Using '{pol_keys[0]}'. Available: {pol_keys}"
                    )
                    # M15: Check memory before reading large SLC
                    ds_shape = swath_group[pol_keys[0]].shape
                    check_memory_for_array(
                        ds_shape[0], ds_shape[1] if len(ds_shape) > 1 else 1,
                        bytes_per_pixel=8, description="NISAR SLC"
                    )
                    complex_data = swath_group[pol_keys[0]][()].astype(np.complex64)
                else:
                    raise FileNotFoundError(
                        f"No polarization data in {swaths_path}. "
                        f"Available keys: {available}"
                    )

            # Read wavelength from metadata
            wavelength = self._read_wavelength(f, sar_band)

            # Read geolocation
            geo = self._read_geotransform(f, sar_band, complex_data.shape)

            # Read pixel spacing
            spacing_range, spacing_azimuth = self._read_pixel_spacing(f, sar_band)

            # Read orbit state vectors
            orbit_vectors = self._read_orbit(f, sar_band)

            # Read acquisition time
            acq_time = self._read_acquisition_time(f, sar_band)

        return SLCImage(
            data=complex_data,
            geo=geo,
            metadata={
                'sensor': 'NISAR',
                'sar_band': sar_band,
                'beam_mode': sar_band,
                'acquisition_time': acq_time,
                'source_file': str(input_path),
                'orbit_state_vectors': orbit_vectors,
            },
            wavelength_m=wavelength,
            pixel_spacing_range=spacing_range,
            pixel_spacing_azimuth=spacing_azimuth,
            is_debursted=True,  # NISAR L1 SLC is already focused
        )

    def _detect_sar_band(self, f) -> str:
        """Detect which SAR band (LSAR or SSAR) is available."""
        if '/science/LSAR' in f:
            return 'LSAR'
        elif '/science/SSAR' in f:
            return 'SSAR'
        else:
            raise FileNotFoundError(
                "Neither LSAR nor SSAR group found in NISAR HDF5"
            )

    def _read_wavelength(self, f, sar_band: str) -> float:
        """Read radar wavelength from NISAR metadata."""
        wavelength_paths = [
            f'/science/{sar_band}/identification/radarWavelength',
            f'/science/{sar_band}/SLC/metadata/radarWavelength',
        ]
        for path in wavelength_paths:
            if path in f:
                return float(f[path][()])

        # Fallback based on band
        fallback = (NISAR_L_BAND_WAVELENGTH if sar_band == 'LSAR'
                    else NISAR_S_BAND_WAVELENGTH)
        logger.warning(
            f"Radar wavelength not found in metadata. "
            f"Using fallback for {sar_band}: {fallback}m"
        )
        return fallback

    def _read_geotransform(self, f, sar_band: str,
                            shape: tuple) -> GeoTransform:
        """Read geotransform from NISAR HDF5 geolocation grid."""
        for product_type in ['RSLC', 'SLC']:
            geo_path = f'/science/{sar_band}/{product_type}/metadata/geolocationGrid'
            if geo_path in f:
                try:
                    x = f[f'{geo_path}/coordinateX'][:]
                    y = f[f'{geo_path}/coordinateY'][:]
                    # Compute approximate affine from corners
                    origin_x = float(x[0, 0])
                    origin_y = float(y[0, 0])
                    # Scale pixel sizes to image dimensions, not geolocation grid dimensions
                    pixel_size_x = float(x[0, -1] - x[0, 0]) / max(shape[1] - 1, 1)
                    pixel_size_y = float(y[-1, 0] - y[0, 0]) / max(shape[0] - 1, 1)
                    epsg_ds = f'{geo_path}/epsg'
                    crs_wkt = ''
                    if epsg_ds in f:
                        from osgeo import osr
                        srs = osr.SpatialReference()
                        srs.ImportFromEPSG(int(f[epsg_ds][()]))
                        crs_wkt = srs.ExportToWkt()
                    return GeoTransform(
                        origin_x=origin_x, origin_y=origin_y,
                        pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y,
                        crs_wkt=crs_wkt,
                        rows=shape[0], cols=shape[1],
                    )
                except (KeyError, ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to read NISAR geolocation grid: {e}")

        # Fallback to radar coordinates
        logger.warning("No geolocation grid found in NISAR product. Using radar coordinates.")
        return GeoTransform(
            origin_x=0.0, origin_y=0.0,
            pixel_size_x=1.0, pixel_size_y=-1.0,
            crs_wkt='',
            rows=shape[0], cols=shape[1],
        )

    def _read_pixel_spacing(self, f, sar_band: str) -> tuple:
        """Read range and azimuth pixel spacing."""
        range_spacing = 1.0
        azimuth_spacing = 1.0

        for product_type in ['RSLC', 'SLC']:
            spacing_paths = [
                (f'/science/{sar_band}/{product_type}/swaths/frequencyA/slantRangeSpacing', 'range'),
                (f'/science/{sar_band}/{product_type}/swaths/frequencyA/sceneCenterAlongTrackSpacing', 'azimuth'),
            ]
            for path, direction in spacing_paths:
                if path in f:
                    value = float(f[path][()])
                    if direction == 'range':
                        range_spacing = value
                    else:
                        azimuth_spacing = value

        return range_spacing, azimuth_spacing

    def _read_orbit(self, f, sar_band: str) -> list:
        """Read orbit state vectors from NISAR metadata."""
        orbit_vectors = []
        orbit_path = None
        for product_type in ['RSLC', 'SLC']:
            candidate = f'/science/{sar_band}/{product_type}/metadata/orbit'
            if candidate in f:
                orbit_path = candidate
                break

        if orbit_path is None:
            logger.warning("No orbit data found in NISAR HDF5")
            return orbit_vectors

        try:
            orbit_group = f[orbit_path]
            times = orbit_group.get('time', None)
            positions = orbit_group.get('position', None)
            velocities = orbit_group.get('velocity', None)

            if times is not None and positions is not None and velocities is not None:
                for i in range(len(times)):
                    t = times[i]
                    if hasattr(t, 'decode'):
                        t = t.decode()
                    pos = positions[i]
                    vel = velocities[i]
                    orbit_vectors.append(OrbitStateVector(
                        time_utc=str(t),
                        x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                    ))
                logger.info(f"Read {len(orbit_vectors)} orbit state vectors from NISAR")
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.warning(f"Failed to read NISAR orbit data: {e}")

        return orbit_vectors

    def _read_acquisition_time(self, f, sar_band: str) -> str:
        """Read acquisition time from NISAR metadata."""
        time_path = f'/science/{sar_band}/identification/zeroDopplerStartTime'
        if time_path in f:
            val = f[time_path][()]
            if hasattr(val, 'decode'):
                val = val.decode()
            return str(val)
        return ""
