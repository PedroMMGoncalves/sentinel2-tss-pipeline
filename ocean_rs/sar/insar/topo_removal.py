"""
Topographic phase removal for InSAR.

Simulates and removes the topographic phase contribution using a DEM
and orbit geometry, leaving only the deformation signal.

Topographic phase formula:
    φ_topo = (4π/λ) · B_perp · h / (R · sin(θ))

where:
    λ = radar wavelength (m)
    B_perp = perpendicular baseline (m)
    h = terrain height (m)
    R = slant range (m)
    θ = incidence angle (rad)

References:
    Rosen, P.A. et al. (2000). Synthetic Aperture Radar Interferometry.
    Proceedings of the IEEE, 88(3), 333-382.
"""

import logging
import os
from typing import List, Optional

import numpy as np

from ..core.data_models import Interferogram, OrbitStateVector, GeoTransform

logger = logging.getLogger('ocean_rs')

# Earth radius (approximate, for slant range computation)
EARTH_RADIUS_M = 6371000.0

# Default satellite altitude for SAR sensors
DEFAULT_SAT_ALTITUDE_M = 700000.0  # ~700 km for S1/NISAR


def remove_topographic_phase(
    interferogram: Interferogram,
    primary_orbits: List[OrbitStateVector],
    secondary_orbits: List[OrbitStateVector],
    dem_path: str = "auto",
    scene_center_lat: float = 0.0,
    scene_center_lon: float = 0.0,
) -> Interferogram:
    """Remove topographic phase contribution from interferogram.

    Steps:
        1. Load DEM (auto-download SRTM if dem_path="auto")
        2. Compute slant range and incidence angle from orbit
        3. Simulate topographic phase: φ_topo = (4π/λ) · B_perp · h / (R·sin(θ))
        4. Subtract from unwrapped phase (or wrapped via complex subtraction)

    Args:
        interferogram: Input interferogram (wrapped or with unwrapped phase).
        primary_orbits: Primary image orbit state vectors.
        secondary_orbits: Secondary image orbit state vectors.
        dem_path: Path to DEM GeoTIFF, or "auto" for SRTM download.
        scene_center_lat: Scene center latitude (for SRTM tile selection).
        scene_center_lon: Scene center longitude (for SRTM tile selection).

    Returns:
        Interferogram with topographic phase removed.
    """
    from .baseline import compute_baseline, compute_incidence_angle

    rows, cols = interferogram.phase.shape
    logger.info(f"Removing topographic phase: {rows}×{cols} pixels")

    # Compute baseline
    scene_time = interferogram.metadata.get('primary_time', '')
    b_perp, b_par = compute_baseline(
        primary_orbits, secondary_orbits, scene_time,
        scene_center_lat, scene_center_lon
    )

    if abs(b_perp) < 1.0:
        logger.info(
            f"Perpendicular baseline is very small ({b_perp:.1f}m). "
            f"Topographic phase contribution is negligible."
        )
        return interferogram

    # Load DEM
    dem = _load_dem(dem_path, interferogram.geo, rows, cols,
                     scene_center_lat, scene_center_lon)

    # Compute incidence angle
    if interferogram.incidence_angle is not None:
        incidence = interferogram.incidence_angle
    else:
        incidence = compute_incidence_angle(
            primary_orbits, scene_time, interferogram.geo
        )

    # Compute slant range (approximate)
    slant_range = _compute_slant_range(incidence, primary_orbits, scene_time)

    # Simulate topographic phase
    wavelength = interferogram.wavelength_m
    if wavelength <= 0:
        raise ValueError("Interferogram wavelength must be positive")

    sin_theta = np.sin(incidence)
    # Mask near-nadir pixels (incidence < 5 degrees) as invalid
    near_nadir_mask = sin_theta < 0.087  # sin(5 degrees)
    sin_theta = np.where(near_nadir_mask, np.nan, sin_theta)

    phi_topo = (4 * np.pi / wavelength) * b_perp * dem / (slant_range * sin_theta)
    phi_topo = phi_topo.astype(np.float32)

    logger.info(
        f"Topographic phase range: [{np.nanmin(phi_topo):.1f}, {np.nanmax(phi_topo):.1f}] rad"
    )

    # Remove topographic phase
    if interferogram.unwrapped_phase is not None:
        # Direct subtraction on unwrapped phase
        defo_unwrapped = interferogram.unwrapped_phase - phi_topo
        # np.angle(np.exp(1j*NaN)) returns 0.0, not NaN — preserve NaN mask
        nan_mask = np.isnan(defo_unwrapped)
        defo_wrapped = np.angle(np.exp(1j * np.nan_to_num(defo_unwrapped))).astype(np.float32)
        defo_wrapped[nan_mask] = np.nan

        return Interferogram(
            phase=defo_wrapped,
            coherence=interferogram.coherence,
            unwrapped_phase=defo_unwrapped,
            geo=interferogram.geo,
            wavelength_m=interferogram.wavelength_m,
            temporal_baseline_days=interferogram.temporal_baseline_days,
            perpendicular_baseline_m=interferogram.perpendicular_baseline_m,
            incidence_angle=incidence,
            metadata={
                **interferogram.metadata,
                'topographic_phase_removed': True,
                'dem_source': dem_path,
            },
        )
    else:
        # Complex subtraction on wrapped phase
        # Preserve NaN mask: np.angle(np.exp(1j*NaN)) returns 0.0, not NaN
        phase_f64 = interferogram.phase.astype(np.float64)
        topo_f64 = phi_topo.astype(np.float64)
        nan_mask = np.isnan(phase_f64) | np.isnan(topo_f64)
        complex_ifg = np.exp(1j * np.nan_to_num(phase_f64))
        complex_topo = np.exp(1j * np.nan_to_num(topo_f64))
        defo_complex = complex_ifg * np.conj(complex_topo)
        defo_phase = np.angle(defo_complex).astype(np.float32)
        defo_phase[nan_mask] = np.nan

        return Interferogram(
            phase=defo_phase,
            coherence=interferogram.coherence,
            geo=interferogram.geo,
            wavelength_m=interferogram.wavelength_m,
            temporal_baseline_days=interferogram.temporal_baseline_days,
            perpendicular_baseline_m=interferogram.perpendicular_baseline_m,
            incidence_angle=incidence,
            metadata={
                **interferogram.metadata,
                'topographic_phase_removed': True,
                'dem_source': dem_path,
            },
        )


def _load_dem(
    dem_path: str,
    geo: GeoTransform,
    rows: int,
    cols: int,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Load DEM, resampled to match interferogram grid.

    If dem_path="auto", attempts to download SRTM tiles via GDAL /vsicurl/.
    Falls back to flat terrain (zeros) if DEM is unavailable.
    """
    if dem_path == "auto" or dem_path == "":
        dem = _download_srtm(center_lat, center_lon, rows, cols, geo)
        if dem is not None:
            return dem
        raise RuntimeError(
            f"SRTM DEM download failed for scene extent. "
            f"Provide a DEM file via dem_path parameter."
        )

    # Load user-provided DEM — no silent fallback (CLAUDE.md rule 11)
    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"User-provided DEM not found: {dem_path}")

    from osgeo import gdal

    dem_ds = gdal.Open(dem_path)
    if dem_ds is None:
        raise RuntimeError(f"GDAL failed to open user-provided DEM: {dem_path}")

    dem_data = dem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Resample to match interferogram grid using GDAL warp
    if dem_data.shape != (rows, cols):
        dem_data = _gdal_resample_dem(dem_path, geo, rows, cols)

    dem_ds = None
    return dem_data


def _download_srtm(
    center_lat: float,
    center_lon: float,
    rows: int,
    cols: int,
    geo: GeoTransform,
) -> Optional[np.ndarray]:
    """Download SRTM DEM tiles via GDAL /vsicurl/.

    Uses CGIAR SRTM v4.1 5x5 degree tiles with correct naming convention:
        srtm_XX_YY.zip where XX=column, YY=row in the CGIAR grid.
        Column: XX = int((lon + 180) / 5) + 1
        Row:    YY = int((60 - lat) / 5) + 1
    """
    try:
        from osgeo import gdal
    except ImportError:
        return None

    if abs(center_lat) < 0.01 and abs(center_lon) < 0.01:
        logger.warning("Scene center coordinates not available for SRTM download")
        return None

    # Set GDAL HTTP timeout for remote access
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '30')

    # CGIAR SRTM tile grid indices
    tile_col = int((center_lon + 180) / 5) + 1
    tile_row = int((60 - center_lat) / 5) + 1

    srtm_url = (
        f"/vsicurl/https://srtm.csi.cgiar.org/wp-content/uploads/files/"
        f"srtm_5x5/TIFF/srtm_{tile_col:02d}_{tile_row:02d}.zip"
    )

    logger.info(
        f"Attempting SRTM download for {center_lat:.1f}, {center_lon:.1f} "
        f"(tile srtm_{tile_col:02d}_{tile_row:02d})"
    )

    try:
        # Try zip archive via /vsizip/
        vsi_path = f"/vsizip/{srtm_url}/srtm_{tile_col:02d}_{tile_row:02d}.tif"
        ds = gdal.Open(vsi_path)
        if ds is None:
            # Try direct TIFF (some mirrors serve uncompressed)
            direct_url = (
                f"/vsicurl/https://srtm.csi.cgiar.org/wp-content/uploads/files/"
                f"srtm_5x5/TIFF/srtm_{tile_col:02d}_{tile_row:02d}.tif"
            )
            ds = gdal.Open(direct_url)

        if ds is None:
            logger.warning("SRTM tile not found via GDAL /vsicurl/")
            return None

        # Resample to match interferogram grid using GDAL Warp
        dem_data = _gdal_resample_from_dataset(ds, geo, rows, cols)
        ds = None

        logger.info(
            f"SRTM loaded: elevation range "
            f"[{dem_data.min():.0f}, {dem_data.max():.0f}]m"
        )
        return dem_data

    except Exception as e:
        logger.warning(f"SRTM download failed: {e}")
        return None


def _gdal_resample_dem(
    dem_path: str,
    geo: GeoTransform,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Resample a DEM file to match interferogram extent using GDAL Warp.

    Args:
        dem_path: Path to the DEM GeoTIFF.
        geo: Target GeoTransform (defines extent and resolution).
        rows: Target number of rows.
        cols: Target number of columns.

    Returns:
        Resampled DEM array as float32.
    """
    from osgeo import gdal

    ds = gdal.Open(dem_path)
    if ds is None:
        raise RuntimeError(f"GDAL failed to open DEM for resampling: {dem_path}")

    result = _gdal_resample_from_dataset(ds, geo, rows, cols)
    ds = None
    return result


def _gdal_resample_from_dataset(
    src_ds,
    geo: GeoTransform,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Resample an open GDAL dataset to match target extent and resolution.

    Uses gdal.Warp with bilinear resampling to properly handle geographic
    extents and coordinate alignment, unlike scipy.ndimage.zoom which
    ignores geographic metadata.

    Args:
        src_ds: Open GDAL dataset.
        geo: Target GeoTransform (defines extent and resolution).
        rows: Target number of rows.
        cols: Target number of columns.

    Returns:
        Resampled array as float32.
    """
    from osgeo import gdal

    # Compute target extent from GeoTransform
    x_min = geo.origin_x
    y_max = geo.origin_y
    x_max = x_min + cols * geo.pixel_size_x
    y_min = y_max + rows * geo.pixel_size_y  # pixel_size_y is negative

    # Warp to in-memory dataset matching target grid
    warp_options = gdal.WarpOptions(
        format='MEM',
        outputBounds=(x_min, y_min, x_max, y_max),
        width=cols,
        height=rows,
        resampleAlg=gdal.GRA_Bilinear,
    )

    warped_ds = gdal.Warp('', src_ds, options=warp_options)
    if warped_ds is None:
        raise RuntimeError("GDAL Warp failed during DEM resampling")

    dem_data = warped_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    warped_ds = None

    return dem_data


def _compute_slant_range(
    incidence_angle: np.ndarray,
    orbit_vectors: Optional[List[OrbitStateVector]] = None,
    scene_center_time: str = "",
) -> np.ndarray:
    """Compute slant range from incidence angle using spherical Earth model.

    Uses the exact spherical Earth formula:
        R = sqrt((R_e + h)^2 - R_e^2 * cos^2(theta)) - R_e * sin(theta)

    where R_e is Earth radius, h is satellite altitude, and theta is the
    incidence angle. This avoids the 5-10% error of the flat-Earth
    approximation R = h / cos(theta).

    Satellite altitude is computed from orbit state vectors when available,
    using the WGS84 ellipsoidal radius at the satellite's geodetic latitude.
    Falls back to 700 km default with a warning.
    """
    from .baseline import _interpolate_orbit

    R_e = EARTH_RADIUS_M
    h = DEFAULT_SAT_ALTITUDE_M  # default fallback

    # Attempt to compute altitude from orbit state vectors
    if orbit_vectors and len(orbit_vectors) >= 2:
        pos = _interpolate_orbit(orbit_vectors, scene_center_time)
        if pos is not None:
            sat_pos = np.array(pos[:3])
            # Compute geocentric latitude from ECEF position
            sat_lat_gc = np.arctan2(
                sat_pos[2], np.sqrt(sat_pos[0]**2 + sat_pos[1]**2)
            )
            # WGS84 ellipsoidal radius at this latitude
            a = 6378137.0  # WGS84 semi-major
            b = 6356752.314245  # WGS84 semi-minor
            cos_lat = np.cos(sat_lat_gc)
            sin_lat = np.sin(sat_lat_gc)
            R_local = np.sqrt(
                ((a**2 * cos_lat)**2 + (b**2 * sin_lat)**2) /
                ((a * cos_lat)**2 + (b * sin_lat)**2)
            )
            h = float(np.linalg.norm(sat_pos) - R_local)
            logger.info(f"Satellite altitude from orbit: {h/1000:.1f} km")
        else:
            logger.warning(
                f"Orbit interpolation failed. Using default satellite "
                f"altitude of {DEFAULT_SAT_ALTITUDE_M/1000:.0f} km."
            )
    else:
        logger.warning(
            f"No orbit state vectors available. Using default satellite "
            f"altitude of {DEFAULT_SAT_ALTITUDE_M/1000:.0f} km."
        )

    theta = incidence_angle

    slant_range = (
        np.sqrt((R_e + h) ** 2 - R_e ** 2 * np.cos(theta) ** 2)
        - R_e * np.sin(theta)
    )
    return slant_range.astype(np.float64)
