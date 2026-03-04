"""
Geocoding for InSAR products.

Transforms InSAR results from radar coordinates to geographic coordinates
using GDAL gdalwarp with GCPs (Ground Control Points).

This is a practical approach that trades some geometric accuracy for
simplicity (compared to full Range-Doppler terrain correction). Suitable
for research-grade processing.

References:
    Small, D. & Schubert, A. (2008). Guide to ASAR Geocoding.
    European Space Agency (ESA).
"""

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np

from ..core.data_models import GeoTransform

logger = logging.getLogger('ocean_rs')


def geocode(
    data: np.ndarray,
    geo: GeoTransform,
    output_spacing_m: float = 30.0,
    output_crs: str = "EPSG:4326",
    dem_path: Optional[str] = None,
) -> Tuple[np.ndarray, GeoTransform]:
    """Geocode radar-coordinates data to geographic coordinates.

    Uses GDAL to create an in-memory raster with the data's existing
    georeference and reproject to the target CRS.

    For data already in a projected/geographic CRS (e.g., after SNAP TC),
    this simply reprojects. For data in radar coordinates (no CRS),
    a simple affine transformation is applied.

    Args:
        data: 2D array to geocode.
        geo: Current geotransform (may be radar or geographic).
        output_spacing_m: Output pixel spacing in meters.
        output_crs: Output CRS (default: WGS84 geographic).
        dem_path: Optional DEM for terrain correction (unused in current impl).

    Returns:
        Tuple of (geocoded_data, new_geotransform).
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError("GDAL is required for geocoding")

    rows, cols = data.shape
    logger.info(f"Geocoding {rows}×{cols} to {output_crs}, spacing={output_spacing_m}m")

    # Check if data already has a valid CRS
    if geo.crs_wkt and geo.crs_wkt.strip():
        return _reproject_with_gdal(data, geo, output_spacing_m, output_crs)
    else:
        logger.warning(
            "Data has no CRS. Geocoding requires georeferenced input. "
            "Returning data with approximate geotransform."
        )
        return data, geo


def geocode_with_gcps(
    data: np.ndarray,
    gcps: list,
    output_spacing_m: float = 30.0,
    output_crs: str = "EPSG:4326",
) -> Tuple[np.ndarray, GeoTransform]:
    """Geocode data using Ground Control Points.

    GCPs are (pixel_x, pixel_y, lon, lat) tuples generated from orbit
    geometry and DEM.

    Args:
        data: 2D array in radar coordinates.
        gcps: List of (pixel_x, pixel_y, lon, lat) tuples.
        output_spacing_m: Output pixel spacing.
        output_crs: Output CRS.

    Returns:
        Tuple of (geocoded_data, new_geotransform).
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError("GDAL is required for geocoding")

    if not gcps or len(gcps) < 3:
        raise ValueError("At least 3 GCPs required for geocoding")

    rows, cols = data.shape
    logger.info(f"Geocoding with {len(gcps)} GCPs")

    # Create in-memory source raster
    mem_driver = gdal.GetDriverByName('MEM')
    src_ds = mem_driver.Create('', cols, rows, 1, gdal.GDT_Float32)

    # Set GCPs
    gdal_gcps = []
    for px, py, lon, lat in gcps:
        gcp = gdal.GCP(lon, lat, 0, px, py)
        gdal_gcps.append(gcp)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src_ds.SetGCPs(gdal_gcps, srs.ExportToWkt())

    # Write data
    if np.issubdtype(data.dtype, np.complexfloating):
        src_ds.GetRasterBand(1).WriteArray(np.abs(data).astype(np.float32))
    else:
        src_ds.GetRasterBand(1).WriteArray(data.astype(np.float32))

    src_ds.GetRasterBand(1).SetNoDataValue(float('nan'))

    # Compute center latitude from GCPs for longitude correction
    center_lat = np.mean([lat for _, _, _, lat in gcps])
    if abs(center_lat) > 85.0:
        logger.warning(
            f"Center latitude {center_lat:.1f}\u00b0 is near-polar. "
            f"Clamping to \u00b185\u00b0 for resolution computation."
        )
        center_lat = min(max(center_lat, -85.0), 85.0)
    y_res = output_spacing_m / 111320.0
    x_res = output_spacing_m / (111320.0 * math.cos(math.radians(center_lat)))

    # Warp to output CRS
    warp_options = gdal.WarpOptions(
        dstSRS=output_crs,
        xRes=x_res,
        yRes=y_res,
        resampleAlg='bilinear',
        tps=True,  # Thin Plate Spline for GCP transformation
    )

    dst_ds = gdal.Warp('', src_ds, options=warp_options, format='MEM')
    if dst_ds is None:
        raise RuntimeError("GDAL Warp failed during geocoding")

    geocoded = dst_ds.GetRasterBand(1).ReadAsArray()
    gt = dst_ds.GetGeoTransform()
    crs_wkt = dst_ds.GetProjection()

    new_geo = GeoTransform(
        origin_x=gt[0], origin_y=gt[3],
        pixel_size_x=gt[1], pixel_size_y=gt[5],
        crs_wkt=crs_wkt,
        rows=geocoded.shape[0], cols=geocoded.shape[1],
    )

    src_ds = None
    dst_ds = None

    logger.info(f"Geocoded to {geocoded.shape[0]}×{geocoded.shape[1]}")
    return geocoded, new_geo


def _reproject_with_gdal(
    data: np.ndarray,
    geo: GeoTransform,
    output_spacing_m: float,
    output_crs: str,
) -> Tuple[np.ndarray, GeoTransform]:
    """Reproject data from one CRS to another using GDAL."""
    from osgeo import gdal, osr

    rows, cols = data.shape

    # Create in-memory source raster
    mem_driver = gdal.GetDriverByName('MEM')

    is_complex = np.issubdtype(data.dtype, np.complexfloating)
    if is_complex:
        # Store real and imaginary as separate bands
        src_ds = mem_driver.Create('', cols, rows, 2, gdal.GDT_Float32)
        src_ds.GetRasterBand(1).WriteArray(data.real.astype(np.float32))
        src_ds.GetRasterBand(2).WriteArray(data.imag.astype(np.float32))
    else:
        src_ds = mem_driver.Create('', cols, rows, 1, gdal.GDT_Float32)
        src_ds.GetRasterBand(1).WriteArray(data.astype(np.float32))

    # Set source geotransform and CRS
    src_ds.SetGeoTransform([
        geo.origin_x, geo.pixel_size_x, 0,
        geo.origin_y, 0, geo.pixel_size_y,
    ])
    src_ds.SetProjection(geo.crs_wkt)

    # Determine output resolution
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(geo.crs_wkt)

    dst_srs = osr.SpatialReference()
    if output_crs.startswith('EPSG:'):
        dst_srs.ImportFromEPSG(int(output_crs.split(':')[1]))
    else:
        dst_srs.ImportFromWkt(output_crs)

    # Use degrees or meters depending on target CRS
    if dst_srs.IsGeographic():
        # Compute center latitude from the geographic extent of the data
        center_lat = geo.origin_y + (rows / 2.0) * geo.pixel_size_y
        if abs(center_lat) > 85.0:
            logger.warning(
                f"Center latitude {center_lat:.1f}\u00b0 is near-polar. "
                f"Clamping to \u00b185\u00b0 for resolution computation."
            )
            center_lat = min(max(center_lat, -85.0), 85.0)
        y_res = output_spacing_m / 111320.0
        x_res = output_spacing_m / (111320.0 * math.cos(math.radians(center_lat)))
    else:
        x_res = output_spacing_m
        y_res = output_spacing_m

    warp_options = gdal.WarpOptions(
        dstSRS=output_crs,
        xRes=x_res,
        yRes=y_res,
        resampleAlg='bilinear',
    )

    dst_ds = gdal.Warp('', src_ds, options=warp_options, format='MEM')
    if dst_ds is None:
        raise RuntimeError("GDAL reprojection failed")

    if is_complex:
        real_part = dst_ds.GetRasterBand(1).ReadAsArray()
        imag_part = dst_ds.GetRasterBand(2).ReadAsArray()
        geocoded = (real_part + 1j * imag_part).astype(np.complex64)
    else:
        geocoded = dst_ds.GetRasterBand(1).ReadAsArray()

    gt = dst_ds.GetGeoTransform()
    crs_wkt = dst_ds.GetProjection()

    new_geo = GeoTransform(
        origin_x=gt[0], origin_y=gt[3],
        pixel_size_x=gt[1], pixel_size_y=gt[5],
        crs_wkt=crs_wkt,
        rows=geocoded.shape[0], cols=geocoded.shape[1],
    )

    src_ds = None
    dst_ds = None

    logger.info(f"Reprojected to {geocoded.shape[0]}×{geocoded.shape[1]}")
    return geocoded, new_geo


def export_geocoded(
    data: np.ndarray,
    geo: GeoTransform,
    output_path: str,
    nodata: float = float('nan'),
) -> None:
    """Export geocoded data to GeoTIFF.

    Args:
        data: Geocoded 2D array.
        geo: Geotransform with CRS.
        output_path: Output file path.
        nodata: NoData value.
    """
    from ocean_rs.shared.raster_io import RasterIO

    metadata = {
        'geotransform': [
            geo.origin_x, geo.pixel_size_x, 0,
            geo.origin_y, 0, geo.pixel_size_y,
        ],
        'projection': geo.crs_wkt,
    }

    RasterIO.write_raster(
        data.astype(np.float32),
        output_path,
        metadata,
        nodata=nodata,
    )
    logger.info(f"Exported: {output_path}")
