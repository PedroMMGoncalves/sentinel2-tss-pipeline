"""
Geometry Loading and Validation Utilities.

This module provides functions to load geometry from various file formats
(Shapefile, KML, GeoJSON) and convert them to WKT for use in SNAP subsetting.

Supports multiple backends (Fiona, OGR) for robust file reading that
bypasses common GeoPandas/PROJ issues.

Function naming follows Action-Object convention:
    - load_geometry: Main entry point for loading from any supported format
    - load_shapefile: Load from ESRI Shapefile
    - load_kml: Load from KML
    - load_geojson: Load from GeoJSON
    - validate_wkt: Validate WKT geometry string
    - generate_area_name: Create descriptive name from geometry
"""

import os
import sys
import logging
import traceback
from typing import Tuple, Optional

logger = logging.getLogger('sentinel2_tss_pipeline')

# Check for optional dependencies
try:
    from shapely.geometry import shape
    from shapely.wkt import loads as wkt_loads
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.warning("Shapely not available - geometry functions limited")

try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False
    logger.debug("Fiona not available - will use OGR fallback")

try:
    from osgeo import ogr
    HAS_OGR = True
except ImportError:
    HAS_OGR = False
    logger.debug("OGR not available")


def _combine_geometries(geometries: list, source_name: str) -> Tuple[object, str]:
    """
    Combine multiple geometries into a single geometry.

    Args:
        geometries: List of shapely geometry objects.
        source_name: Name of the source (for info message).

    Returns:
        Tuple of (combined_geometry, info_message).
    """
    if len(geometries) > 1:
        combined = unary_union(geometries)
        info_msg = f"Loaded {len(geometries)} features from {source_name}, combined into single geometry"
    else:
        combined = geometries[0]
        info_msg = f"Loaded single feature from {source_name}"

    return combined, info_msg


def _fix_invalid_geometry(geometry, info_msg: str) -> Tuple[object, str]:
    """
    Attempt to fix invalid geometry using buffer(0) trick.

    Args:
        geometry: Shapely geometry object.
        info_msg: Current info message to append to.

    Returns:
        Tuple of (fixed_geometry, updated_info_msg).
    """
    if hasattr(geometry, 'is_valid') and not geometry.is_valid:
        logger.warning("Geometry is not valid, attempting to fix...")
        try:
            geometry = geometry.buffer(0)
            info_msg += " (fixed invalid geometry)"
            logger.info("Geometry fixed successfully")
        except Exception as e:
            logger.warning(f"Could not fix geometry: {e}")
            info_msg += " (warning: geometry may be invalid)"

    return geometry, info_msg


def _add_bounds_info(geometry, info_msg: str, crs_note: str = "WGS84") -> str:
    """
    Add bounds information to info message.

    Args:
        geometry: Shapely geometry object.
        info_msg: Current info message.
        crs_note: Note about coordinate reference system.

    Returns:
        Updated info message with bounds.
    """
    try:
        bounds = geometry.bounds
        info_msg += f"\nGeometry type: {geometry.geom_type}"
        info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"
        info_msg += f"\nCoordinates: {crs_note}"
    except Exception as e:
        info_msg += f"\nWarning: Could not calculate bounds: {e}"

    return info_msg


def load_shapefile(shapefile_path: str) -> Tuple[Optional[str], str, bool]:
    """
    Load geometry from ESRI Shapefile.

    Uses Fiona as primary method, with OGR as fallback.
    Bypasses GeoPandas PROJ issues by using lower-level libraries.

    Args:
        shapefile_path: Path to .shp file.

    Returns:
        Tuple of (wkt_string, info_message, success).
        If loading fails, wkt_string is None and success is False.
    """
    if not HAS_SHAPELY:
        return None, "Shapely not installed. Install with: conda install -c conda-forge shapely", False

    try:
        logger.info(f"Loading shapefile: {shapefile_path}")
        combined_geometry = None
        info_msg = ""
        crs_info = "Unknown CRS"

        # Method 1: Try Fiona first
        if HAS_FIONA:
            try:
                features = []
                with fiona.open(shapefile_path) as src:
                    logger.info(f"Opened with Fiona - CRS: {src.crs}, Features: {len(src)}")
                    # Extract CRS information from shapefile
                    if src.crs:
                        crs_info = str(src.crs)
                    for feature in src:
                        features.append(feature)

                if not features:
                    return None, "Shapefile contains no features", False

                geometries = []
                for feature in features:
                    try:
                        geom = shape(feature['geometry'])
                        geometries.append(geom)
                    except Exception as e:
                        logger.warning(f"Skipped invalid geometry: {e}")

                if not geometries:
                    return None, "No valid geometries found", False

                combined_geometry, info_msg = _combine_geometries(geometries, "Shapefile with Fiona")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as fiona_error:
                logger.warning(f"Fiona failed: {fiona_error}, trying OGR...")
                combined_geometry = None

        # Method 2: Try OGR as fallback
        if combined_geometry is None and HAS_OGR:
            try:
                driver = ogr.GetDriverByName("ESRI Shapefile")
                datasource = driver.Open(shapefile_path, 0)

                if datasource is None:
                    return None, "Could not open shapefile with OGR", False

                layer = datasource.GetLayer()
                feature_count = layer.GetFeatureCount()
                # Extract CRS from OGR spatial reference
                srs = layer.GetSpatialRef()
                if srs:
                    auth_name = srs.GetAuthorityName(None)
                    auth_code = srs.GetAuthorityCode(None)
                    if auth_name and auth_code:
                        crs_info = f"{auth_name}:{auth_code}"
                    else:
                        crs_info = srs.ExportToProj4() or "Unknown CRS"
                logger.info(f"Opened with OGR - Features: {feature_count}")

                geometries = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    if geom:
                        wkt = geom.ExportToWkt()
                        try:
                            shapely_geom = wkt_loads(wkt)
                            geometries.append(shapely_geom)
                        except Exception as e:
                            logger.warning(f"Could not convert geometry: {e}")

                if not geometries:
                    return None, "No valid geometries found with OGR", False

                combined_geometry, info_msg = _combine_geometries(geometries, "Shapefile with OGR")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed: {ogr_error}", False

        if combined_geometry is None:
            return None, "No geometry loading backend available (install fiona or gdal)", False

        # Fix invalid geometry if needed
        combined_geometry, info_msg = _fix_invalid_geometry(combined_geometry, info_msg)

        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            logger.info(f"Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False

        # Add bounds information
        info_msg = _add_bounds_info(combined_geometry, info_msg, crs_info)

        logger.info(f"Successfully loaded geometry from: {os.path.basename(shapefile_path)}")
        return wkt_string, info_msg, True

    except Exception as e:
        logger.error(f"Critical error loading shapefile: {e}")
        logger.error(traceback.format_exc())
        return None, f"Critical error loading shapefile: {str(e)}", False


def load_kml(kml_path: str) -> Tuple[Optional[str], str, bool]:
    """
    Load geometry from KML file.

    Uses Fiona as primary method, with OGR as fallback.
    KML files are always in WGS84 (EPSG:4326).

    Args:
        kml_path: Path to .kml file.

    Returns:
        Tuple of (wkt_string, info_message, success).
    """
    if not HAS_SHAPELY:
        return None, "Shapely not installed. Install with: conda install -c conda-forge shapely", False

    try:
        logger.info(f"Loading KML file: {kml_path}")
        combined_geometry = None
        info_msg = ""

        # Method 1: Try Fiona first
        if HAS_FIONA:
            try:
                with fiona.open(kml_path, driver='KML') as src:
                    logger.info(f"Opened KML with Fiona - CRS: {src.crs}, Features: {len(src)}")
                    features = list(src)

                if not features:
                    return None, "KML file contains no features", False

                geometries = []
                for feature in features:
                    try:
                        geom = shape(feature['geometry'])
                        geometries.append(geom)
                    except Exception as e:
                        logger.warning(f"Skipped invalid geometry: {e}")

                if not geometries:
                    return None, "No valid geometries found", False

                combined_geometry, info_msg = _combine_geometries(geometries, "KML with Fiona")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as fiona_error:
                logger.warning(f"Fiona failed: {fiona_error}, trying OGR...")
                combined_geometry = None

        # Method 2: Try OGR as fallback
        if combined_geometry is None and HAS_OGR:
            try:
                driver = ogr.GetDriverByName("KML")
                if driver is None:
                    driver = ogr.GetDriverByName("LIBKML")

                if driver is None:
                    return None, "KML driver not available in OGR", False

                datasource = driver.Open(kml_path, 0)
                if datasource is None:
                    return None, "Could not open KML file with OGR", False

                geometries = []
                layer_count = datasource.GetLayerCount()
                logger.info(f"Opened KML with OGR - Layers: {layer_count}")

                total_features = 0
                for layer_idx in range(layer_count):
                    layer = datasource.GetLayer(layer_idx)
                    layer_feature_count = layer.GetFeatureCount()
                    total_features += layer_feature_count
                    logger.debug(f"Layer {layer_idx}: {layer_feature_count} features")

                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        if geom:
                            wkt = geom.ExportToWkt()
                            try:
                                shapely_geom = wkt_loads(wkt)
                                geometries.append(shapely_geom)
                            except Exception as e:
                                logger.warning(f"Could not convert geometry: {e}")

                if not geometries:
                    return None, f"No valid geometries found in KML (total features: {total_features})", False

                combined_geometry, info_msg = _combine_geometries(geometries, "KML with OGR")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed for KML: {ogr_error}", False

        if combined_geometry is None:
            return None, "No geometry loading backend available", False

        # Fix invalid geometry if needed
        combined_geometry, info_msg = _fix_invalid_geometry(combined_geometry, info_msg)

        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            logger.info(f"Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False

        # Add bounds information
        info_msg = _add_bounds_info(combined_geometry, info_msg, "KML is always in WGS84 (EPSG:4326)")

        logger.info(f"Successfully loaded geometry from KML: {os.path.basename(kml_path)}")
        return wkt_string, info_msg, True

    except Exception as e:
        logger.error(f"Critical error loading KML: {e}")
        logger.error(traceback.format_exc())
        return None, f"Critical error loading KML: {str(e)}", False


def load_geojson(geojson_path: str) -> Tuple[Optional[str], str, bool]:
    """
    Load geometry from GeoJSON file.

    Uses Fiona as primary method, with OGR as fallback.
    GeoJSON default CRS is WGS84 (EPSG:4326).

    Args:
        geojson_path: Path to .geojson or .json file.

    Returns:
        Tuple of (wkt_string, info_message, success).
    """
    if not HAS_SHAPELY:
        return None, "Shapely not installed. Install with: conda install -c conda-forge shapely", False

    try:
        logger.info(f"Loading GeoJSON file: {geojson_path}")
        combined_geometry = None
        info_msg = ""

        # Method 1: Try Fiona first
        if HAS_FIONA:
            try:
                with fiona.open(geojson_path, driver='GeoJSON') as src:
                    logger.info(f"Opened GeoJSON with Fiona - CRS: {src.crs}, Features: {len(src)}")
                    features = list(src)

                if not features:
                    return None, "GeoJSON contains no features", False

                geometries = []
                for feature in features:
                    try:
                        geom = shape(feature['geometry'])
                        geometries.append(geom)
                    except Exception as e:
                        logger.warning(f"Skipped invalid geometry: {e}")

                if not geometries:
                    return None, "No valid geometries found", False

                combined_geometry, info_msg = _combine_geometries(geometries, "GeoJSON with Fiona")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as fiona_error:
                logger.warning(f"Fiona failed: {fiona_error}, trying OGR...")
                combined_geometry = None

        # Method 2: Try OGR as fallback
        if combined_geometry is None and HAS_OGR:
            try:
                driver = ogr.GetDriverByName("GeoJSON")
                if driver is None:
                    return None, "GeoJSON driver not available in OGR", False

                datasource = driver.Open(geojson_path, 0)
                if datasource is None:
                    return None, "Could not open GeoJSON file with OGR", False

                layer = datasource.GetLayer()
                feature_count = layer.GetFeatureCount()
                logger.info(f"Opened GeoJSON with OGR - Features: {feature_count}")

                geometries = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    if geom:
                        wkt = geom.ExportToWkt()
                        try:
                            shapely_geom = wkt_loads(wkt)
                            geometries.append(shapely_geom)
                        except Exception as e:
                            logger.warning(f"Could not convert geometry: {e}")

                if not geometries:
                    return None, "No valid geometries found with OGR", False

                combined_geometry, info_msg = _combine_geometries(geometries, "GeoJSON with OGR")
                logger.info(f"Geometry type: {combined_geometry.geom_type}")

            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed for GeoJSON: {ogr_error}", False

        if combined_geometry is None:
            return None, "No geometry loading backend available", False

        # Fix invalid geometry if needed
        combined_geometry, info_msg = _fix_invalid_geometry(combined_geometry, info_msg)

        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            logger.info(f"Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False

        # Add bounds information
        info_msg = _add_bounds_info(combined_geometry, info_msg, "Assumed WGS84 (GeoJSON default)")

        logger.info(f"Successfully loaded geometry from GeoJSON: {os.path.basename(geojson_path)}")
        return wkt_string, info_msg, True

    except Exception as e:
        logger.error(f"Critical error loading GeoJSON: {e}")
        logger.error(traceback.format_exc())
        return None, f"Critical error loading GeoJSON: {str(e)}", False


def load_geometry(file_path: str) -> Tuple[Optional[str], str, bool]:
    """
    Load geometry from various file formats and convert to WKT.

    Automatically detects file format based on extension and calls
    the appropriate loader function.

    Supported formats:
        - Shapefile (.shp)
        - KML (.kml)
        - GeoJSON (.geojson, .json)

    Args:
        file_path: Path to geometry file.

    Returns:
        Tuple of (wkt_string, info_message, success).
        If loading fails, wkt_string is None and success is False.

    Example:
        >>> wkt, info, success = load_geometry("area.shp")
        >>> if success:
        ...     print(f"Loaded geometry: {info}")
    """
    if not HAS_SHAPELY:
        return None, "Shapely not installed. Install with: conda install -c conda-forge shapely", False

    try:
        file_extension = file_path.lower().split('.')[-1]

        if file_extension == 'shp':
            return load_shapefile(file_path)
        elif file_extension == 'kml':
            return load_kml(file_path)
        elif file_extension in ['geojson', 'json']:
            return load_geojson(file_path)
        else:
            return None, f"Unsupported file format: {file_extension}. Supported: shp, kml, geojson, json", False

    except Exception as e:
        logger.error(f"Error loading geometry from file: {e}")
        return None, f"Error loading file: {str(e)}", False


def validate_wkt(wkt_string: str) -> Tuple[bool, str]:
    """
    Validate WKT geometry string.

    Checks if the WKT string can be parsed and represents a valid geometry.

    Args:
        wkt_string: WKT geometry string to validate.

    Returns:
        Tuple of (is_valid, message).
        If valid, message contains geometry info.
        If invalid, message contains error description.

    Example:
        >>> is_valid, msg = validate_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
        >>> print(f"Valid: {is_valid}, Info: {msg}")
    """
    if not HAS_SHAPELY:
        return True, "Shapely not available for validation"

    try:
        test_geom = wkt_loads(wkt_string)

        if not test_geom.is_valid:
            return False, "WKT geometry is not valid"

        bounds = test_geom.bounds
        info_msg = f"Valid {test_geom.geom_type} geometry"
        info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"

        return True, info_msg

    except Exception as e:
        return False, f"Invalid WKT string: {str(e)}"


def generate_area_name(wkt_string: Optional[str]) -> str:
    """
    Generate a descriptive area name from a WKT geometry string.

    Creates a name based on the geometry bounds for use in output
    file naming.

    Args:
        wkt_string: WKT geometry string, or None for full scene.

    Returns:
        Area name string. Returns "FullScene" if no geometry provided.

    Example:
        >>> name = generate_area_name("POLYGON ((-9 40, -7 40, -7 42, -9 42, -9 40))")
        >>> print(name)  # "CustomArea_-9.000_40.000_-7.000_42.000"
    """
    if wkt_string:
        try:
            if HAS_SHAPELY:
                geom = wkt_loads(wkt_string)
                bounds = geom.bounds
                return f"CustomArea_{bounds[0]:.3f}_{bounds[1]:.3f}_{bounds[2]:.3f}_{bounds[3]:.3f}"
            else:
                return "CustomArea"
        except Exception:
            return "CustomArea"
    return "FullScene"


def create_bbox_wkt(north: float, south: float, east: float, west: float) -> str:
    """
    Create a WKT POLYGON from bounding box coordinates.

    Args:
        north: Northern latitude bound.
        south: Southern latitude bound.
        east: Eastern longitude bound.
        west: Western longitude bound.

    Returns:
        WKT POLYGON string.

    Raises:
        ValueError: If coordinates are invalid.

    Example:
        >>> wkt = create_bbox_wkt(42.0, 40.0, -7.0, -9.0)
        >>> print(wkt)  # "POLYGON ((-9.0 40.0, -7.0 40.0, -7.0 42.0, -9.0 42.0, -9.0 40.0))"
    """
    # Validate coordinate ranges
    if not (-90 <= south <= north <= 90):
        raise ValueError("Invalid latitude values. Must be between -90 and 90 degrees, south <= north.")

    if not (-180 <= west <= east <= 180):
        raise ValueError("Invalid longitude values. Must be between -180 and 180 degrees, west <= east.")

    return f"POLYGON (({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"


__all__ = [
    'load_geometry',
    'load_shapefile',
    'load_kml',
    'load_geojson',
    'validate_wkt',
    'generate_area_name',
    'create_bbox_wkt',
    'HAS_SHAPELY',
    'HAS_FIONA',
    'HAS_OGR',
]
