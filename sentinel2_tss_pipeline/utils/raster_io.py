"""
Raster I/O utilities using GDAL.

Provides functions for reading and writing geospatial raster files.
"""

import os
import logging
from typing import Dict, Tuple, Optional

import numpy as np

try:
    from osgeo import gdal, gdalconst
    GDAL_AVAILABLE = True
    # Suppress PROJ/GDAL error messages (e.g., "PROJ: proj_identify: SQLite error")
    gdal.DontUseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
except ImportError:
    GDAL_AVAILABLE = False

logger = logging.getLogger('sentinel2_tss_pipeline')


class RasterIO:
    """Utilities for raster input/output operations using GDAL"""

    @staticmethod
    def read_raster(file_path: str) -> Tuple[np.ndarray, dict]:
        """
        Read raster file and return data array with metadata.

        Args:
            file_path: Path to raster file

        Returns:
            Tuple of (data array, metadata dict)
        """
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for raster operations")

        try:
            dataset = gdal.Open(file_path, gdalconst.GA_ReadOnly)
            if dataset is None:
                raise ValueError(f"Could not open raster file: {file_path}")

            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray().astype(np.float32)
            nodata = band.GetNoDataValue()

            # Apply nodata mask
            if nodata is not None:
                data[data == nodata] = np.nan

            metadata = {
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection(),
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'nodata': nodata if nodata is not None else -9999
            }

            dataset = None  # Close dataset
            return data, metadata

        except Exception as e:
            logger.error(f"Error reading raster {file_path}: {e}")
            raise

    @staticmethod
    def write_raster(data: np.ndarray, output_path: str, metadata: dict,
                     description: str = "", nodata: float = -9999) -> bool:
        """
        Write numpy array to raster file.

        Args:
            data: 2D numpy array to write
            output_path: Output file path
            metadata: Geospatial metadata dict
            description: Band description
            nodata: NoData value

        Returns:
            True if successful
        """
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for raster operations")

        dataset = None
        try:
            # Validate inputs
            if not isinstance(data, np.ndarray):
                logger.error(f"Error writing raster {output_path}: not a numpy array")
                return False

            if not isinstance(metadata, dict):
                logger.error(f"Invalid metadata type: {type(metadata)}")
                return False

            if data.ndim != 2:
                logger.error(f"Data must be 2D array, got {data.ndim}D with shape {data.shape}")
                return False

            # Ensure contiguous array for GDAL
            output_data = np.ascontiguousarray(data, dtype=np.float32)

            # Replace NaN with nodata
            nan_mask = np.isnan(output_data)
            output_data[nan_mask] = nodata

            height, width = output_data.shape

            # Handle metadata safely
            geotransform = metadata.get('geotransform')
            projection = metadata.get('projection')

            if geotransform is None:
                logger.warning("No geotransform in metadata, using default")
                geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

            if projection is None:
                logger.warning("No projection in metadata, using default WGS84")
                projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'

            # Create GDAL dataset
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                logger.error("GTiff driver not available")
                return False

            dataset = driver.Create(
                output_path,
                width,
                height,
                1,  # Single band
                gdal.GDT_Float32,
                ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
            )

            if dataset is None:
                logger.error(f"Failed to create GDAL dataset: {output_path}")
                return False

            # Set georeference information
            try:
                dataset.SetGeoTransform(geotransform)
                dataset.SetProjection(projection)
            except Exception as georef_error:
                logger.warning(f"Error setting georeference: {georef_error}")

            # Write data
            band = dataset.GetRasterBand(1)
            write_result = band.WriteArray(output_data)

            if write_result != 0:
                logger.error(f"GDAL WriteArray failed with code: {write_result}")
                return False

            band.SetNoDataValue(nodata)
            if description:
                band.SetDescription(description)

            # Calculate statistics
            try:
                band.ComputeStatistics(False)
                band.FlushCache()
                dataset.FlushCache()
            except Exception as stats_error:
                logger.warning(f"Could not compute statistics: {stats_error}")

            dataset = None  # Close dataset

            logger.info(f"Successfully wrote raster: {os.path.basename(output_path)}")
            return True

        except Exception as e:
            logger.error(f"Error writing raster {output_path}: {e}")
            return False
        finally:
            if dataset is not None:
                dataset = None

    @staticmethod
    def calculate_statistics(data: np.ndarray, nodata: float = -9999) -> Dict:
        """
        Calculate statistics for data array.

        Args:
            data: Input numpy array
            nodata: NoData value to exclude

        Returns:
            Dict with count, min, max, mean, std, coverage_percent
        """
        valid_data = data[~np.isnan(data) & (data != nodata)]

        if len(valid_data) == 0:
            return {
                'count': 0, 'min': nodata, 'max': nodata,
                'mean': nodata, 'std': nodata, 'coverage_percent': 0.0
            }

        return {
            'count': len(valid_data),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'coverage_percent': (len(valid_data) / data.size) * 100
        }

    @staticmethod
    def load_bands_safely(band_paths: Dict[str, str], logger_instance=None) -> Tuple[Dict, Dict]:
        """
        Load multiple bands with consistent error handling.

        Args:
            band_paths: Dict mapping band names to file paths
            logger_instance: Optional logger

        Returns:
            Tuple of (data dict, metadata dict)
        """
        log = logger_instance or logger
        data = {}
        metadata = {}

        for name, path in band_paths.items():
            try:
                if os.path.exists(path) and os.path.getsize(path) > 1024:
                    d, m = RasterIO.read_raster(path)
                    data[name] = d
                    metadata[name] = m
            except Exception as e:
                log.warning(f"Failed to load {name}: {e}")

        return data, metadata
