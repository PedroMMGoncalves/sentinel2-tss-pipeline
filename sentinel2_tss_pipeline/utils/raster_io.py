"""
Raster I/O utilities using GDAL.

Provides functions for reading and writing geospatial raster files.
"""

import os
import logging
from typing import Dict, Tuple, Optional

import numpy as np

# Suppress PROJ/GDAL stderr messages BEFORE importing GDAL
os.environ['CPL_LOG'] = 'NUL' if os.name == 'nt' else '/dev/null'
os.environ['PROJ_DEBUG'] = '0'

try:
    from osgeo import gdal, gdalconst
    GDAL_AVAILABLE = True
    gdal.UseExceptions()

    def _gdal_error_handler(err_class, err_num, err_msg):
        """Route GDAL errors through Python logging instead of stderr."""
        if err_class == gdal.CE_Warning:
            logger.debug(f"GDAL warning {err_num}: {err_msg}")
        elif err_class == gdal.CE_Failure:
            logger.warning(f"GDAL error {err_num}: {err_msg}")
        elif err_class == gdal.CE_Fatal:
            logger.error(f"GDAL fatal {err_num}: {err_msg}")

    gdal.PushErrorHandler(_gdal_error_handler)
except ImportError:
    GDAL_AVAILABLE = False

logger = logging.getLogger('sentinel2_tss_pipeline')


class RasterIO:
    """Utilities for raster input/output operations using GDAL"""

    @staticmethod
    def read_raster(file_path: str, band_index: int = 1) -> Tuple[np.ndarray, dict]:
        """
        Read raster file and return data array with metadata.

        Args:
            file_path: Path to raster file
            band_index: Band number to read (1-based, default=1)

        Returns:
            Tuple of (data array, metadata dict)
        """
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL is required for raster operations")

        try:
            dataset = gdal.Open(file_path, gdalconst.GA_ReadOnly)
            if dataset is None:
                raise ValueError(f"Could not open raster file: {file_path}")

            band = dataset.GetRasterBand(band_index)
            data = band.ReadAsArray(buf_type=gdal.GDT_Float32)
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
                     description: str = "", nodata: float = -9999,
                     dtype: str = "float32") -> bool:
        """
        Write numpy array to raster file.

        Args:
            data: 2D numpy array to write
            output_path: Output file path
            metadata: Geospatial metadata dict
            description: Band description
            nodata: NoData value
            dtype: Output data type - "float32" (default), "uint8", "int16", "float64"

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

            # Map dtype string to GDAL type and numpy type
            dtype_map = {
                'float32': (gdal.GDT_Float32, np.float32, '3'),  # PREDICTOR=3 for floats
                'float64': (gdal.GDT_Float64, np.float64, '3'),
                'uint8':   (gdal.GDT_Byte, np.uint8, '2'),       # PREDICTOR=2 for integers
                'int16':   (gdal.GDT_Int16, np.int16, '2'),
                'int32':   (gdal.GDT_Int32, np.int32, '2'),
                'uint16':  (gdal.GDT_UInt16, np.uint16, '2'),
            }
            gdal_dtype, np_dtype, predictor = dtype_map.get(dtype, (gdal.GDT_Float32, np.float32, '3'))

            # Ensure contiguous array with correct type
            output_data = np.ascontiguousarray(data, dtype=np_dtype)

            # Replace NaN with nodata (only for float types)
            if np.issubdtype(np_dtype, np.floating):
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
                # Warn if geotransform suggests projected coordinates
                if geotransform and abs(geotransform[0]) > 360:
                    logger.warning("Geotransform suggests projected coordinates but falling back to WGS84 - "
                                   "output CRS may be incorrect")

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
                gdal_dtype,
                ['COMPRESS=LZW', f'PREDICTOR={predictor}', 'TILED=YES', 'BIGTIFF=IF_SAFER']
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

            # Set provenance metadata
            from datetime import datetime
            dataset.SetMetadataItem('PROCESSING_SOFTWARE', 'sentinel2_tss_pipeline v2.0')
            dataset.SetMetadataItem('PROCESSING_DATE', datetime.now().isoformat())
            if description:
                dataset.SetMetadataItem('ALGORITHM', description)

            # Calculate statistics
            try:
                band.ComputeStatistics(False)
                band.FlushCache()
                dataset.FlushCache()
            except Exception as stats_error:
                logger.warning(f"Could not compute statistics: {stats_error}")

            dataset = None  # Close dataset

            logger.debug(f"Successfully wrote raster: {os.path.basename(output_path)}")
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
