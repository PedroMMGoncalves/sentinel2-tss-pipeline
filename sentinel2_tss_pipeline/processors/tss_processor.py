"""
Jiang TSS Processor.

Complete implementation of Jiang et al. (2021) TSS methodology.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386

Water Type Classification:
    Type I (Clear): Rrs(490) > Rrs(560) - uses 560nm
    Type II (Moderately turbid): Rrs(490) > Rrs(620) - uses 665nm
    Type III (Highly turbid): Default - uses 740nm
    Type IV (Extremely turbid): Rrs(740) > Rrs(490) AND Rrs(740) > 0.010 - uses 865nm
"""

import os
import glob
import logging
import warnings
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import TSSConfig, WaterQualityConfig
from ..utils.raster_io import RasterIO
from ..utils.output_structure import OutputStructure
from .tsm_chl_calculator import ProcessingResult
from .water_quality_processor import WaterQualityProcessor
from .visualization_processor import VisualizationProcessor

logger = logging.getLogger('sentinel2_tss_pipeline')


class TSSConstants:
    """
    Complete TSS configuration constants from Jiang et al. (2021).

    Reference:
        Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
        "Remotely Estimating Total Suspended Solids Concentration in Clear to
        Extremely Turbid Waters Using a Novel Semi-Analytical Method."
        Remote Sensing of Environment, 258, 112386.
        DOI: https://doi.org/10.1016/j.rse.2021.112386

    Water Type Classification:
        Type I (Clear): Rrs(490) > Rrs(560) - uses 560nm
        Type II (Moderately turbid): Rrs(490) > Rrs(620) - uses 665nm
        Type III (Highly turbid): Default - uses 740nm
        Type IV (Extremely turbid): Rrs(740) > Rrs(490) AND Rrs(740) > 0.010 - uses 865nm
    """

    # Pure water absorption coefficients (aw) in m^-1 - FROM ORIGINAL R CODE
    PURE_WATER_ABSORPTION = {
        443: 0.00515124,    # Band B1
        490: 0.01919594,    # Band B2
        560: 0.06299986,    # Band B3 (Type I)
        665: 0.41395333,    # Band B4 (Type II)
        705: 0.70385758,    # Band B5
        740: 2.71167020,    # Band B6 (Type III)
        783: 2.62000141,    # Band B7
        865: 4.61714226     # Band B8A (Type IV)
    }

    # Pure water backscattering coefficients (bbw) in m^-1 - FROM ORIGINAL R CODE
    PURE_WATER_BACKSCATTERING = {
        443: 0.00215037,    # Band B1
        490: 0.00138116,    # Band B2
        560: 0.00078491,    # Band B3 (Type I)
        665: 0.00037474,    # Band B4 (Type II)
        705: 0.00029185,    # Band B5
        740: 0.00023499,    # Band B6 (Type III)
        783: 0.00018516,    # Band B7
        865: 0.00012066     # Band B8A (Type IV)
    }

    # TSS conversion factors from Jiang et al. (2021) - FROM ORIGINAL R CODE
    TSS_CONVERSION_FACTORS = {
        560: 94.48785,      # Type I: Clear water
        665: 113.87498,     # Type II: Moderately turbid
        740: 134.91845,     # Type III: Highly turbid
        865: 166.07382      # Type IV: Extremely turbid
    }

    # Rrs620 estimation coefficients - EXACT from R code
    RRS620_COEFFICIENTS = {
        'a': 1.693846e+02,  # a <- 1.693846e+02
        'b': -1.557556e+01, # b <- -1.557556e+01
        'c': 1.316727e+00,  # c <- 1.316727e+00
        'd': 1.484814e-04   # d <- 1.484814e-04
    }


class TSSProcessor:
    """Complete implementation of Jiang et al. 2021 TSS methodology"""

    def __init__(self, config: TSSConfig):
        """Initialize TSS Processor with configuration."""
        self.config = config
        self.constants = TSSConstants()

        # Initialize visualization processor
        if config.enable_visualization:
            self.viz_processor = VisualizationProcessor(config.output_categories)
            logger.debug("Visualization processor initialized")
        else:
            self.viz_processor = None
            logger.debug("Visualization disabled")

        # Initialize water quality processor if enabled
        if self.config.enable_water_quality:
            self.water_quality_processor = WaterQualityProcessor()
            if self.config.water_quality_config is None:
                self.config.water_quality_config = WaterQualityConfig()
        else:
            self.water_quality_processor = None

        logger.debug("Initialized TSS Processor")
        logger.debug(f"TSS processing enabled: {self.config.enable_tss_processing}")
        logger.debug(f"Water quality enabled: {self.config.enable_water_quality}")
        logger.debug(f"Visualization enabled: {self.config.enable_visualization}")

        if self.config.enable_water_quality and self.config.water_quality_config:
            logger.debug("Water quality algorithms available:")
            logger.debug(f"  Water Clarity: {self.config.water_quality_config.enable_water_clarity}")
            logger.debug(f"  HAB Detection: {self.config.water_quality_config.enable_hab_detection}")

        logger.debug("TSS Processor initialization completed successfully")

    def _load_bands_data(self, band_paths: Dict[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load band data arrays from file paths"""
        bands_data = {}
        reference_metadata = None

        logger.debug(f"Loading {len(band_paths)} spectral bands into memory")

        for wavelength, file_path in band_paths.items():
            try:
                data, metadata = RasterIO.read_raster(file_path)
                bands_data[wavelength] = data

                if reference_metadata is None:
                    reference_metadata = metadata

                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.debug(f"Loaded {wavelength}nm: {file_size_mb:.1f}MB")

            except Exception as e:
                logger.error(f"Failed to load band {wavelength}nm from {file_path}: {e}")
                return None, None

        logger.debug(f"Successfully loaded {len(bands_data)} bands into memory")
        return bands_data, reference_metadata

    def _create_nir_water_mask(self, bands_data: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """
        Create water mask using simple NIR threshold.

        Water absorbs NIR strongly, so low NIR reflectance indicates water.
        This is simpler and more robust for turbid waters than index-based methods.

        Args:
            bands_data: Dictionary mapping wavelength (nm) to Rrs data

        Returns:
            Boolean array where True = water (NIR < threshold), False = land
            Returns None if NIR band not available
        """
        try:
            # Try B8A (865nm) first, fall back to B8 (842nm)
            if 865 in bands_data:
                nir_wl = 865
            elif 842 in bands_data:
                nir_wl = 842
                logger.debug("Using B8 (842nm) as fallback for water mask (B8A 865nm not available)")
            else:
                logger.debug("Cannot create NIR water mask: neither 865nm nor 842nm available")
                return None

            nir = bands_data[nir_wl]
            threshold = self.config.water_mask_threshold  # Default 0.03

            valid = ~np.isnan(nir)
            water_mask = np.zeros_like(nir, dtype=bool)
            water_mask[valid] = nir[valid] < threshold

            water_pixels = np.sum(water_mask)
            total_valid = np.sum(valid)
            if total_valid > 0:
                water_pct = 100.0 * water_pixels / total_valid
                logger.info(f"NIR water mask (< {threshold}): {water_pct:.1f}% water, {100-water_pct:.1f}% land")

            return water_mask

        except Exception as e:
            logger.error(f"Error creating NIR water mask: {e}")
            return None

    def _rasterize_shapefile_mask(self, shapefile_path: str,
                                   reference_metadata: dict) -> Optional[np.ndarray]:
        """
        Rasterize shapefile to create water mask matching output resolution.

        Handles CRS reprojection if shapefile CRS differs from raster CRS.

        Args:
            shapefile_path: Path to water polygon shapefile (UTM or WGS84)
            reference_metadata: GDAL metadata from reference raster

        Returns:
            Boolean array where True = inside polygon (water), False = outside (land)
            Returns None if shapefile cannot be read
        """
        try:
            from osgeo import gdal, ogr, osr

            # Open shapefile
            shp = ogr.Open(shapefile_path)
            if shp is None:
                logger.error(f"Could not open shapefile: {shapefile_path}")
                return None

            layer = shp.GetLayer()
            if layer is None:
                logger.error(f"Could not get layer from shapefile: {shapefile_path}")
                return None

            # Check CRS and reproject if needed
            shp_srs = layer.GetSpatialRef()
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(reference_metadata['projection'])

            # Keep reference to memory dataset for reprojected layer
            mem_ds = None
            rasterize_layer = layer

            if shp_srs and not shp_srs.IsSame(raster_srs):
                logger.info(f"Reprojecting shapefile from {shp_srs.GetName()} to {raster_srs.GetName()}")
                try:
                    # Create coordinate transformation
                    transform = osr.CoordinateTransformation(shp_srs, raster_srs)

                    # Create in-memory reprojected layer
                    mem_driver = ogr.GetDriverByName('Memory')
                    mem_ds = mem_driver.CreateDataSource('')
                    mem_layer = mem_ds.CreateLayer('reprojected', raster_srs, ogr.wkbPolygon)

                    # Copy and transform features
                    layer.ResetReading()
                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        if geom is not None:
                            geom_clone = geom.Clone()
                            geom_clone.Transform(transform)
                            new_feature = ogr.Feature(mem_layer.GetLayerDefn())
                            new_feature.SetGeometry(geom_clone)
                            mem_layer.CreateFeature(new_feature)

                    rasterize_layer = mem_layer
                    logger.debug(f"Reprojected {mem_layer.GetFeatureCount()} features")

                except Exception as e:
                    logger.warning(f"CRS reprojection failed: {e}, using original layer")
                    rasterize_layer = layer

            # Create memory raster matching reference
            driver = gdal.GetDriverByName('MEM')
            target = driver.Create('',
                                  reference_metadata['width'],
                                  reference_metadata['height'],
                                  1, gdal.GDT_Byte)
            target.SetGeoTransform(reference_metadata['geotransform'])
            target.SetProjection(reference_metadata['projection'])

            # Initialize with zeros (outside polygon = land)
            band = target.GetRasterBand(1)
            band.Fill(0)

            # Rasterize: 1 inside polygon, 0 outside
            gdal.RasterizeLayer(target, [1], rasterize_layer, burn_values=[1])

            mask = band.ReadAsArray()
            water_mask = mask == 1  # True = water (inside polygon)

            water_pixels = np.sum(water_mask)
            total_pixels = water_mask.size
            water_pct = 100.0 * water_pixels / total_pixels
            logger.info(f"Shapefile water mask: {water_pct:.1f}% water, {100-water_pct:.1f}% land")

            # Cleanup
            shp = None
            mem_ds = None
            target = None

            return water_mask

        except Exception as e:
            logger.error(f"Error rasterizing shapefile mask: {e}")
            return None

    def _convert_rhow_to_rrs(self, bands_data: Dict[int, np.ndarray],
                              band_paths: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Convert rhow (water-leaving reflectance) to Rrs (remote sensing reflectance).

        Applies Rrs = rhow / pi conversion for rhow bands.
        Rrs bands are returned unchanged.
        """
        converted_data = {}
        conversion_log = []

        # Analyze what band types we have
        band_types = set()
        for file_path in band_paths.values():
            filename = os.path.basename(file_path)
            if filename.startswith('rhow_'):
                band_types.add('rhow')
            elif filename.startswith('rrs_'):
                band_types.add('rrs')

        logger.debug(f"Detected band types: {list(band_types)}")

        # Apply conversions
        for wavelength, data in bands_data.items():
            file_path = band_paths[wavelength]
            filename = os.path.basename(file_path)

            if filename.startswith('rhow_'):
                # Convert rhow to Rrs: Rrs = rhow / pi
                converted_data[wavelength] = data / np.pi
                conversion_log.append(f"{wavelength}nm: rhow -> Rrs (/pi)")

            elif filename.startswith('rrs_'):
                # Already in Rrs units, no conversion needed
                converted_data[wavelength] = data.copy()
                conversion_log.append(f"{wavelength}nm: rrs (no conversion)")

            else:
                logger.warning(f"Unexpected band type for {wavelength}nm: {filename}")
                converted_data[wavelength] = data.copy()
                conversion_log.append(f"{wavelength}nm: unknown (no conversion)")

        # Log conversion summary (not individual bands)
        if len(band_types) > 1:
            logger.debug("Mixed band types - individual conversions applied")
        elif 'rrs' in band_types:
            logger.debug("All rrs bands - no conversion needed")
        elif 'rhow' in band_types:
            logger.debug("All rhow bands - converted to Rrs")

        for log_entry in conversion_log:
            logger.debug(f"  {log_entry}")

        # Validate converted data ranges
        self._validate_rrs_ranges(converted_data)

        return converted_data

    def _validate_rrs_ranges(self, rrs_data: Dict[int, np.ndarray]):
        """Validate that Rrs values are in reasonable ranges after conversion"""
        for wavelength, data in rrs_data.items():
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                logger.warning(f"Band {wavelength}nm: All values are NaN")
                continue

            min_val = np.min(valid_data)
            max_val = np.max(valid_data)

            # Typical Rrs ranges for water (sr^-1)
            if wavelength < 600:
                expected_max = 0.08
            elif wavelength < 700:
                expected_max = 0.04
            else:
                expected_max = 0.02

            if max_val > expected_max:
                logger.warning(f"Band {wavelength}nm: High Rrs values detected (max={max_val:.6f})")

            if min_val < -0.01:
                negative_count = np.sum(valid_data < -0.01)
                logger.warning(f"Band {wavelength}nm: {negative_count} significantly negative values detected")

            logger.debug(f"Band {wavelength}nm Rrs stats: min={min_val:.6f}, max={max_val:.6f}")

    def _update_band_mapping_for_mixed_types(self, c2rcc_path: str) -> Dict[int, str]:
        """Load bands that can be used for Jiang TSS algorithm"""
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"

        if not os.path.exists(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return {}

        # Only complete 8-band water reflectance products
        band_mapping = {
            443: ['rhow_B1.img', 'rrs_B1.img'],
            490: ['rhow_B2.img', 'rrs_B2.img'],
            560: ['rhow_B3.img', 'rrs_B3.img'],
            665: ['rhow_B4.img', 'rrs_B4.img'],
            705: ['rhow_B5.img', 'rrs_B5.img'],
            740: ['rhow_B6.img', 'rrs_B6.img'],
            783: ['rhow_B7.img', 'rrs_B7.img'],
            865: ['rhow_B8A.img', 'rrs_B8A.img']
        }

        found_bands = {}
        band_type_summary = {}

        logger.debug(f"Searching for COMPLETE 8-band datasets in: {data_folder}")

        for wavelength, possible_names in band_mapping.items():
            for name in possible_names:
                file_path = os.path.join(data_folder, name)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
                    found_bands[wavelength] = file_path

                    if name.startswith('rhow_'):
                        band_type = 'rhow'
                    elif name.startswith('rrs_'):
                        band_type = 'rrs'

                    band_type_summary[band_type] = band_type_summary.get(band_type, 0) + 1
                    logger.debug(f"Found {wavelength}nm: {name} ({band_type})")
                    break
            else:
                logger.warning(f"Missing {wavelength}nm - CRITICAL for Jiang algorithm")

        logger.debug(f"Band type summary: {band_type_summary}")

        total_found = len(found_bands)
        required_bands = [443, 490, 560, 665, 705, 740, 783, 865]
        missing_critical = [wl for wl in required_bands if wl not in found_bands]

        if total_found == 8 and not missing_critical:
            logger.debug("PERFECT: Complete 8-band dataset found - Jiang algorithm ready")
        elif total_found >= 6 and 783 in found_bands and 865 in found_bands:
            logger.debug(f"USABLE: {total_found}/8 bands found including critical NIR bands")
        else:
            logger.error(f"INSUFFICIENT: {total_found}/8 bands found")
            if missing_critical:
                logger.error(f"Missing CRITICAL bands: {missing_critical}nm")

        return found_bands

    def process_tss(self, c2rcc_path: str, output_folder: str, product_name: str,
                        intermediate_paths: Optional[Dict[str, str]] = None) -> Dict[str, ProcessingResult]:
        """Process Jiang TSS methodology from C2RCC outputs"""
        try:
            logger.debug(f"Starting Jiang TSS processing for: {product_name}")

            # Extract georeference from C2RCC output
            try:
                data_folder = c2rcc_path.replace('.dim', '.data')
                sample_band_path = os.path.join(data_folder, 'rrs_B4.img')

                if os.path.exists(sample_band_path):
                    _, reference_metadata = RasterIO.read_raster(sample_band_path)
                    logger.debug("Using C2RCC georeference for proper geographic positioning")
                else:
                    logger.error("Cannot find reference band for georeference")
                    reference_metadata = {
                        'geotransform': None,
                        'projection': None,
                        'width': None,
                        'height': None,
                        'nodata': -9999
                    }
            except Exception as e:
                logger.error(f"Error extracting georeference: {e}")

            # Create intermediate tracking
            intermediate_paths = {}

            # Load spectral bands directly
            logger.debug("Loading spectral bands from C2RCC output")
            data_folder = c2rcc_path.replace('.dim', '.data')

            band_files = {
                443: ['rrs_B1.img', 'rhow_B1.img'],
                490: ['rrs_B2.img', 'rhow_B2.img'],
                560: ['rrs_B3.img', 'rhow_B3.img'],
                665: ['rrs_B4.img', 'rhow_B4.img'],
                705: ['rrs_B5.img', 'rhow_B5.img'],
                740: ['rrs_B6.img', 'rhow_B6.img'],
                783: ['rrs_B7.img', 'rhow_B7.img'],
                865: ['rrs_B8A.img', 'rhow_B8A.img']
            }

            bands_data = {}
            band_paths = {}
            for wavelength, possible_files in band_files.items():
                for filename in possible_files:
                    file_path = os.path.join(data_folder, filename)
                    if os.path.exists(file_path):
                        try:
                            data, _ = RasterIO.read_raster(file_path)
                            bands_data[wavelength] = data
                            band_paths[wavelength] = file_path
                            logger.debug(f"Loaded {wavelength}nm: {filename}")
                            break
                        except Exception as e:
                            logger.error(f"Failed to load {wavelength}nm from {filename}: {e}")

            if not bands_data:
                error_msg = "No spectral bands found in C2RCC output"
                logger.error(error_msg)
                return {'error': ProcessingResult(False, "", None, error_msg)}

            # Apply unit conversion
            logger.debug("Converting rhow to Rrs")
            converted_bands_data = self._convert_rhow_to_rrs(bands_data, band_paths)

            # Create water mask to exclude land pixels
            # Priority: 1) Shapefile mask, 2) NIR threshold mask, 3) No mask
            water_mask = None

            # Option 1: Shapefile mask (highest priority)
            if hasattr(self.config, 'water_mask_shapefile') and self.config.water_mask_shapefile:
                if os.path.exists(self.config.water_mask_shapefile):
                    logger.info(f"Using shapefile water mask: {self.config.water_mask_shapefile}")
                    water_mask = self._rasterize_shapefile_mask(
                        self.config.water_mask_shapefile, reference_metadata)
                else:
                    logger.error(f"Shapefile not found: {self.config.water_mask_shapefile}")

            # Option 2: NIR threshold mask (if enabled and no shapefile)
            elif self.config.auto_water_mask:
                logger.info(f"Using NIR threshold water mask (< {self.config.water_mask_threshold})")
                water_mask = self._create_nir_water_mask(converted_bands_data)

            # Option 3: No mask
            else:
                logger.info("No water mask applied")

            # Apply Jiang methodology
            logger.debug("Applying Jiang TSS methodology")
            jiang_results = self._estimate_tss_all_pixels(converted_bands_data)

            # Process advanced algorithms
            advanced_results = {}
            if (self.config.enable_water_quality and
                hasattr(self, 'water_quality_processor') and
                self.water_quality_processor is not None):

                logger.debug("Processing water quality products")
                advanced_results = self._process_water_quality_products(
                    c2rcc_path, jiang_results, converted_bands_data, product_name
                )

                if advanced_results is None:
                    advanced_results = {}

                logger.debug(f"Water quality processing completed: {len(advanced_results)} additional products")

            # Combine all results
            all_algorithm_results = jiang_results.copy()

            for key, result in advanced_results.items():
                if isinstance(result, ProcessingResult) and result.stats and 'numpy_data' in result.stats:
                    all_algorithm_results[key] = result.stats['numpy_data']

            # Save complete results (with water mask applied to remove land pixels)
            logger.debug("Saving complete results including advanced algorithms")
            saved_results = self._save_tss_products(all_algorithm_results, output_folder,
                                                   product_name, reference_metadata, water_mask)

            # Update ProcessingResult objects with actual file paths
            final_results = saved_results.copy()

            for key, advanced_result in advanced_results.items():
                if key in saved_results:
                    advanced_result.output_path = saved_results[key].output_path
                    final_results[key] = advanced_result

            # Visualization processing (RGB composites + spectral indices)
            if (self.config.enable_visualization and
                hasattr(self, 'viz_processor') and
                self.viz_processor is not None):

                logger.debug("Processing marine visualizations")

                try:
                    # Use output_folder directly (not its parent) to match where s2_processor creates geometric products
                    geometric_folder = OutputStructure.get_intermediate_folder(
                        output_folder, OutputStructure.GEOMETRIC_FOLDER
                    )

                    if intermediate_paths is None:
                        intermediate_paths = {}

                    clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
                    if 'MSIL1C' in clean_product_name:
                        parts = clean_product_name.split('_')
                        if len(parts) >= 6:
                            clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
                        else:
                            clean_name = clean_product_name.replace('MSIL1C_', '')
                    else:
                        clean_name = clean_product_name

                    geometric_filename = f"Resampled_{clean_name}_Subset.dim"
                    geometric_path = os.path.join(geometric_folder, geometric_filename)
                    intermediate_paths['geometric_path'] = geometric_path

                    if os.path.exists(geometric_path):
                        logger.debug("Geometric products found for marine visualization")

                        viz_results = self.viz_processor.process_marine_visualizations(
                            c2rcc_path, output_folder, product_name, intermediate_paths
                        )

                        final_results.update(viz_results)

                        rgb_count = len([k for k in viz_results.keys() if k.startswith('rgb_')])
                        index_count = len([k for k in viz_results.keys() if k.startswith('index_')])
                        logger.debug(f"Marine visualization completed: {rgb_count} RGB + {index_count} indices")

                        # Cleanup intermediate products
                        logger.debug("Starting intermediate products cleanup...")
                        try:
                            cleanup_success = self.viz_processor._cleanup_intermediate_products(
                                output_folder, product_name)
                            if cleanup_success:
                                logger.debug("Geometric products cleanup completed successfully")
                            else:
                                logger.warning("Geometric products cleanup had issues")
                        except Exception as cleanup_error:
                            logger.error(f"Geometric products cleanup failed: {cleanup_error}")

                    else:
                        logger.error(f"Geometric products not found at: {geometric_path}")
                        final_results['visualization_error'] = ProcessingResult(
                            success=False,
                            output_path="",
                            stats=None,
                            error_message="Geometric products not found for visualization"
                        )

                except Exception as viz_error:
                    logger.error(f"Marine visualization processing failed: {viz_error}")
                    final_results['visualization_error'] = ProcessingResult(
                        success=False,
                        output_path="",
                        stats=None,
                        error_message=str(viz_error)
                    )

            # Final validation
            success_count = 0
            for key, result in final_results.items():
                if isinstance(result, ProcessingResult) and hasattr(result, 'success'):
                    if result.success:
                        success_count += 1
                elif result is not None:
                    success_count += 1
            total_count = len(final_results)

            logger.debug("=" * 80)
            logger.debug(f"COMPLETE TSS PROCESSING FINISHED: {product_name}")
            logger.debug(f"   Total products generated: {success_count}/{total_count}")
            logger.debug(f"   Success rate: {(success_count/total_count)*100:.1f}%")
            logger.debug("=" * 80)

            # Create product index after all processing is complete
            try:
                self._create_product_index(final_results, output_folder, product_name)
            except Exception as e:
                logger.debug(f"Could not create product index: {e}")

            return final_results

        except Exception as e:
            error_msg = f"Jiang TSS processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def _process_water_quality_products(self, c2rcc_path: str, jiang_results: Dict,
                                        rrs_bands_data: Dict, product_name: str) -> Dict[str, ProcessingResult]:
        """Process water quality products gated by OutputCategoryConfig toggles."""
        try:
            logger.debug("Processing water quality algorithms")
            advanced_results = {}

            config = self.config.water_quality_config
            if config is None:
                config = WaterQualityConfig()

            categories = self.config.output_categories

            # Water clarity calculation — gated by output_categories.enable_water_clarity
            if categories.enable_water_clarity and 'absorption' in jiang_results and 'backscattering' in jiang_results:
                logger.debug("Calculating water clarity indices")

                absorption = jiang_results['absorption']
                backscattering = jiang_results['backscattering']

                try:
                    clarity_results = self.water_quality_processor.calculate_water_clarity(
                        absorption, backscattering, config.solar_zenith_angle
                    )

                    for key, value in clarity_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            stats = RasterIO.calculate_statistics(value)
                            stats['numpy_data'] = value
                            stats['product_type'] = 'water_clarity'
                            stats['description'] = f"Water clarity {key}"

                            advanced_results[f'advanced_clarity_{key}'] = ProcessingResult(
                                success=True, output_path="", stats=stats, error_message=None
                            )

                    logger.debug(f"Water clarity completed: {len(clarity_results)} products")

                except Exception as e:
                    logger.error(f"Water clarity calculation failed: {e}")

            # HAB detection — gated by output_categories.enable_hab
            if categories.enable_hab and rrs_bands_data:
                logger.debug("Detecting harmful algal blooms")

                try:
                    hab_results = self.water_quality_processor.detect_harmful_algal_blooms(
                        chlorophyll=None, phycocyanin=None, rrs_bands=rrs_bands_data
                    )

                    for key, value in hab_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            stats = RasterIO.calculate_statistics(value)
                            stats['numpy_data'] = value
                            stats['product_type'] = 'hab_detection'
                            stats['description'] = f"HAB {key}"
                            stats['biomass_threshold'] = config.hab_biomass_threshold
                            stats['extreme_threshold'] = config.hab_extreme_threshold

                            advanced_results[f'advanced_hab_{key}'] = ProcessingResult(
                                success=True, output_path="", stats=stats, error_message=None
                            )

                    logger.debug(f"HAB detection completed: {len(hab_results)} products")

                except Exception as e:
                    logger.error(f"HAB detection failed: {e}")

            # Trophic State Index — gated by output_categories.enable_trophic_state
            if categories.enable_trophic_state:
                logger.debug("Calculating Trophic State Index (Carlson 1977)")

                try:
                    # Get chlorophyll from SNAP C2RCC output
                    chl_data = self._extract_snap_chlorophyll(c2rcc_path)

                    if chl_data is not None:
                        # Get Secchi depth from clarity results if available
                        secchi_depth = None
                        secchi_key = 'advanced_clarity_secchi_depth'
                        if secchi_key in advanced_results:
                            secchi_depth = advanced_results[secchi_key].stats.get('numpy_data')

                        tsi_results = self.water_quality_processor.calculate_trophic_state(
                            chl_data, secchi_depth
                        )

                        for key, value in tsi_results.items():
                            if value is not None and isinstance(value, np.ndarray):
                                stats = RasterIO.calculate_statistics(value)
                                stats['numpy_data'] = value
                                stats['product_type'] = 'trophic_state'
                                stats['description'] = f"Trophic state {key} (Carlson 1977)"

                                advanced_results[f'advanced_tsi_{key}'] = ProcessingResult(
                                    success=True, output_path="", stats=stats, error_message=None
                                )

                        logger.debug(f"Trophic state completed: {len(tsi_results)} products")
                    else:
                        logger.warning("Trophic state skipped: SNAP chlorophyll not available")

                except Exception as e:
                    logger.error(f"Trophic state calculation failed: {e}")

            logger.debug(f"Water quality processing completed: {len(advanced_results)} products")
            return advanced_results

        except Exception as e:
            logger.error(f"Error in water quality processing: {e}")
            logger.error(traceback.format_exc())
            return {}

    def _extract_snap_chlorophyll(self, c2rcc_path: str) -> Optional[np.ndarray]:
        """Extract chlorophyll from SNAP C2RCC output"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            chl_path = os.path.join(data_folder, 'conc_chl.img')

            if os.path.exists(chl_path):
                chl_data, _ = RasterIO.read_raster(chl_path)
                logger.debug("Successfully extracted SNAP chlorophyll data")
                return chl_data
            else:
                logger.warning("SNAP chlorophyll data not found")
                return None

        except Exception as e:
            logger.error(f"Error extracting SNAP chlorophyll: {e}")
            return None

    def _estimate_tss_all_pixels(self, rrs_data: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """Estimate TSS for all valid pixels using Jiang methodology."""
        logger.debug("Applying Jiang methodology to pre-converted Rrs data")

        shape = rrs_data[443].shape

        # Initialize output arrays
        absorption = np.full(shape, np.nan, dtype=np.float32)
        backscattering = np.full(shape, np.nan, dtype=np.float32)
        reference_band = np.full(shape, np.nan, dtype=np.float32)
        tss_concentration = np.full(shape, np.nan, dtype=np.float32)
        water_type_classification = np.full(shape, 0, dtype=np.uint8)

        logger.debug("Using pre-converted Rrs data (no additional pi division)")

        # Create valid pixel mask
        valid_mask = self._create_valid_pixel_mask(rrs_data)

        if np.any(valid_mask):
            pixel_results = self._process_valid_pixels(rrs_data, valid_mask)

            absorption[valid_mask] = pixel_results['absorption']
            backscattering[valid_mask] = np.maximum(pixel_results['backscattering'], 0)  # bbp must be >= 0
            reference_band[valid_mask] = pixel_results['reference_band']
            tss_concentration[valid_mask] = pixel_results['tss']

            # Clamp TSS to physically valid range
            tss_min, tss_max = self.config.tss_valid_range
            tss_concentration[valid_mask] = np.clip(tss_concentration[valid_mask], tss_min, tss_max)

            ref_bands = pixel_results['reference_band']

            water_type_classification[valid_mask] = np.select(
                [
                    ref_bands == 560,
                    ref_bands == 665,
                    ref_bands == 740,
                    ref_bands == 865
                ],
                [1, 2, 3, 4],
                default=0
            )

        # Calculate statistics
        valid_pixels = np.sum(valid_mask)
        total_pixels = shape[0] * shape[1]
        coverage_percent = (valid_pixels / total_pixels) * 100

        # Log water type distribution summary
        water_type_summary = []
        if valid_pixels > 0:
            ref_bands_valid = reference_band[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]

            if len(ref_bands_valid) > 0:
                type_names = {560: "I", 665: "II", 740: "III", 865: "IV"}
                for band in [560, 665, 740, 865]:
                    count = np.sum(ref_bands_valid == band)
                    if count > 0:
                        percentage = (count / len(ref_bands_valid)) * 100
                        water_type_summary.append(f"Type {type_names[band]}:{percentage:.0f}%")

        logger.info(f"    TSS: {coverage_percent:.1f}% coverage ({' '.join(water_type_summary)})")

        return {
            'absorption': absorption,
            'backscattering': backscattering,
            'reference_band': reference_band,
            'tss': tss_concentration,
            'valid_mask': valid_mask,
            'water_type_classification': water_type_classification
        }

    def _create_valid_pixel_mask(self, rrs_data: Dict[int, np.ndarray]) -> np.ndarray:
        """Validation matching R algorithm requirements"""
        required_bands = [490, 560, 665, 740]
        valid_mask = np.ones(rrs_data[443].shape, dtype=bool)

        for band in required_bands:
            if band in rrs_data:
                band_valid = (~np.isnan(rrs_data[band])) & (rrs_data[band] > 0)
                valid_mask &= band_valid
            else:
                valid_mask[:] = False
                break

        return valid_mask

    def _process_valid_pixels(self, rrs_data: Dict[int, np.ndarray],
                              valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized pixel processing using exact Jiang et al. (2021) algorithm.

        Replaces pixel-by-pixel loop with NumPy array operations.
        Mathematically identical to _estimate_tss_single_pixel but ~100-1000x faster.
        """
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING
        tsf = self.constants.TSS_CONVERSION_FACTORS

        # Extract valid pixel arrays
        vp = {}
        for wavelength, data in rrs_data.items():
            vp[wavelength] = data[valid_mask]

        n_pixels = len(vp[443])
        logger.debug(f"Processing {n_pixels} valid pixels with vectorized Jiang algorithm")

        # Step 1: Below-water rrs conversion for all wavelengths
        # rrs = Rrs / (0.52 + 1.7 * Rrs)  — Lee et al. (2002)
        rrs = {}
        for wl, Rrs_val in vp.items():
            rrs[wl] = Rrs_val / (0.52 + 1.7 * Rrs_val)

        # Step 2: Compute u for all wavelengths
        # u = (-g0 + sqrt(g0^2 + 4*g1*rrs)) / (2*g1)  where g0=0.089, g1=0.125
        u = {}
        for wl, rrs_val in rrs.items():
            discriminant = 0.007921 + 0.5 * rrs_val  # 0.089^2 + 4*0.125*rrs
            u_val = np.full_like(rrs_val, np.nan)
            pos = (rrs_val > 0) & (discriminant >= 0)
            u_val[pos] = (-0.089 + np.sqrt(discriminant[pos])) / 0.25
            u[wl] = u_val

        # Step 3: Estimate Rrs620 for water type classification
        coeffs = self.constants.RRS620_COEFFICIENTS
        rrs665_orig = vp[665]
        rrs620 = (coeffs['a'] * rrs665_orig**3 + coeffs['b'] * rrs665_orig**2 +
                  coeffs['c'] * rrs665_orig + coeffs['d'])

        # Step 4: Classify water types (vectorized boolean masks)
        type1 = vp[490] > vp[560]                                                  # Clear
        type2 = (~type1) & (vp[490] > rrs620)                                      # Moderate
        type4 = (~type1) & (~type2) & (vp[740] > vp[490]) & (vp[740] > 0.010)      # Extreme
        type3 = (~type1) & (~type2) & (~type4)                                      # Turbid (default)

        # Initialize output arrays
        absorption_out = np.full(n_pixels, np.nan, dtype=np.float32)
        backscattering_out = np.full(n_pixels, np.nan, dtype=np.float32)
        reference_band_out = np.full(n_pixels, np.nan, dtype=np.float32)
        tss_out = np.full(n_pixels, np.nan, dtype=np.float32)

        # Step 5: Type I (Clear Water) — 560nm reference
        if np.any(type1):
            m = type1
            numerator = rrs[443][m] + rrs[490][m]
            denominator = rrs[560][m] + 5.0 * rrs[665][m]**2 / rrs[490][m]
            ratio = numerator / denominator
            x = np.full_like(ratio, np.nan)
            valid_r = ratio > 0
            x[valid_r] = np.log10(ratio[valid_r])
            a560 = aw[560] + 10.0**(-1.146 - 1.366 * x - 0.469 * x**2)
            denom_u = 1.0 - u[560][m]
            safe = np.abs(denom_u) > 1e-10
            bbp = np.full_like(a560, np.nan)
            bbp[safe] = (u[560][m][safe] * a560[safe]) / denom_u[safe] - bbw[560]
            absorption_out[m] = a560
            backscattering_out[m] = bbp
            reference_band_out[m] = 560
            tss_out[m] = tsf[560] * bbp

        # Step 6: Type II (Moderately Turbid) — 665nm reference
        if np.any(type2):
            m = type2
            ratio = vp[665][m] / (vp[443][m] + vp[490][m])
            a665 = aw[665] + 0.39 * np.power(ratio, 1.14)
            denom_u = 1.0 - u[665][m]
            safe = np.abs(denom_u) > 1e-10
            bbp = np.full_like(a665, np.nan)
            bbp[safe] = (u[665][m][safe] * a665[safe]) / denom_u[safe] - bbw[665]
            absorption_out[m] = a665
            backscattering_out[m] = bbp
            reference_band_out[m] = 665
            tss_out[m] = tsf[665] * bbp

        # Step 7: Type III (Highly Turbid) — 740nm reference
        if np.any(type3):
            m = type3
            denom_u = 1.0 - u[740][m]
            safe = np.abs(denom_u) > 1e-10
            bbp = np.full_like(denom_u, np.nan)
            bbp[safe] = (u[740][m][safe] * aw[740]) / denom_u[safe] - bbw[740]
            absorption_out[m] = aw[740]
            backscattering_out[m] = bbp
            reference_band_out[m] = 740
            tss_out[m] = tsf[740] * bbp

        # Step 8: Type IV (Extremely Turbid) — 865nm reference
        if np.any(type4):
            m = type4
            denom_u = 1.0 - u[865][m]
            safe = np.abs(denom_u) > 1e-10
            bbp = np.full_like(denom_u, np.nan)
            bbp[safe] = (u[865][m][safe] * aw[865]) / denom_u[safe] - bbw[865]
            absorption_out[m] = aw[865]
            backscattering_out[m] = bbp
            reference_band_out[m] = 865
            tss_out[m] = tsf[865] * bbp

        return {
            'absorption': absorption_out,
            'backscattering': backscattering_out,
            'reference_band': reference_band_out,
            'tss': tss_out
        }

    def _estimate_tss_single_pixel(self, pixel_rrs: Dict[int, float]) -> Optional[Dict]:
        """EXACT R implementation: Estimate_TSS_Jiang_MSI"""
        try:
            required_bands = [490, 560, 665, 740]

            if all(v == 0 for v in pixel_rrs.values()):
                return None

            if all(np.isnan(v) for v in pixel_rrs.values()):
                return None

            if any(np.isnan(pixel_rrs.get(band, np.nan)) for band in required_bands):
                return None

            rrs620 = self._estimate_rrs620_from_rrs665(pixel_rrs[665])

            if pixel_rrs[490] > pixel_rrs[560]:
                result = self._qaa_type1_clear_560nm(pixel_rrs)
            elif pixel_rrs[490] > rrs620:
                result = self._qaa_type2_moderate_665nm(pixel_rrs)
            elif pixel_rrs[740] > pixel_rrs[490] and pixel_rrs[740] > 0.010:
                result = self._qaa_type4_extreme_865nm(pixel_rrs)
            else:
                result = self._qaa_type3_turbid_740nm(pixel_rrs)

            return result

        except Exception as e:
            logger.debug(f"Error processing pixel with Jiang algorithm: {e}")
            return None

    def _estimate_rrs620_from_rrs665(self, rrs665: float) -> float:
        """EXACT R implementation: estimate_Rrs620"""
        coeffs = self.constants.RRS620_COEFFICIENTS
        a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
        rrs620 = a * (rrs665**3) + b * (rrs665**2) + c * rrs665 + d
        return rrs620

    def _qaa_type1_clear_560nm(self, site_rrs: Dict[int, float]) -> Dict:
        """QAA algorithm for Type I (clear) water using 560nm reference band."""
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING

        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)

        u = {}
        for wl, rrs_val in rrs.items():
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan

        numerator = rrs[443] + rrs[490]
        denominator = rrs[560] + 5 * rrs[665] * rrs[665] / rrs[490]
        x = np.log10(numerator / denominator)

        a560 = aw[560] + 10**(-1.146 - 1.366*x - 0.469*(x**2))
        bbp560 = ((u[560] * a560) / (1 - u[560])) - bbw[560]
        tss = self.constants.TSS_CONVERSION_FACTORS[560] * bbp560

        return {'a': a560, 'bbp': bbp560, 'band': 560, 'tss': tss}

    def _qaa_type2_moderate_665nm(self, site_rrs: Dict[int, float]) -> Dict:
        """QAA algorithm for Type II (moderately turbid) water using 665nm reference band."""
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING

        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)

        u = {}
        for wl, rrs_val in rrs.items():
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan

        ratio = site_rrs[665] / (site_rrs[443] + site_rrs[490])
        a665 = aw[665] + 0.39 * (ratio**1.14)
        bbp665 = ((u[665] * a665) / (1 - u[665])) - bbw[665]
        tss = self.constants.TSS_CONVERSION_FACTORS[665] * bbp665

        return {'a': a665, 'bbp': bbp665, 'band': 665, 'tss': tss}

    def _qaa_type3_turbid_740nm(self, site_rrs: Dict[int, float]) -> Dict:
        """QAA algorithm for Type III (highly turbid) water using 740nm reference band."""
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING

        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)

        u = {}
        for wl, rrs_val in rrs.items():
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan

        bbp740 = ((u[740] * aw[740]) / (1 - u[740])) - bbw[740]
        tss = self.constants.TSS_CONVERSION_FACTORS[740] * bbp740

        return {'a': aw[740], 'bbp': bbp740, 'band': 740, 'tss': tss}

    def _qaa_type4_extreme_865nm(self, site_rrs: Dict[int, float]) -> Dict:
        """QAA algorithm for Type IV (extremely turbid) water using 865nm reference band."""
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING

        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)

        u = {}
        for wl, rrs_val in rrs.items():
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan

        bbp865 = ((u[865] * aw[865]) / (1 - u[865])) - bbw[865]
        tss = self.constants.TSS_CONVERSION_FACTORS[865] * bbp865

        return {'a': aw[865], 'bbp': bbp865, 'band': 865, 'tss': tss}

    def _save_tss_products(self, results: Dict[str, np.ndarray], output_folder: str,
                           product_name: str, reference_metadata: Dict,
                           water_mask: Optional[np.ndarray] = None) -> Dict[str, ProcessingResult]:
        """Save TSS products including core Jiang, water types, and advanced algorithms.

        Args:
            results: Dictionary of product name to numpy array
            output_folder: Base output folder path
            product_name: Scene/product identifier
            reference_metadata: Georeference metadata for output files
            water_mask: Optional boolean mask (True=water, False=land) to apply to products

        Returns:
            Dictionary of ProcessingResult objects
        """
        try:
            output_results = {}

            # Log if water masking is being applied
            if water_mask is not None:
                logger.debug("Applying NDWI water mask to TSS products")

            # Extract clean scene name using OutputStructure helper
            scene_name = OutputStructure.extract_clean_scene_name(product_name)

            # Create scene-based output structure
            scene_folder = OutputStructure.get_scene_folder(output_folder, scene_name)
            tss_folder = OutputStructure.get_category_folder(scene_folder, OutputStructure.TSS_FOLDER)

            # Core Jiang products - gated by output_categories.enable_tss
            if not self.config.output_categories.enable_tss:
                logger.debug("TSS output category disabled - skipping core TSS products")

            jiang_products = {} if not self.config.output_categories.enable_tss else {
                'absorption': {
                    'data': results.get('absorption'),
                    'filename': f"{scene_name}_Absorption.tif",
                    'description': "Absorption coefficient (m^-1) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'backscattering': {
                    'data': results.get('backscattering'),
                    'filename': f"{scene_name}_Backscattering.tif",
                    'description': "Particulate backscattering coefficient (m^-1) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'reference_band': {
                    'data': results.get('reference_band'),
                    'filename': f"{scene_name}_ReferenceBand.tif",
                    'description': "Reference wavelength used (nm) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'tss': {
                    'data': results.get('tss'),
                    'filename': f"{scene_name}_TSS.tif",
                    'description': "Total Suspended Solids (g/m3) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'water_type_classification': {
                    'data': results.get('water_type_classification'),
                    'filename': f"{scene_name}_WaterTypes.tif",
                    'description': "Water Type Classification (0=Invalid, 1=Clear, 2=Moderate, 3=Highly turbid, 4=Extremely turbid)",
                    'folder': tss_folder
                },
                'valid_mask': {
                    'data': results.get('valid_mask'),
                    'filename': f"{scene_name}_ValidMask.tif",
                    'description': "Valid pixel mask - Jiang processing",
                    'folder': tss_folder
                }
            }

            # Advanced algorithm products - routed to scene-level category folders
            advanced_products = {}
            for key, data in results.items():
                if key.startswith('advanced_'):
                    if 'clarity' in key or 'secchi' in key or 'euphotic' in key or 'kd' in key:
                        category = OutputStructure.WATER_CLARITY_FOLDER
                        description = "Water clarity analysis"
                    elif 'hab' in key or 'cyano' in key or 'bloom' in key:
                        category = OutputStructure.HAB_FOLDER
                        description = "Harmful Algal Bloom detection"
                    elif 'tsi' in key or 'trophic' in key:
                        category = OutputStructure.TROPHIC_STATE_FOLDER
                        description = "Trophic state assessment"
                    else:
                        category = OutputStructure.TSS_FOLDER
                        description = "Advanced algorithm product"

                    subfolder = OutputStructure.get_category_folder(scene_folder, category)

                    # Create clean product name (e.g., secchi_depth -> SecchiDepth)
                    clean_key = key.replace('advanced_', '')
                    product_type = ''.join(word.title() for word in clean_key.split('_'))
                    filename = f"{scene_name}_{product_type}.tif"

                    advanced_products[key] = {
                        'data': data,
                        'filename': filename,
                        'description': description,
                        'folder': subfolder
                    }

            # NOTE: RGB composites and spectral indices are now handled by visualization_processor.py
            # They are saved directly there with proper band naming
            rgb_products = {}
            spectral_products = {}

            all_products = {**jiang_products, **advanced_products, **rgb_products, **spectral_products}

            logger.debug(f"Saving {len(all_products)} products: {len(jiang_products)} core + {len(advanced_products)} advanced")

            # Classification products that need uint8
            classification_product_keys = [
                'hab_risk_level', 'reference_band', 'valid_mask', 'high_biomass_alert',
                'extreme_biomass_alert', 'ndci_bloom', 'flh_bloom', 'mci_bloom',
                'cyanobacteria_bloom', 'water_type_classification'
            ]

            saved_count = 0
            skipped_count = 0

            for product_key, product_info in all_products.items():
                if product_info['data'] is not None:
                    output_path = os.path.join(product_info['folder'], product_info['filename'])

                    # Apply water mask (set land pixels to NaN) if available
                    # Skip masking for mask products themselves
                    if water_mask is not None and 'mask' not in product_key.lower():
                        try:
                            masked_data = np.where(water_mask, product_info['data'], np.nan)
                            product_info['data'] = masked_data
                        except Exception as e:
                            logger.debug(f"Could not apply water mask to {product_key}: {e}")

                    if any(class_key in product_key for class_key in classification_product_keys):
                        nodata_value = 255
                        data_to_save = product_info['data'].copy().astype(np.float64)
                        data_to_save[np.isnan(data_to_save)] = nodata_value
                        data_to_save = np.clip(data_to_save, 0, 254)
                        original_data = product_info['data']
                        data_to_save[np.isnan(original_data)] = 255
                        data_to_save = data_to_save.astype(np.uint8)
                    else:
                        nodata_value = -9999
                        data_to_save = product_info['data'].astype(np.float32)

                    success = RasterIO.write_raster(
                        data_to_save,
                        output_path,
                        reference_metadata,
                        product_info['description'],
                        nodata=nodata_value
                    )

                    if success:
                        try:
                            stats = RasterIO.calculate_statistics(product_info['data'])
                        except Exception:
                            valid_pixels = np.sum(~np.isnan(product_info['data'])) if isinstance(product_info['data'], np.ndarray) else 0
                            total_pixels = product_info['data'].size if isinstance(product_info['data'], np.ndarray) else 1
                            stats = {
                                'coverage_percent': (valid_pixels / total_pixels) * 100,
                                'valid_pixels': valid_pixels,
                                'data_range': [0, 1]
                            }

                        logger.debug(f"Saved {product_key}: {stats.get('coverage_percent', 0):.1f}% coverage")

                        output_results[product_key] = ProcessingResult(
                            True, output_path, stats, None
                        )
                        saved_count += 1
                    else:
                        logger.error(f"Failed to save {product_key}")
                        output_results[product_key] = ProcessingResult(
                            False, output_path, None, f"Failed to write {product_key}"
                        )
                else:
                    logger.debug(f"Skipping {product_key}: no data available")
                    skipped_count += 1

            # Create water type legend
            if 'water_type_classification' in results and results['water_type_classification'] is not None:
                try:
                    self._create_water_type_legend(tss_folder, scene_name)
                except Exception as e:
                    logger.warning(f"Could not create water type legend: {e}")

            logger.debug(f"Product saving completed:")
            logger.debug(f"  Scene folder: {scene_name}/")
            logger.debug(f"  Successfully saved: {saved_count} products")
            logger.debug(f"  Skipped (no data): {skipped_count} products")

            return output_results

        except Exception as e:
            error_msg = f"Error saving complete results: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def _create_water_type_legend(self, output_folder: str, product_name: str):
        """Create a legend file for water type classification"""
        legend_file = os.path.join(output_folder, f"{product_name}_WaterTypes_Legend.txt")

        legend_content = """JIANG WATER TYPE CLASSIFICATION LEGEND
======================================

Value | Water Type          | Algorithm Used | Characteristics
------|--------------------|--------------  |----------------
0   | Invalid/Land       | N/A            | No valid data or land pixels
1   | Clear Water        | QAA-560        | Low turbidity, high transparency
2   | Moderately Turbid  | QAA-665        | Moderate suspended matter
3   | Highly Turbid      | QAA-740        | High suspended matter concentration
4   | Extremely Turbid   | QAA-865        | Very high turbidity, possible algal blooms

ALGORITHM SELECTION CRITERIA (Jiang et al. 2021):
================================================

Type I (Clear Water - 560nm):
- Condition: Rrs(490) > Rrs(560)
- Typical TSS: < 2 g/m3
- Water clarity: High (Secchi depth > 10m)

Type II (Moderately Turbid - 665nm):
- Condition: Rrs(490) > Rrs(620) AND Rrs(490) <= Rrs(560)
- Typical TSS: 2-10 g/m3
- Water clarity: Moderate (Secchi depth 3-10m)

Type III (Highly Turbid - 740nm):
- Condition: Rrs(740) <= Rrs(490) OR Rrs(740) <= 0.010
- Typical TSS: 10-50 g/m3
- Water clarity: Low (Secchi depth 1-3m)

Type IV (Extremely Turbid - 865nm):
- Condition: Rrs(740) > Rrs(490) AND Rrs(740) > 0.010
- Typical TSS: > 50 g/m3
- Water clarity: Very low (Secchi depth < 1m)

Reference: Jiang, D., et al. (2021). Remote Sensing of Environment, 258, 112386.
DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

        try:
            with open(legend_file, 'w', encoding='utf-8') as f:
                f.write(legend_content)
            logger.debug(f"Water type legend created: {os.path.basename(legend_file)}")
        except Exception as e:
            logger.warning(f"Could not create water type legend: {e}")

    def _create_product_index(self, output_results: Dict[str, ProcessingResult],
                            output_folder: str, product_name: str):
        """Create an index file listing all generated products by scanning output folders"""
        try:
            index_file = os.path.join(output_folder, f"{product_name}_ProductIndex.txt")

            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(f"SENTINEL-2 TSS PROCESSING RESULTS\n")
                f.write(f"{'='*50}\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing v2.0\n\n")

                total_count = 0

                # CORE TSS PRODUCTS (from TSS subfolder)
                tss_folder = os.path.join(output_folder, 'TSS')
                if os.path.exists(tss_folder):
                    tss_files = sorted(glob.glob(os.path.join(tss_folder, '*.tif')))
                    if tss_files:
                        f.write(f"CORE TSS PRODUCTS:\n")
                        f.write(f"{'-'*20}\n")
                        for filepath in tss_files:
                            f.write(f"+ {os.path.basename(filepath)}\n")
                            total_count += 1

                # Additional output categories
                for cat_name, cat_label in [('RGB', 'RGB COMPOSITES'),
                                             ('Indices', 'SPECTRAL INDICES'),
                                             ('WaterClarity', 'WATER CLARITY PRODUCTS'),
                                             ('HAB', 'HAB PRODUCTS'),
                                             ('TrophicState', 'TROPHIC STATE PRODUCTS')]:
                    cat_folder = os.path.join(output_folder, cat_name)
                    if os.path.exists(cat_folder):
                        cat_files = sorted(glob.glob(os.path.join(cat_folder, '*.tif')))
                        if cat_files:
                            f.write(f"\n{cat_label}:\n")
                            f.write(f"{'-'*30}\n")
                            for filepath in cat_files:
                                f.write(f"+ {os.path.basename(filepath)}\n")
                                total_count += 1

                f.write(f"\nTotal products generated: {total_count}\n")

            logger.debug(f"Product index created: {os.path.basename(index_file)}")

        except Exception as e:
            logger.warning(f"Could not create product index: {e}")

    def _log_processing_summary(self, results: Dict[str, np.ndarray], product_name: str):
        """Log comprehensive processing summary"""
        tss_data = results.get('tss')
        if tss_data is None:
            return

        reference_bands = results.get('reference_band')
        valid_mask = results.get('valid_mask')

        tss_stats = RasterIO.calculate_statistics(tss_data)

        logger.debug(f"=== FULL JIANG TSS PROCESSING SUMMARY: {product_name} ===")
        logger.debug(f"Total coverage: {tss_stats['coverage_percent']:.1f}%")
        logger.debug(f"TSS range: {tss_stats['min']:.2f} - {tss_stats['max']:.2f} g/m3")
        logger.debug(f"TSS mean: {tss_stats['mean']:.2f} g/m3")

        if valid_mask is not None and np.any(valid_mask):
            ref_bands_valid = reference_bands[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]

            if len(ref_bands_valid) > 0:
                logger.debug("Water type classification results:")
                for band in [560, 665, 740, 865]:
                    count = np.sum(ref_bands_valid == band)
                    percentage = (count / len(ref_bands_valid)) * 100
                    if count > 0:
                        water_type = {
                            560: "Type I (Clear)",
                            665: "Type II (Moderately turbid)",
                            740: "Type III (Highly turbid)",
                            865: "Type IV (Extremely turbid)"
                        }[band]
                        logger.debug(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")

        logger.debug("=" * 60)
