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
import logging
import warnings
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import JiangTSSConfig, WaterQualityConfig, MarineVisualizationConfig
from ..utils.raster_io import RasterIO
from .snap_calculator import ProcessingResult
from .water_quality_processor import WaterQualityProcessor
from .marine_viz import VisualizationProcessor

logger = logging.getLogger('sentinel2_tss_pipeline')


class JiangTSSConstants:
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


class JiangTSSProcessor:
    """Complete implementation of Jiang et al. 2021 TSS methodology"""

    def __init__(self, config: JiangTSSConfig):
        """Initialize Jiang TSS Processor with configuration and marine visualization"""
        self.config = config
        self.constants = JiangTSSConstants()

        # Initialize marine visualization processor
        if hasattr(config, 'enable_marine_visualization') and config.enable_marine_visualization:
            self.marine_viz_processor = VisualizationProcessor(config.marine_viz_config)
            logger.info("Marine visualization processor initialized")
        else:
            self.marine_viz_processor = None
            logger.info("Marine visualization disabled")

        # Initialize advanced processor if enabled
        if self.config.enable_advanced_algorithms:
            self.advanced_processor = WaterQualityProcessor()
            if self.config.water_quality_config is None:
                self.config.water_quality_config = WaterQualityConfig()
        else:
            self.advanced_processor = None

        logger.info("Initialized Jiang TSS Processor with enhanced methodology")
        logger.info(f"Jiang TSS enabled: {self.config.enable_jiang_tss}")
        logger.info(f"Advanced algorithms enabled: {self.config.enable_advanced_algorithms}")
        logger.info(f"Marine visualization enabled: {getattr(self.config, 'enable_marine_visualization', False)}")

        if self.config.enable_advanced_algorithms and self.config.water_quality_config:
            logger.info("Working algorithms available:")
            logger.info(f"  Water Clarity: {self.config.water_quality_config.enable_water_clarity}")
            logger.info(f"  HAB Detection: {self.config.water_quality_config.enable_hab_detection}")

        logger.info("Jiang TSS Processor initialization completed successfully")

    def _load_bands_data(self, band_paths: Dict[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load band data arrays from file paths"""
        bands_data = {}
        reference_metadata = None

        logger.info(f"Loading {len(band_paths)} spectral bands into memory")

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

        logger.info(f"Successfully loaded {len(bands_data)} bands into memory")
        return bands_data, reference_metadata

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

        logger.info(f"Detected band types: {list(band_types)}")

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

        # Log all conversions applied
        if len(band_types) > 1:
            logger.info("Mixed band types - individual conversions:")
        elif 'rrs' in band_types:
            logger.info("All rrs bands - no conversion needed:")
        elif 'rhow' in band_types:
            logger.info("All rhow bands - converting to Rrs:")

        for log_entry in conversion_log:
            logger.info(f"  {log_entry}")

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

        logger.info(f"Searching for COMPLETE 8-band datasets in: {data_folder}")

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
                    logger.info(f"Found {wavelength}nm: {name} ({band_type})")
                    break
            else:
                logger.warning(f"Missing {wavelength}nm - CRITICAL for Jiang algorithm")

        logger.info(f"Band type summary: {band_type_summary}")

        total_found = len(found_bands)
        required_bands = [443, 490, 560, 665, 705, 740, 783, 865]
        missing_critical = [wl for wl in required_bands if wl not in found_bands]

        if total_found == 8 and not missing_critical:
            logger.info("PERFECT: Complete 8-band dataset found - Jiang algorithm ready")
        elif total_found >= 6 and 783 in found_bands and 865 in found_bands:
            logger.info(f"USABLE: {total_found}/8 bands found including critical NIR bands")
        else:
            logger.error(f"INSUFFICIENT: {total_found}/8 bands found")
            if missing_critical:
                logger.error(f"Missing CRITICAL bands: {missing_critical}nm")

        return found_bands

    def process_jiang_tss(self, c2rcc_path: str, output_folder: str, product_name: str,
                        intermediate_paths: Optional[Dict[str, str]] = None) -> Dict[str, ProcessingResult]:
        """Process Jiang TSS methodology from C2RCC outputs"""
        try:
            logger.info(f"Starting Jiang TSS processing for: {product_name}")

            # Extract georeference from C2RCC output
            try:
                data_folder = c2rcc_path.replace('.dim', '.data')
                sample_band_path = os.path.join(data_folder, 'rrs_B4.img')

                if os.path.exists(sample_band_path):
                    _, reference_metadata = RasterIO.read_raster(sample_band_path)
                    logger.info("Using C2RCC georeference for proper geographic positioning")
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

            # Step 1: Load spectral bands directly
            logger.info("Step 1: Loading spectral bands from C2RCC output")
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
                            logger.info(f"Loaded {wavelength}nm: {filename}")
                            break
                        except Exception as e:
                            logger.error(f"Failed to load {wavelength}nm from {filename}: {e}")

            if not bands_data:
                error_msg = "No spectral bands found in C2RCC output"
                logger.error(error_msg)
                return {'error': ProcessingResult(False, "", None, error_msg)}

            # Step 3: Apply unit conversion
            logger.info("Step 3: Converting rhow to Rrs")
            converted_bands_data = self._convert_rhow_to_rrs(bands_data, band_paths)

            # Step 4: Apply Jiang methodology
            logger.info("Step 4: Applying Jiang TSS methodology")
            jiang_results = self._estimate_tss_all_pixels(converted_bands_data)

            # Step 5: Process advanced algorithms
            advanced_results = {}
            if (hasattr(self.config, 'enable_advanced_algorithms') and
                self.config.enable_advanced_algorithms and
                hasattr(self, 'advanced_processor') and
                self.advanced_processor is not None):

                logger.info("Step 5: Processing advanced algorithms")
                advanced_results = self._process_water_quality_products(
                    c2rcc_path, jiang_results, converted_bands_data, product_name
                )

                if advanced_results is None:
                    advanced_results = {}

                logger.info(f"Advanced algorithms completed: {len(advanced_results)} additional products")

            # Combine all results
            all_algorithm_results = jiang_results.copy()

            for key, result in advanced_results.items():
                if isinstance(result, ProcessingResult) and result.stats and 'numpy_data' in result.stats:
                    all_algorithm_results[key] = result.stats['numpy_data']

            # Step 6: Save complete results
            logger.info("Step 6: Saving complete results including advanced algorithms")
            saved_results = self._save_tss_products(all_algorithm_results, output_folder,
                                                   product_name, reference_metadata)

            # Update ProcessingResult objects with actual file paths
            final_results = saved_results.copy()

            for key, advanced_result in advanced_results.items():
                if key in saved_results:
                    advanced_result.output_path = saved_results[key].output_path
                    final_results[key] = advanced_result

            # Step 7: Marine visualization processing
            if (hasattr(self.config, 'enable_marine_visualization') and
                self.config.enable_marine_visualization and
                hasattr(self, 'marine_viz_processor') and
                self.marine_viz_processor is not None):

                logger.info("Step 7: Processing marine visualizations")

                try:
                    results_folder = os.path.dirname(output_folder)
                    geometric_folder = os.path.join(results_folder, "Geometric_Products")

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
                        logger.info("Geometric products found for marine visualization")

                        viz_results = self.marine_viz_processor.process_marine_visualizations(
                            c2rcc_path, output_folder, product_name, intermediate_paths
                        )

                        final_results.update(viz_results)

                        rgb_count = len([k for k in viz_results.keys() if k.startswith('rgb_')])
                        index_count = len([k for k in viz_results.keys() if k.startswith('index_')])
                        logger.info(f"Marine visualization completed: {rgb_count} RGB + {index_count} indices")

                        # Cleanup intermediate products
                        logger.info("Starting intermediate products cleanup...")
                        try:
                            cleanup_success = self.marine_viz_processor._cleanup_intermediate_products(
                                results_folder, product_name)
                            if cleanup_success:
                                logger.info("Geometric products cleanup completed successfully")
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

            logger.info("=" * 80)
            logger.info(f"COMPLETE TSS PROCESSING FINISHED: {product_name}")
            logger.info(f"   Total products generated: {success_count}/{total_count}")
            logger.info(f"   Success rate: {(success_count/total_count)*100:.1f}%")
            logger.info("=" * 80)

            return final_results

        except Exception as e:
            error_msg = f"Jiang TSS processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def _process_water_quality_products(self, c2rcc_path: str, jiang_results: Dict,
                                        rrs_bands_data: Dict, product_name: str) -> Dict[str, ProcessingResult]:
        """Process water quality products (clarity, HAB) with converted Rrs data."""
        try:
            logger.info("Processing advanced algorithms with unit-converted data")
            advanced_results = {}

            config = self.config.water_quality_config

            if config is None:
                logger.warning("No advanced config available, using defaults")
                config = WaterQualityConfig()

            # Water clarity calculation
            if config.enable_water_clarity and 'absorption' in jiang_results and 'backscattering' in jiang_results:
                logger.info("Calculating water clarity indices")

                absorption = jiang_results['absorption']
                backscattering = jiang_results['backscattering']

                try:
                    clarity_results = self.advanced_processor.calculate_water_clarity(
                        absorption, backscattering, config.solar_zenith_angle
                    )

                    for key, value in clarity_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            stats = RasterIO.calculate_statistics(value)
                            stats['numpy_data'] = value
                            stats['product_type'] = 'water_clarity'
                            stats['algorithm'] = 'advanced_aquatic'
                            stats['description'] = f"Water clarity {key} - Advanced Aquatic Processing"
                            stats['processing_date'] = datetime.now().isoformat()

                            advanced_results[f'advanced_clarity_{key}'] = ProcessingResult(
                                success=True,
                                output_path="",
                                stats=stats,
                                error_message=None
                            )

                    logger.info(f"Water clarity calculation completed: {len(clarity_results)} products")

                except Exception as e:
                    logger.error(f"Water clarity calculation failed: {e}")

            # HAB detection
            if config.enable_hab_detection and rrs_bands_data:
                logger.info("Detecting harmful algal blooms using converted Rrs data")

                try:
                    hab_results = self.advanced_processor.detect_harmful_algal_blooms(
                        chlorophyll=None,
                        phycocyanin=None,
                        rrs_bands=rrs_bands_data
                    )

                    for key, value in hab_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            stats = RasterIO.calculate_statistics(value)
                            stats['numpy_data'] = value
                            stats['product_type'] = 'hab_detection'
                            stats['algorithm'] = 'advanced_aquatic'
                            stats['description'] = f"HAB {key} - Advanced Aquatic Processing"
                            stats['processing_date'] = datetime.now().isoformat()
                            stats['biomass_threshold'] = config.hab_biomass_threshold
                            stats['extreme_threshold'] = config.hab_extreme_threshold

                            advanced_results[f'advanced_hab_{key}'] = ProcessingResult(
                                success=True,
                                output_path="",
                                stats=stats,
                                error_message=None
                            )

                    logger.info(f"HAB detection completed: {len(hab_results)} products")

                except Exception as e:
                    logger.error(f"HAB detection failed: {e}")
            else:
                logger.warning("No suitable converted spectral bands available for HAB detection")

            logger.info(f"Advanced algorithms completed: {len(advanced_results)} products generated")
            return advanced_results

        except Exception as e:
            logger.error(f"Error in advanced algorithms processing: {e}")
            traceback.print_exc()
            return {}

    def _extract_snap_chlorophyll(self, c2rcc_path: str) -> Optional[np.ndarray]:
        """Extract chlorophyll from SNAP C2RCC output"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            chl_path = os.path.join(data_folder, 'conc_chl.img')

            if os.path.exists(chl_path):
                chl_data, _ = RasterIO.read_raster(chl_path)
                logger.info("Successfully extracted SNAP chlorophyll data")
                return chl_data
            else:
                logger.warning("SNAP chlorophyll data not found")
                return None

        except Exception as e:
            logger.error(f"Error extracting SNAP chlorophyll: {e}")
            return None

    def _estimate_tss_all_pixels(self, rrs_data: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """Estimate TSS for all valid pixels using Jiang methodology."""
        logger.info("Applying Jiang methodology to pre-converted Rrs data")

        shape = rrs_data[443].shape

        # Initialize output arrays
        absorption = np.full(shape, np.nan, dtype=np.float32)
        backscattering = np.full(shape, np.nan, dtype=np.float32)
        reference_band = np.full(shape, np.nan, dtype=np.float32)
        tss_concentration = np.full(shape, np.nan, dtype=np.float32)
        water_type_classification = np.full(shape, 0, dtype=np.uint8)

        logger.info("Using pre-converted Rrs data (no additional pi division)")

        # Create valid pixel mask
        valid_mask = self._create_valid_pixel_mask(rrs_data)

        if np.any(valid_mask):
            pixel_results = self._process_valid_pixels(rrs_data, valid_mask)

            absorption[valid_mask] = pixel_results['absorption']
            backscattering[valid_mask] = pixel_results['backscattering']
            reference_band[valid_mask] = pixel_results['reference_band']
            tss_concentration[valid_mask] = pixel_results['tss']

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

        logger.info(f"Jiang processing completed:")
        logger.info(f"  Valid pixels: {valid_pixels}/{total_pixels} ({coverage_percent:.1f}%)")

        # Log water type distribution
        if valid_pixels > 0:
            ref_bands_valid = reference_band[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]

            if len(ref_bands_valid) > 0:
                logger.info("Water type distribution:")
                for band in [560, 665, 740, 865]:
                    count = np.sum(ref_bands_valid == band)
                    if count > 0:
                        percentage = (count / len(ref_bands_valid)) * 100
                        water_type = {
                            560: "Type I (Clear water)",
                            665: "Type II (Moderately turbid)",
                            740: "Type III (Highly turbid)",
                            865: "Type IV (Extremely turbid)"
                        }[band]
                        logger.info(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")

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
        """Pixel processing using exact R algorithm"""
        valid_pixels = {}
        for wavelength, data in rrs_data.items():
            valid_pixels[wavelength] = data[valid_mask]

        n_pixels = len(valid_pixels[443])
        logger.info(f"Processing {n_pixels} valid pixels with corrected Jiang algorithm")

        absorption_out = np.full(n_pixels, np.nan, dtype=np.float32)
        backscattering_out = np.full(n_pixels, np.nan, dtype=np.float32)
        reference_band_out = np.full(n_pixels, np.nan, dtype=np.float32)
        tss_out = np.full(n_pixels, np.nan, dtype=np.float32)

        for i in range(n_pixels):
            pixel_rrs = {wl: valid_pixels[wl][i] for wl in valid_pixels.keys()}
            result = self._estimate_tss_single_pixel(pixel_rrs)

            if result is not None:
                absorption_out[i] = result['a']
                backscattering_out[i] = result['bbp']
                reference_band_out[i] = result['band']
                tss_out[i] = result['tss']

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
                           product_name: str, reference_metadata: Dict) -> Dict[str, ProcessingResult]:
        """Save TSS products including core Jiang, water types, and advanced algorithms."""
        try:
            output_results = {}
            clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')

            # Create main output structure
            scene_folder = os.path.join(output_folder, clean_product_name)
            tss_folder = os.path.join(scene_folder, "TSS_Products")
            advanced_folder = os.path.join(scene_folder, "Advanced_Products")
            rgb_folder = os.path.join(scene_folder, "RGB_Composites")
            spectral_folder = os.path.join(scene_folder, "Spectral_Indices")

            os.makedirs(tss_folder, exist_ok=True)
            os.makedirs(advanced_folder, exist_ok=True)
            os.makedirs(rgb_folder, exist_ok=True)
            os.makedirs(spectral_folder, exist_ok=True)

            # Core Jiang products
            jiang_products = {
                'absorption': {
                    'data': results.get('absorption'),
                    'filename': f"{clean_product_name}_Jiang_Absorption.tif",
                    'description': "Absorption coefficient (m^-1) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'backscattering': {
                    'data': results.get('backscattering'),
                    'filename': f"{clean_product_name}_Jiang_Backscattering.tif",
                    'description': "Particulate backscattering coefficient (m^-1) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'reference_band': {
                    'data': results.get('reference_band'),
                    'filename': f"{clean_product_name}_Jiang_ReferenceBand.tif",
                    'description': "Reference wavelength used (nm) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'tss': {
                    'data': results.get('tss'),
                    'filename': f"{clean_product_name}_Jiang_TSS.tif",
                    'description': "Total Suspended Solids (g/m3) - Jiang et al. 2021",
                    'folder': tss_folder
                },
                'water_type_classification': {
                    'data': results.get('water_type_classification'),
                    'filename': f"{clean_product_name}_Jiang_WaterTypes.tif",
                    'description': "Water Type Classification (0=Invalid, 1=Clear, 2=Moderate, 3=Highly turbid, 4=Extremely turbid)",
                    'folder': tss_folder
                },
                'valid_mask': {
                    'data': results.get('valid_mask'),
                    'filename': f"{clean_product_name}_Jiang_ValidMask.tif",
                    'description': "Valid pixel mask - Jiang processing",
                    'folder': tss_folder
                }
            }

            # Advanced algorithm products
            advanced_products = {}
            for key, data in results.items():
                if key.startswith('advanced_'):
                    if 'clarity' in key:
                        category = 'WaterClarity'
                        description = "Water clarity analysis - Advanced Aquatic Processing"
                    elif 'hab' in key:
                        category = 'HAB'
                        description = "Harmful Algal Bloom detection - Advanced Aquatic Processing"
                    elif 'tsi' in key:
                        category = 'TrophicState'
                        description = "Trophic state assessment - Advanced Aquatic Processing"
                    else:
                        category = 'General'
                        description = "Advanced algorithm product"

                    clean_key = key.replace('advanced_', '').replace('_', '').title()
                    filename = f"{clean_product_name}_Advanced_{category}_{clean_key}.tif"

                    advanced_products[key] = {
                        'data': data,
                        'filename': filename,
                        'description': description,
                        'folder': advanced_folder
                    }

            # RGB composites and spectral indices
            rgb_products = {}
            spectral_products = {}

            for key, data in results.items():
                if key.startswith('rgb_'):
                    rgb_name = key.replace('rgb_', '').replace('_', '').title()
                    rgb_products[key] = {
                        'data': data,
                        'filename': f"{clean_product_name}_RGB_{rgb_name}.tif",
                        'description': f"RGB composite - {rgb_name}",
                        'folder': rgb_folder
                    }
                elif key.startswith('index_'):
                    index_name = key.replace('index_', '').replace('_', '').upper()
                    spectral_products[key] = {
                        'data': data,
                        'filename': f"{clean_product_name}_Index_{index_name}.tif",
                        'description': f"Spectral index - {index_name}",
                        'folder': spectral_folder
                    }

            all_products = {**jiang_products, **advanced_products, **rgb_products, **spectral_products}

            logger.info(f"Saving {len(all_products)} products:")
            logger.info(f"  Core Jiang products: {len(jiang_products)}")
            logger.info(f"  Advanced products: {len(advanced_products)}")

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
                    self._create_water_type_legend(scene_folder, clean_product_name)
                except Exception as e:
                    logger.warning(f"Could not create water type legend: {e}")

            logger.info(f"Product saving completed:")
            logger.info(f"  Scene folder: {clean_product_name}/")
            logger.info(f"  Successfully saved: {saved_count} products")
            logger.info(f"  Skipped (no data): {skipped_count} products")

            # Create product index
            try:
                self._create_product_index(output_results, scene_folder, clean_product_name)
            except Exception as e:
                logger.debug(f"Could not create product index: {e}")

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
            logger.info(f"Water type legend created: {os.path.basename(legend_file)}")
        except Exception as e:
            logger.warning(f"Could not create water type legend: {e}")

    def _create_product_index(self, output_results: Dict[str, ProcessingResult],
                            output_folder: str, product_name: str):
        """Create an index file listing all generated products"""
        try:
            index_file = os.path.join(output_folder, f"{product_name}_ProductIndex.txt")

            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(f"SENTINEL-2 TSS PROCESSING RESULTS\n")
                f.write(f"{'='*50}\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing v2.0\n\n")

                f.write(f"CORE TSS PRODUCTS:\n")
                f.write(f"{'-'*20}\n")
                jiang_products = ['absorption', 'backscattering', 'reference_band', 'tss', 'valid_mask',
                                'water_type_classification']
                for product in jiang_products:
                    if product in output_results and output_results[product].success:
                        f.write(f"+ {os.path.basename(output_results[product].output_path)}\n")
                    else:
                        f.write(f"- {product} (failed or not generated)\n")

                f.write(f"\nADVANCED ALGORITHM PRODUCTS:\n")
                f.write(f"{'-'*30}\n")

                for key, result in output_results.items():
                    if key.startswith('advanced_') and result.success:
                        f.write(f"+ {os.path.basename(result.output_path)}\n")

                f.write(f"\nTotal products generated: {len([r for r in output_results.values() if r.success])}\n")

            logger.info(f"Product index created: {os.path.basename(index_file)}")

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

        logger.info(f"=== FULL JIANG TSS PROCESSING SUMMARY: {product_name} ===")
        logger.info(f"Total coverage: {tss_stats['coverage_percent']:.1f}%")
        logger.info(f"TSS range: {tss_stats['min']:.2f} - {tss_stats['max']:.2f} g/m3")
        logger.info(f"TSS mean: {tss_stats['mean']:.2f} g/m3")

        if valid_mask is not None and np.any(valid_mask):
            ref_bands_valid = reference_bands[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]

            if len(ref_bands_valid) > 0:
                logger.info("Water type classification results:")
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
                        logger.info(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")

        logger.info("=" * 60)
