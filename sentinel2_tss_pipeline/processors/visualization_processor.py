"""
Visualization Processor.

Generate RGB composites and spectral indices from Sentinel-2 imagery.

RGB Composites (15 unique, deduplicated):
- Natural Color (2): natural_color, natural_enhanced
- False Color (2): false_color_infrared, false_color_nir
- Water-Specific (4): turbidity_enhanced, chlorophyll_enhanced, coastal_aerosol, water_quality
- Research (7): sediment_transport, atmospheric_penetration, ocean_color_standard,
               coastal_turbidity, cdom_enhanced, water_change_detection, advanced_atmospheric

Spectral Indices (14):
- Water Quality (7): NDWI, MNDWI, NDTI, NDMI, AWEI, WI, WRI
- Chlorophyll & Algae (3): NDCI, CHL_RED_EDGE, GNDVI
- Turbidity & Sediment (2): TSI_Turbidity, NGRDI
- Advanced (2): CDOM, pSDB (Pseudo Satellite-Derived Bathymetry)

Note: PC, FAI, FUI removed - require spectral bands not available in Sentinel-2.
Note: SDD removed - WaterClarity SecchiDepth (IOP-derived) is more rigorous.

Part of the sentinel2_tss_pipeline package.
"""

import os
import time
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    try:
        import gdal
        GDAL_AVAILABLE = True
    except ImportError:
        gdal = None
        GDAL_AVAILABLE = False

from ..config.output_categories import OutputCategoryConfig
from ..utils.raster_io import RasterIO
from ..utils.output_structure import OutputStructure
from .tsm_chl_calculator import ProcessingResult

logger = logging.getLogger('sentinel2_tss_pipeline')



class VisualizationProcessor:
    """Process RGB composites and spectral indices from Sentinel-2 data."""

    def __init__(self, output_categories: Optional[OutputCategoryConfig] = None):
        """Initialize visualization processor.

        Args:
            output_categories: Controls which RGB/Index categories are generated.
                             If None, uses defaults (RGB + Indices ON).
        """
        if not GDAL_AVAILABLE:
            logger.warning("GDAL not available - visualization functionality will be limited")

        self.output_categories = output_categories or OutputCategoryConfig()
        self.logger = logger

        self._water_type = None  # Set per-product by process_marine_visualizations()

        self.logger.debug("Initialized Visualization Processor")
        self.logger.debug(f"  RGB enabled: {self.output_categories.enable_rgb}")
        self.logger.debug(f"  Indices enabled: {self.output_categories.enable_indices}")

    def process_marine_visualizations(self, c2rcc_path: str, output_folder: str,
                                    product_name: str, intermediate_paths: Optional[Dict[str, str]] = None,
                                    water_type: Optional[np.ndarray] = None) -> Dict[str, ProcessingResult]:
        """
        Generate RGB composites and spectral indices from geometric products

        Args:
            c2rcc_path: Path to C2RCC output directory (fallback only)
            output_folder: Output directory for visualizations
            product_name: Name of the product being processed
            intermediate_paths: Optional dictionary containing geometric_path
            water_type: Optional Jiang water type classification array (1-4) for pSDB band switching

        Returns:
            Dictionary of visualization results (RGB composites + spectral indices)
        """
        try:
            self.logger.debug(f"Starting marine visualization processing for {product_name}")
            self._water_type = water_type

            # Determine input source
            input_source = None
            input_type = "unknown"

            # PRIORITY 1: Use intermediate_paths geometric_path
            if intermediate_paths and 'geometric_path' in intermediate_paths:
                geometric_path = intermediate_paths['geometric_path']
                self.logger.debug(f"Checking intermediate_paths geometric_path: {geometric_path}")

                if os.path.exists(geometric_path):
                    input_source = geometric_path
                    input_type = "geometric"
                    self.logger.debug(f"Using GEOMETRIC products from intermediate_paths: {os.path.basename(geometric_path)}")
                else:
                    self.logger.warning(f"Geometric path in intermediate_paths not found: {geometric_path}")

            # PRIORITY 2: Look in Intermediate/Geometric folder
            if input_source is None:
                self.logger.debug("Trying Intermediate/Geometric folder location")
                geometric_folder = OutputStructure.get_intermediate_folder(
                    output_folder, OutputStructure.GEOMETRIC_FOLDER
                )

                if os.path.exists(geometric_folder):
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

                    if os.path.exists(geometric_path):
                        input_source = geometric_path
                        input_type = "geometric"
                        self.logger.debug(f"Using GEOMETRIC products from standard location: {geometric_filename}")
                    else:
                        for file in os.listdir(geometric_folder):
                            if file.endswith('.dim') and 'Subset' in file:
                                input_source = os.path.join(geometric_folder, file)
                                input_type = "geometric"
                                self.logger.debug(f"Found geometric product: {file}")
                                break

            # Create visualization output directories using OutputStructure helper
            scene_name = OutputStructure.extract_clean_scene_name(product_name)
            scene_folder = OutputStructure.get_scene_folder(output_folder, scene_name)
            rgb_output_dir = OutputStructure.get_category_folder(scene_folder, OutputStructure.RGB_FOLDER)
            indices_output_dir = OutputStructure.get_category_folder(scene_folder, OutputStructure.INDICES_FOLDER)

            self.logger.debug(f"Scene name: {scene_name}")
            self.logger.debug(f"RGB output directory: {rgb_output_dir}")
            self.logger.debug(f"Indices output directory: {indices_output_dir}")

            viz_results = {}

            # Load spectral bands
            self.logger.debug(f"Loading spectral bands from {input_type.upper()} products")

            try:
                if input_type == "geometric":
                    band_paths = self._load_bands_from_resampled_product(input_source)
                else:
                    band_paths = self._load_available_bands(input_source)

                if not band_paths:
                    error_msg = f"No suitable spectral bands found in {input_type} products"
                    self.logger.error(error_msg)
                    return {'error': ProcessingResult(False, "", None, error_msg)}

                self.logger.debug(f"Found {len(band_paths)} spectral bands for visualization:")
                for wavelength, path in band_paths.items():
                    self.logger.debug(f"  {wavelength}nm: {os.path.basename(path)}")

                bands_data, reference_metadata = self._load_bands_data_from_paths(band_paths)

                if bands_data is None or reference_metadata is None:
                    error_msg = "Failed to load band data arrays"
                    self.logger.error(error_msg)
                    return {'error': ProcessingResult(False, "", None, error_msg)}

                self.logger.debug(f"Successfully loaded {len(bands_data)} bands for visualization")

                sample_band = list(bands_data.values())[0]
                data_min = np.nanmin(sample_band)
                data_max = np.nanmax(sample_band)
                self.logger.debug(f"Data characteristics:")
                self.logger.debug(f"  Input type: {input_type.upper()}")
                self.logger.debug(f"  Value range: {data_min:.4f} to {data_max:.4f}")
                self.logger.debug(f"  Spatial dimensions: {sample_band.shape}")

            except Exception as band_error:
                error_msg = f"Error loading spectral bands: {str(band_error)}"
                self.logger.error(error_msg)
                import traceback
                self.logger.error(f"Band loading traceback: {traceback.format_exc()}")
                return {'error': ProcessingResult(False, "", None, error_msg)}

            # Generate RGB composites
            if bands_data and len(bands_data) >= 3:
                self.logger.debug("Generating RGB composites")

                try:
                    rgb_start = time.time()
                    rgb_results = self._generate_rgb_composites(bands_data, reference_metadata, rgb_output_dir, scene_name)
                    viz_results.update(rgb_results)
                    rgb_count = len([k for k in rgb_results.keys() if k.startswith('rgb_')])
                    rgb_elapsed = time.time() - rgb_start
                    self.logger.info(f"    RGB Composites... done ({rgb_count} products, {rgb_elapsed:.1f}s)")

                except Exception as rgb_error:
                    self.logger.error(f"Error generating RGB composites: {rgb_error}")
                    viz_results['rgb_error'] = ProcessingResult(False, "", None, str(rgb_error))

            # Generate spectral indices
            if bands_data and len(bands_data) >= 2:
                self.logger.debug("Generating spectral indices")

                try:
                    idx_start = time.time()
                    index_results = self._generate_spectral_indices(bands_data, reference_metadata, indices_output_dir, scene_name)
                    viz_results.update(index_results)
                    index_count = len([k for k in index_results.keys() if k.startswith('index_')])
                    idx_elapsed = time.time() - idx_start
                    self.logger.info(f"    Spectral Indices... done ({index_count} products, {idx_elapsed:.1f}s)")

                except Exception as index_error:
                    self.logger.error(f"Error generating spectral indices: {index_error}")
                    viz_results['index_error'] = ProcessingResult(False, "", None, str(index_error))

            # Create processing summary
            self.logger.debug("Creating visualization summary")
            try:
                self._create_visualization_summary(viz_results, output_folder, product_name)
            except Exception as summary_error:
                self.logger.warning(f"Could not create visualization summary: {summary_error}")

            # Final results
            successful_viz = len([r for r in viz_results.values() if isinstance(r, ProcessingResult) and r.success])
            total_viz = len([r for r in viz_results.values() if isinstance(r, ProcessingResult)])
            success_rate = (successful_viz / total_viz) * 100 if total_viz > 0 else 0

            self.logger.debug("Marine visualization processing completed!")
            self.logger.debug(f"   Input source: {input_type.upper()} products")
            self.logger.debug(f"   Results: {successful_viz}/{total_viz} products successful ({success_rate:.1f}%)")
            self.logger.debug(f"   RGB output: {rgb_output_dir}")
            self.logger.debug(f"   Indices output: {indices_output_dir}")

            self._water_type = None  # Clear per-product data

            return viz_results

        except Exception as e:
            error_msg = f"Marine visualization processing failed: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def _load_bands_from_resampled_product(self, geometric_path: str) -> Dict[int, str]:
        """Load bands from resampled SNAP product."""

        if geometric_path.endswith('.dim'):
            data_folder = geometric_path.replace('.dim', '.data')
        else:
            data_folder = f"{geometric_path}.data"

        if not os.path.exists(data_folder):
            self.logger.error(f"Geometric data folder not found: {data_folder}")
            return {}

        # Standard S2 band mapping
        band_mapping = {
            443: ['B1.img'],    # Coastal aerosol
            490: ['B2.img'],    # Blue
            560: ['B3.img'],    # Green
            665: ['B4.img'],    # Red
            705: ['B5.img'],    # Red Edge 1
            740: ['B6.img'],    # Red Edge 2
            783: ['B7.img'],    # Red Edge 3
            842: ['B8.img'],    # NIR (broad)
            865: ['B8A.img'],   # NIR (narrow)
            945: ['B9.img'],    # Water vapour
            1375: ['B10.img'],  # SWIR Cirrus
            1610: ['B11.img'],  # SWIR 1
            2190: ['B12.img']   # SWIR 2
        }

        available_bands = {}

        self.logger.debug(f"Loading bands from: {os.path.basename(data_folder)}")

        for wavelength, possible_files in band_mapping.items():
            for filename in possible_files:
                band_path = os.path.join(data_folder, filename)
                if os.path.exists(band_path):
                    file_size = os.path.getsize(band_path)
                    if file_size > 1024:  # Basic size check
                        available_bands[wavelength] = band_path
                        self.logger.debug(f"Found {wavelength}nm: {filename}")
                        break

        self.logger.debug(f"Found {len(available_bands)} bands: {sorted(available_bands.keys())}")
        return available_bands

    def _load_available_bands(self, c2rcc_path: str) -> Dict[int, str]:
        """Load bands from actual SNAP C2RCC output structure"""
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"

        if not os.path.exists(data_folder):
            self.logger.error(f"Data folder not found: {data_folder}")
            return {}

        band_mapping = {
            443: ['rho_toa_B1.img', 'rtoa_B1.img'],
            490: ['rho_toa_B2.img', 'rtoa_B2.img'],
            560: ['rho_toa_B3.img', 'rtoa_B3.img'],
            665: ['rho_toa_B4.img', 'rtoa_B4.img'],
            705: ['rho_toa_B5.img', 'rtoa_B5.img'],
            740: ['rho_toa_B6.img', 'rtoa_B6.img'],
            783: ['rho_toa_B7.img', 'rtoa_B7.img'],
            842: ['rho_toa_B8.img', 'rtoa_B8.img'],
            865: ['rho_toa_B8A.img', 'rtoa_B8A.img'],
            945: ['rho_toa_B9.img', 'rtoa_B9.img'],
            1610: ['rho_toa_B11.img', 'rtoa_B11.img'],
            2190: ['rho_toa_B12.img', 'rtoa_B12.img']
        }

        available_bands = {}
        found_files = []
        missing_files = []

        for wavelength, possible_files in band_mapping.items():
            band_found = False
            for filename in possible_files:
                band_path = os.path.join(data_folder, filename)
                if os.path.exists(band_path):
                    file_size = os.path.getsize(band_path)
                    if file_size > 1024:
                        available_bands[wavelength] = band_path
                        found_files.append(filename)
                        band_found = True
                        break

            if not band_found:
                missing_files.append(f"{wavelength}nm ({possible_files[0]})")

        self.logger.debug(f"Band loading results:")
        self.logger.debug(f"  Found {len(available_bands)} bands: {sorted(list(available_bands.keys()))}")
        if missing_files:
            self.logger.warning(f"  Missing: {missing_files}")

        return available_bands

    def _load_bands_data_from_paths(self, band_paths: Dict[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load band data from geometric products"""
        bands_data = {}
        reference_metadata = None

        self.logger.debug(f"Loading {len(band_paths)} bands into memory")

        for wavelength, file_path in band_paths.items():
            try:
                if not os.path.exists(file_path):
                    self.logger.error(f"Band file missing: {file_path}")
                    return None, None

                data, metadata = RasterIO.read_raster(file_path)

                if data is None or data.size == 0:
                    self.logger.error(f"Invalid data from {file_path}")
                    return None, None

                data = data.astype(np.float32)

                if metadata and 'nodata' in metadata and metadata['nodata'] is not None:
                    data[data == metadata['nodata']] = np.nan

                bands_data[wavelength] = data

                if reference_metadata is None:
                    reference_metadata = metadata

                self.logger.debug(f"Loaded {wavelength}nm: shape={data.shape}")

            except Exception as e:
                self.logger.error(f"Failed to load {wavelength}nm: {e}")
                return None, None

        self.logger.debug(f"Successfully loaded {len(bands_data)} bands")
        return bands_data, reference_metadata

    def _generate_rgb_composites(self, bands_data: Dict[int, np.ndarray], reference_metadata: Dict,
                           output_folder: str, product_name: str) -> Dict[str, ProcessingResult]:
        """Generate RGB composites based on available bands and configuration"""
        results = {}

        try:
            available_wavelengths = set(bands_data.keys())
            self.logger.debug(f"Available wavelengths for RGB: {sorted(available_wavelengths)}")

            # 15 unique RGB composites (3 duplicates removed: algal_bloom_enhanced=turbidity_enhanced,
            # deep_water_clarity=ocean_color_standard bands, riverine_waters=water_quality bands)
            rgb_enabled = self.output_categories.enable_rgb

            rgb_combinations = {
                # Natural Color (2)
                'natural_color': {
                    'red': 665, 'green': 560, 'blue': 490,
                    'description': 'Natural color (True color)',
                    'application': 'General visualization, publications',
                    'enabled': rgb_enabled
                },
                'natural_enhanced': {
                    'red': 665, 'green': 560, 'blue': 443,
                    'description': 'Natural color with enhanced contrast',
                    'application': 'Better water-land contrast',
                    'enabled': rgb_enabled
                },
                # False Color (2)
                'false_color_infrared': {
                    'red': 842, 'green': 665, 'blue': 560,
                    'description': 'False color infrared',
                    'application': 'Vegetation (red), clear water (dark)',
                    'enabled': rgb_enabled
                },
                'false_color_nir': {
                    'red': 865, 'green': 665, 'blue': 560,
                    'description': 'False color NIR',
                    'application': 'Enhanced vegetation/water contrast',
                    'enabled': rgb_enabled
                },
                # Water-Specific (4)
                'turbidity_enhanced': {
                    'red': 705, 'green': 665, 'blue': 560,
                    'description': 'Turbidity-enhanced RGB',
                    'application': 'Enhanced turbidity visualization',
                    'enabled': rgb_enabled
                },
                'chlorophyll_enhanced': {
                    'red': 705, 'green': 665, 'blue': 490,
                    'description': 'Chlorophyll-enhanced RGB',
                    'application': 'Enhanced chlorophyll visualization',
                    'enabled': rgb_enabled
                },
                'coastal_aerosol': {
                    'red': 665, 'green': 490, 'blue': 443,
                    'description': 'Coastal aerosol RGB',
                    'application': 'Coastal water analysis',
                    'enabled': rgb_enabled
                },
                'water_quality': {
                    'red': 740, 'green': 665, 'blue': 560,
                    'description': 'Water quality RGB',
                    'application': 'General water quality assessment',
                    'enabled': rgb_enabled
                },
                # Research (7) - duplicates removed
                'sediment_transport': {
                    'red': 783, 'green': 705, 'blue': 665,
                    'description': 'Sediment transport RGB',
                    'application': 'Sediment plume visualization',
                    'enabled': rgb_enabled
                },
                'atmospheric_penetration': {
                    'red': 865, 'green': 783, 'blue': 740,
                    'description': 'Atmospheric penetration RGB',
                    'application': 'Deep water analysis',
                    'enabled': rgb_enabled
                },
                'ocean_color_standard': {
                    'red': 490, 'green': 560, 'blue': 443,
                    'description': 'NASA Ocean Color standard composite',
                    'application': 'Ocean color research',
                    'enabled': rgb_enabled
                },
                # algal_bloom_enhanced REMOVED: identical bands (705,665,560) as turbidity_enhanced
                'coastal_turbidity': {
                    'red': 865, 'green': 740, 'blue': 490,
                    'description': 'NIR-enhanced for coastal sediment monitoring',
                    'application': 'Coastal sediment transport and turbidity',
                    'enabled': rgb_enabled
                },
                # deep_water_clarity REMOVED: identical bands (560,490,443) as ocean_color_standard
                # riverine_waters REMOVED: identical bands (740,705,665) as water_quality
                'cdom_enhanced': {
                    'red': 490, 'green': 443, 'blue': 560,
                    'description': 'Blue-shifted for CDOM visualization',
                    'application': 'Colored dissolved organic matter detection',
                    'enabled': rgb_enabled
                },
                'water_change_detection': {
                    'red': 865, 'green': 1610, 'blue': 560,
                    'description': 'SWIR-enhanced for change detection',
                    'application': 'Water body change detection',
                    'enabled': rgb_enabled
                },
                'advanced_atmospheric': {
                    'red': 1375, 'green': 945, 'blue': 705,
                    'description': 'Atmospheric correction enhanced',
                    'application': 'Atmospheric interference reduction',
                    'enabled': rgb_enabled
                }
            }

            active_combinations = {name: config for name, config in rgb_combinations.items()
                                if config['enabled']}

            self.logger.debug(f"Processing {len(active_combinations)} RGB combinations")

            for rgb_name, config in active_combinations.items():
                try:
                    required_bands = [config['red'], config['green'], config['blue']]
                    missing_bands = [wl for wl in required_bands if wl not in available_wavelengths]

                    if missing_bands:
                        self.logger.debug(f"Skipping {rgb_name}: missing wavelengths {missing_bands}")
                        continue

                    self.logger.debug(f"Creating {rgb_name}: R={config['red']}nm, G={config['green']}nm, B={config['blue']}nm")

                    red_data = bands_data[config['red']]
                    green_data = bands_data[config['green']]
                    blue_data = bands_data[config['blue']]

                    rgb_array = self._create_rgb_composite(red_data, green_data, blue_data)

                    # Use new naming: {scene_name}_RGB_{bands}.tif (e.g., S2A_20190105_T29TNF_RGB_432.tif)
                    rgb_filename = OutputStructure.get_rgb_filename(
                        product_name, config['red'], config['green'], config['blue']
                    )
                    output_path = os.path.join(output_folder, rgb_filename)
                    success = self._save_rgb_geotiff(rgb_array, output_path, reference_metadata, config['description'])

                    if success:
                        valid_pixels = np.sum(np.any(rgb_array > 0, axis=2))
                        total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
                        coverage_percent = (valid_pixels / total_pixels) * 100
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

                        stats = {
                            'description': config['description'],
                            'application': config['application'],
                            'bands_used': f"R:{config['red']}nm, G:{config['green']}nm, B:{config['blue']}nm",
                            'file_size_mb': file_size_mb,
                            'coverage_percent': coverage_percent,
                            'output_path': output_path
                        }

                        results[f'rgb_{rgb_name}'] = ProcessingResult(True, output_path, stats, None)
                        self.logger.debug(f"Created {rgb_name}: {coverage_percent:.1f}% coverage, {file_size_mb:.1f}MB")
                    else:
                        results[f'rgb_{rgb_name}'] = ProcessingResult(False, output_path, None, "Failed to save RGB composite")

                except Exception as e:
                    self.logger.error(f"Error generating {rgb_name}: {e}")
                    results[f'rgb_{rgb_name}'] = ProcessingResult(False, "", None, str(e))

            return results

        except Exception as e:
            self.logger.error(f"Error in RGB composite generation: {e}")
            return {'rgb_error': ProcessingResult(False, "", None, str(e))}

    def _generate_spectral_indices(self, bands_data: Dict[int, np.ndarray], reference_metadata: Dict,
                                output_folder: str, product_name: str) -> Dict[str, ProcessingResult]:
        """Generate spectral indices based on configuration and available bands"""
        results = {}

        try:
            available_wavelengths = set(bands_data.keys())
            self.logger.debug(f"Available wavelengths for indices: {sorted(available_wavelengths)}")

            # Note: Water masking is controlled by TSSConfig and applied at save time
            water_mask = None

            # 14 spectral indices (SDD removed - WaterClarity SecchiDepth is more rigorous)
            # TSI renamed to TSI_Turbidity to avoid collision with Trophic State Index
            # pSDB replaces former RDI (Stumpf 2003 + Caballero & Stumpf 2020)
            indices_enabled = self.output_categories.enable_indices

            spectral_indices = {
                # Water Quality (7)
                'NDWI': {
                    'formula': '(B3 - B8) / (B3 + B8)',
                    'required_bands': [560, 842],
                    'description': 'Normalized Difference Water Index (McFeeters 1996)',
                    'application': 'Water body delineation',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'MNDWI': {
                    'formula': '(B3 - B11) / (B3 + B11)',
                    'required_bands': [560, 1610],
                    'fallback_bands': [560, 865],
                    'description': 'Modified Normalized Difference Water Index',
                    'application': 'Enhanced water detection',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'NDTI': {
                    'formula': '(B4 - B3) / (B4 + B3)',
                    'required_bands': [665, 560],
                    'description': 'Normalized Difference Turbidity Index',
                    'application': 'Turbidity assessment',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'NDMI': {
                    'formula': '(B8A - B11) / (B8A + B11)',
                    'required_bands': [865, 1610],
                    'fallback_bands': [842, 1610],
                    'description': 'Normalized Difference Moisture Index',
                    'application': 'Water vs non-water separation, moisture content',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'AWEI': {
                    'formula': '4 * (B3 - B11) - (0.25 * B8A + 2.75 * B12)',
                    'required_bands': [560, 1610, 865, 2190],
                    'fallback_bands': [560, 1610, 842, 2190],
                    'description': 'Automated Water Extraction Index',
                    'application': 'Enhanced water body extraction',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'WI': {
                    'formula': 'B2 / (B3 + B8A)',
                    'required_bands': [490, 560, 865],
                    'fallback_bands': [490, 560, 842],
                    'description': 'Water Index',
                    'application': 'Turbid water detection and delineation',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                'WRI': {
                    'formula': '(B3 + B4) / (B8A + B11)',
                    'required_bands': [560, 665, 865, 1610],
                    'fallback_bands': [560, 665, 842, 1610],
                    'description': 'Water Ratio Index',
                    'application': 'Water/land separation and water quality assessment',
                    'enabled': indices_enabled,
                    'category': 'water_quality'
                },
                # Chlorophyll & Algae (3)
                'NDCI': {
                    'formula': '(B5 - B4) / (B5 + B4)',
                    'required_bands': [705, 665],
                    'description': 'Normalized Difference Chlorophyll Index',
                    'application': 'Chlorophyll concentration',
                    'enabled': indices_enabled,
                    'category': 'chlorophyll'
                },
                'CHL_RED_EDGE': {
                    'formula': '(B5 / B4) - 1',
                    'required_bands': [705, 665],
                    'description': 'Chlorophyll Red Edge',
                    'application': 'Chlorophyll using red edge',
                    'enabled': indices_enabled,
                    'category': 'chlorophyll'
                },
                'GNDVI': {
                    'formula': '(B8 - B3) / (B8 + B3)',
                    'required_bands': [842, 560],
                    'description': 'Green Normalized Difference Vegetation Index',
                    'application': 'Aquatic vegetation',
                    'enabled': indices_enabled,
                    'category': 'chlorophyll'
                },
                # Turbidity & Sediment (2) - TSI renamed to TSI_Turbidity
                'TSI_Turbidity': {
                    'formula': '(B4 + B3) / 2',
                    'required_bands': [665, 560],
                    'description': 'Turbidity Spectral Index',
                    'application': 'Turbidity estimation',
                    'enabled': indices_enabled,
                    'category': 'turbidity'
                },
                'NGRDI': {
                    'formula': '(B3 - B4) / (B3 + B4)',
                    'required_bands': [560, 665],
                    'description': 'Normalized Green Red Difference Index',
                    'application': 'Water-vegetation separation',
                    'enabled': indices_enabled,
                    'category': 'turbidity'
                },
                # Advanced (2) - SDD removed (WaterClarity SecchiDepth is IOP-derived, more rigorous)
                'CDOM': {
                    'formula': 'B1 / B3',
                    'required_bands': [443, 560],
                    'description': 'Colored Dissolved Organic Matter proxy',
                    'application': 'CDOM concentration',
                    'enabled': indices_enabled,
                    'category': 'advanced'
                },
                'pSDB': {
                    'formula': 'ln(1000*B2) / ln(1000*B3|B4)',
                    'required_bands': [490, 560, 665],
                    'description': 'Pseudo Satellite-Derived Bathymetry (Stumpf 2003, Caballero & Stumpf 2020)',
                    'application': 'Relative bathymetry with turbidity-adaptive band switching',
                    'enabled': indices_enabled,
                    'category': 'advanced'
                }
            }

            active_indices = {name: config for name, config in spectral_indices.items() if config['enabled']}

            self.logger.debug(f"Processing {len(active_indices)} spectral indices")

            for index_name, config in active_indices.items():
                try:
                    required_bands = config['required_bands']
                    fallback_bands = config.get('fallback_bands', None)

                    if all(band in available_wavelengths for band in required_bands):
                        bands_to_use = required_bands
                    elif fallback_bands and all(band in available_wavelengths for band in fallback_bands):
                        bands_to_use = fallback_bands
                        self.logger.debug(f"Using fallback bands for {index_name}")
                    else:
                        missing = [b for b in required_bands if b not in available_wavelengths]
                        self.logger.debug(f"Skipping {index_name}: missing wavelengths {missing}")
                        continue

                    self.logger.debug(f"Calculating {index_name} using bands: {bands_to_use}")

                    index_data = self._calculate_spectral_index(index_name, config, bands_data, bands_to_use)

                    if index_data is not None:
                        # Apply water mask (set land pixels to NaN) if available
                        if water_mask is not None:
                            index_data = np.where(water_mask, index_data, np.nan)

                        # Use new naming: {scene_name}_{INDEX}.tif (e.g., S2A_20190105_T29TNF_NDWI.tif)
                        index_filename = f"{product_name}_{index_name.upper()}.tif"
                        output_path = os.path.join(output_folder, index_filename)
                        success = self._save_single_band_geotiff(index_data, output_path, reference_metadata, config['description'])

                        if success:
                            valid_data = index_data[~np.isnan(index_data)]
                            if len(valid_data) > 0:
                                stats = {
                                    'description': config['description'],
                                    'application': config['application'],
                                    'formula': config['formula'],
                                    'bands_used': bands_to_use,
                                    'category': config['category'],
                                    'min_value': float(np.min(valid_data)),
                                    'max_value': float(np.max(valid_data)),
                                    'mean_value': float(np.mean(valid_data)),
                                    'std_value': float(np.std(valid_data)),
                                    'valid_pixels': len(valid_data),
                                    'coverage_percent': (len(valid_data) / index_data.size) * 100,
                                    'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
                                }
                            else:
                                stats = {'description': config['description'], 'error': 'No valid data'}

                            results[f'index_{index_name.lower()}'] = ProcessingResult(True, output_path, stats, None)
                            self.logger.debug(f"Created {index_name}: {stats.get('coverage_percent', 0):.1f}% coverage")
                        else:
                            results[f'index_{index_name.lower()}'] = ProcessingResult(False, output_path, None, "Failed to save index")
                    else:
                        results[f'index_{index_name.lower()}'] = ProcessingResult(False, "", None, "Failed to calculate index")

                except Exception as e:
                    self.logger.error(f"Error calculating {index_name}: {e}")
                    results[f'index_{index_name.lower()}'] = ProcessingResult(False, "", None, str(e))

            return results

        except Exception as e:
            self.logger.error(f"Error in spectral index generation: {e}")
            return {'index_error': ProcessingResult(False, "", None, str(e))}


    def _calculate_spectral_index(self, index_name: str, config: Dict, bands_data: Dict[int, np.ndarray],
                                bands_to_use: List[int]) -> Optional[np.ndarray]:
        """Calculate specific spectral index using provided bands"""
        try:
            # Water Quality Indices
            if index_name == 'NDWI':
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b8 = bands_data[bands_to_use[1]]  # 842nm (B8, broad NIR)
                return (b3 - b8) / (b3 + b8 + 1e-8)

            elif index_name == 'MNDWI':
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b_swir = bands_data[bands_to_use[1]]  # 865nm or 1610nm
                return (b3 - b_swir) / (b3 + b_swir + 1e-8)

            elif index_name == 'NDTI':
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b4 - b3) / (b4 + b3 + 1e-8)

            elif index_name == 'NDMI':
                nir = bands_data[bands_to_use[0]]   # 865nm or 842nm
                swir1 = bands_data[bands_to_use[1]] # 1610nm
                return (nir - swir1) / (nir + swir1 + 1e-8)

            elif index_name == 'AWEI':
                green = bands_data[bands_to_use[0]]  # 560nm
                swir1 = bands_data[bands_to_use[1]]  # 1610nm
                nir = bands_data[bands_to_use[2]]    # 865nm or 842nm
                swir2 = bands_data[bands_to_use[3]]  # 2190nm
                return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)

            elif index_name == 'WI':
                blue = bands_data[bands_to_use[0]]   # 490nm
                green = bands_data[bands_to_use[1]]  # 560nm
                nir = bands_data[bands_to_use[2]]    # 865nm or 842nm
                return blue / (green + nir + 1e-8)

            elif index_name == 'WRI':
                green = bands_data[bands_to_use[0]]  # 560nm
                red = bands_data[bands_to_use[1]]    # 665nm
                nir = bands_data[bands_to_use[2]]    # 865nm or 842nm
                swir1 = bands_data[bands_to_use[3]]  # 1610nm
                return (green + red) / (nir + swir1 + 1e-8)

            # Chlorophyll & Algae Indices
            elif index_name == 'NDCI':
                b5 = bands_data[bands_to_use[0]]  # 705nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b5 - b4) / (b5 + b4 + 1e-8)

            elif index_name == 'CHL_RED_EDGE':
                b5 = bands_data[bands_to_use[0]]  # 705nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b5 / (b4 + 1e-8)) - 1

            elif index_name == 'GNDVI':
                b8 = bands_data[bands_to_use[0]]  # 842nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b8 - b3) / (b8 + b3 + 1e-8)

            # PC, FAI removed: require spectral bands not available in Sentinel-2

            # Turbidity & Sediment Indices
            elif index_name == 'TSI_Turbidity':
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b4 + b3) / 2

            elif index_name == 'NGRDI':
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b3 - b4) / (b3 + b4 + 1e-8)

            # SDD removed: WaterClarity SecchiDepth (IOP-derived) is more rigorous

            elif index_name == 'CDOM':
                b1 = bands_data[bands_to_use[0]]  # 443nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return b1 / (b3 + 1e-8)

            elif index_name == 'pSDB':
                # Pseudo Satellite-Derived Bathymetry
                # Stumpf (2003) log-ratio with Caballero & Stumpf (2020) band switching
                # pSDB = ln(n * Rrs_λ1) / ln(n * Rrs_λ2), n=1000
                # Clear/moderate water: B2/B3 (green ratio, deeper penetration)
                # Turbid water: B2/B4 (red ratio, less turbidity-sensitive)
                n = 1000.0
                MIN_LOG_INPUT = 1.6487  # e^0.5, ensures ln() > 0.5

                blue = bands_data[bands_to_use[0]]    # 490nm (B2)
                green = bands_data[bands_to_use[1]]   # 560nm (B3)
                red = bands_data[bands_to_use[2]]     # 665nm (B4)

                psdb = np.full_like(blue, np.nan, dtype=np.float32)

                # Green ratio (B2/B3) — default for clear/moderate water
                valid_g = ((blue > 0) & (green > 0) &
                           (n * blue > MIN_LOG_INPUT) & (n * green > MIN_LOG_INPUT))
                psdb_green = np.full_like(blue, np.nan, dtype=np.float32)
                psdb_green[valid_g] = np.log(n * blue[valid_g]) / np.log(n * green[valid_g])

                # Red ratio (B2/B4) — for turbid water
                valid_r = ((blue > 0) & (red > 0) &
                           (n * blue > MIN_LOG_INPUT) & (n * red > MIN_LOG_INPUT))
                psdb_red = np.full_like(blue, np.nan, dtype=np.float32)
                psdb_red[valid_r] = np.log(n * blue[valid_r]) / np.log(n * red[valid_r])

                # Band switching: use water type if available, else green ratio
                if self._water_type is not None and self._water_type.shape == blue.shape:
                    wt = self._water_type
                    # Type I (clear) + Type II (moderate) → green ratio
                    clear_mod = (wt == 1) | (wt == 2)
                    psdb[clear_mod] = psdb_green[clear_mod]
                    # Type III (turbid) → red ratio
                    psdb[wt == 3] = psdb_red[wt == 3]
                    # Type 0 (invalid/land) and Type IV (extreme) → remain NaN

                    n_green = int(np.sum(clear_mod))
                    n_red = int(np.sum(wt == 3))
                    n_masked = int(np.sum((wt == 0) | (wt == 4)))
                    self.logger.debug(
                        f"pSDB band switching: {n_green} green, "
                        f"{n_red} red, {n_masked} masked (Type 0/IV)"
                    )
                elif self._water_type is not None:
                    self.logger.warning(
                        f"pSDB: water_type shape {self._water_type.shape} != "
                        f"band shape {blue.shape}, falling back to green ratio"
                    )
                    psdb = psdb_green
                else:
                    psdb = psdb_green

                # Free intermediate arrays
                del psdb_green, psdb_red, valid_g, valid_r

                # Guard against infinity, then clamp to valid range [0, 5]
                psdb[np.isinf(psdb)] = np.nan
                psdb = np.clip(psdb, 0.0, 5.0)
                return psdb

            else:
                self.logger.warning(f"Unknown index calculation for {index_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error calculating {index_name}: {e}")
            return None

    # Note: _create_water_mask() removed - water masking is now controlled by
    # TSSConfig (shapefile mask or NIR threshold) and applied centrally

    def _create_rgb_composite(self, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Create RGB composite with percentile-based normalization."""
        try:
            if red.shape != green.shape or red.shape != blue.shape:
                raise ValueError(f"Band shape mismatch: R={red.shape}, G={green.shape}, B={blue.shape}")

            height, width = red.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.float32)

            bands = [red, green, blue]
            band_names = ['Red', 'Green', 'Blue']

            for i, (band, name) in enumerate(zip(bands, band_names)):
                valid_mask = (~np.isnan(band)) & (~np.isinf(band)) & (band >= 0)

                if not np.any(valid_mask):
                    self.logger.warning(f"{name} band: No valid data found")
                    rgb_array[:, :, i] = 0
                    continue

                valid_data = band[valid_mask]
                p2, p98 = np.percentile(valid_data, [2, 98])

                if p98 > p2:
                    normalized_band = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    normalized_band = np.clip(band, 0, 1)

                rgb_array[:, :, i] = normalized_band

                mean_val = np.mean(valid_data)
                self.logger.debug(f"{name}: mean={mean_val:.4f}, range=[{p2:.4f}, {p98:.4f}]")

            return rgb_array

        except Exception as e:
            self.logger.error(f"Error in RGB composite creation: {e}")
            return np.zeros((red.shape[0], red.shape[1], 3), dtype=np.float32)

    def _save_rgb_geotiff(self, rgb_array: np.ndarray, output_path: str,
                        metadata: Dict, description: str) -> bool:
        """Save RGB array as GeoTIFF with proper metadata"""
        try:
            rgb_uint8 = (rgb_array * 255).astype(np.uint8)
            height, width, bands = rgb_uint8.shape

            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                self.logger.error("GTiff driver not available")
                return False

            dataset = driver.Create(
                output_path,
                width,
                height,
                bands,
                gdal.GDT_Byte,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )

            if dataset is None:
                self.logger.error(f"Could not create dataset: {output_path}")
                return False

            if 'geotransform' in metadata:
                dataset.SetGeoTransform(metadata['geotransform'])
            if 'projection' in metadata:
                dataset.SetProjection(metadata['projection'])

            for i in range(bands):
                band = dataset.GetRasterBand(i + 1)
                band.WriteArray(rgb_uint8[:, :, i])

                if i == 0:
                    band.SetDescription('Red')
                    band.SetColorInterpretation(gdal.GCI_RedBand)
                elif i == 1:
                    band.SetDescription('Green')
                    band.SetColorInterpretation(gdal.GCI_GreenBand)
                elif i == 2:
                    band.SetDescription('Blue')
                    band.SetColorInterpretation(gdal.GCI_BlueBand)

                band.SetStatistics(0, 255, float(np.mean(rgb_uint8[:, :, i])), float(np.std(rgb_uint8[:, :, i])))

            dataset.SetDescription(description)
            dataset.SetMetadataItem('DESCRIPTION', description)
            dataset.SetMetadataItem('CREATION_DATE', datetime.now().isoformat())
            dataset.SetMetadataItem('SOURCE', 'Unified S2 TSS Pipeline - Marine Visualization')

            dataset.SetMetadataItem('PROCESSING_METHOD', 'Marine RGB Composite')

            dataset.FlushCache()
            dataset = None

            self.logger.debug(f"Saved RGB composite: {os.path.basename(output_path)}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving RGB GeoTIFF {output_path}: {e}")
            return False

    def _save_single_band_geotiff(self, data: np.ndarray, output_path: str,
                                metadata: Dict, description: str) -> bool:
        """Save single band data as GeoTIFF with proper metadata"""
        try:
            height, width = data.shape

            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                self.logger.error("GTiff driver not available")
                return False

            dataset = driver.Create(
                output_path,
                width,
                height,
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )

            if dataset is None:
                self.logger.error(f"Could not create dataset: {output_path}")
                return False

            if 'geotransform' in metadata:
                dataset.SetGeoTransform(metadata['geotransform'])
            if 'projection' in metadata:
                dataset.SetProjection(metadata['projection'])

            band = dataset.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))

            nodata_value = metadata.get('nodata', float('nan'))
            band.SetNoDataValue(float(nodata_value))

            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                band.SetStatistics(
                    float(np.min(valid_data)),
                    float(np.max(valid_data)),
                    float(np.mean(valid_data)),
                    float(np.std(valid_data))
                )

            band.SetDescription(description)

            dataset.SetDescription(description)
            dataset.SetMetadataItem('DESCRIPTION', description)
            dataset.SetMetadataItem('CREATION_DATE', datetime.now().isoformat())
            dataset.SetMetadataItem('SOURCE', 'Unified S2 TSS Pipeline - Marine Visualization')

            dataset.SetMetadataItem('PROCESSING_METHOD', 'Spectral Index Calculation')
            dataset.SetMetadataItem('UNITS', 'Dimensionless')

            dataset.FlushCache()
            dataset = None

            self.logger.debug(f"Saved spectral index: {os.path.basename(output_path)}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving single band GeoTIFF {output_path}: {e}")
            return False

    def _create_visualization_summary(self, viz_results: Dict[str, ProcessingResult],
                                    output_folder: str, product_name: str):
        """Create comprehensive visualization summary report"""
        try:
            summary_file = os.path.join(output_folder, f"{product_name}_Marine_Visualization_Summary.txt")

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("MARINE VISUALIZATION PROCESSING SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing with Marine Visualization\n")
                f.write(f"Output directory: {output_folder}\n\n")

                total_products = len(viz_results)
                successful_products = len([r for r in viz_results.values() if r.success])
                failed_products = total_products - successful_products
                success_rate = (successful_products / total_products) * 100 if total_products > 0 else 0

                f.write("PROCESSING STATISTICS:\n")
                f.write(f"  Total products attempted: {total_products}\n")
                f.write(f"  Successfully generated: {successful_products}\n")
                f.write(f"  Failed: {failed_products}\n")
                f.write(f"  Success rate: {success_rate:.1f}%\n\n")

                rgb_products = {k: v for k, v in viz_results.items() if k.startswith('rgb_')}
                index_products = {k: v for k, v in viz_results.items() if k.startswith('index_')}

                f.write("PRODUCT CATEGORIES:\n")
                f.write(f"  RGB Composites: {len(rgb_products)} products\n")
                f.write(f"  Spectral Indices: {len(index_products)} products\n\n")

                if rgb_products:
                    f.write("RGB COMPOSITES:\n")
                    f.write("-" * 40 + "\n")

                    for product_key, result in rgb_products.items():
                        product_name_clean = product_key.replace('rgb_', '').replace('_', ' ').title()

                        if result.success and result.stats:
                            stats = result.stats
                            f.write(f"+ {product_name_clean}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Application: {stats.get('application', 'N/A')}\n")
                            f.write(f"  Bands used: {stats.get('bands_used', 'N/A')}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  File size: {stats.get('file_size_mb', 0):.1f} MB\n")
                            f.write(f"  Output: {os.path.basename(result.output_path)}\n\n")
                        else:
                            f.write(f"- {product_name_clean} (failed)\n")
                            if result.error_message:
                                f.write(f"  Error: {result.error_message}\n\n")
                            else:
                                f.write("  Error: Unknown failure\n\n")

                if index_products:
                    f.write("SPECTRAL INDICES:\n")
                    f.write("-" * 40 + "\n")

                    for product_key, result in index_products.items():
                        index_name = product_key.replace('index_', '').upper()

                        if result.success and result.stats:
                            stats = result.stats
                            f.write(f"+ {index_name}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Application: {stats.get('application', 'N/A')}\n")
                            f.write(f"  Formula: {stats.get('formula', 'N/A')}\n")
                            f.write(f"  Bands used: {stats.get('bands_used', 'N/A')}\n")
                            f.write(f"  Value range: [{stats.get('min_value', 0):.4f}, {stats.get('max_value', 0):.4f}]\n")
                            f.write(f"  Mean +/- Std: {stats.get('mean_value', 0):.4f} +/- {stats.get('std_value', 0):.4f}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  File size: {stats.get('file_size_mb', 0):.1f} MB\n")
                            f.write(f"  Output: {os.path.basename(result.output_path)}\n\n")
                        else:
                            f.write(f"- {index_name} (failed)\n")
                            if result.error_message:
                                f.write(f"  Error: {result.error_message}\n\n")
                            else:
                                f.write("  Error: Unknown failure\n\n")

                f.write("OUTPUT CATEGORIES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  RGB Composites: {self.output_categories.enable_rgb}\n")
                f.write(f"  Spectral Indices: {self.output_categories.enable_indices}\n\n")

                f.write("OUTPUT FILES:\n")
                f.write("-" * 40 + "\n")
                successful_files = [result.output_path for result in viz_results.values()
                                if result.success and result.output_path]

                for file_path in sorted(successful_files):
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    f.write(f"  {os.path.basename(file_path)} ({file_size_mb:.1f} MB)\n")

                total_size_mb = sum(os.path.getsize(fp) / (1024 * 1024) for fp in successful_files)
                f.write(f"\nTotal output size: {total_size_mb:.1f} MB\n")

            self.logger.debug(f"Visualization summary created: {os.path.basename(summary_file)}")

        except Exception as e:
            self.logger.warning(f"Could not create visualization summary: {e}")

    def _cleanup_intermediate_products(self, results_folder: str, product_name: str) -> bool:
        """Clean up intermediate resampled products after processing."""
        try:
            geometric_folder = OutputStructure.get_intermediate_folder(
                results_folder, OutputStructure.GEOMETRIC_FOLDER
            )

            if not os.path.exists(geometric_folder):
                logger.debug("No geometric folder to delete")
                return True

            logger.debug("Attempting cleanup of geometric products")

            try:
                original_items = os.listdir(geometric_folder)
                resampled_items = [item for item in original_items if item.startswith('Resampled_')]

                if not resampled_items:
                    logger.debug("No Resampled_ items found to delete")
                    return True

                logger.debug(f"Found {len(resampled_items)} Resampled items to delete")

                total_size = 0
                for item in resampled_items:
                    item_path = os.path.join(geometric_folder, item)
                    try:
                        if os.path.isfile(item_path):
                            total_size += os.path.getsize(item_path)
                        elif os.path.isdir(item_path):
                            for root, dirs, files in os.walk(item_path):
                                for file in files:
                                    try:
                                        total_size += os.path.getsize(os.path.join(root, file))
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                total_size_mb = total_size / (1024 * 1024)
                logger.debug(f"Total size to delete: {total_size_mb:.1f} MB")

            except Exception as e:
                logger.warning(f"Could not analyze items: {e}")
                resampled_items = []
                total_size_mb = 0

            # Method 1: Try shutil.rmtree
            try:
                logger.debug("Using shutil.rmtree for cleanup...")

                for item in resampled_items:
                    item_path = os.path.join(geometric_folder, item)
                    if os.path.exists(item_path):
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        elif os.path.isfile(item_path):
                            try:
                                os.remove(item_path)
                            except Exception:
                                pass

                remaining_items = os.listdir(geometric_folder) if os.path.exists(geometric_folder) else []
                remaining_resampled = [item for item in remaining_items if item.startswith('Resampled_')]

                if not remaining_resampled:
                    logger.debug(f"Cleanup successful: Deleted all geometric products ({total_size_mb:.1f} MB freed)")
                    return True
                else:
                    logger.error(f"Cleanup failed: {len(remaining_resampled)} items remain: {remaining_resampled}")
                    return False

            except Exception as e:
                logger.error(f"Cleanup with shutil failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            import traceback
            logger.error(f"Cleanup traceback: {traceback.format_exc()}")
            return False

