"""
Visualization Processor.

Generate RGB composites and spectral indices from Sentinel-2 imagery.

This module provides:
- RGBCompositeDefinitions: Defines 20 RGB combinations and 15 spectral indices
- VisualizationProcessor: Complete visualization processor

RGB Categories:
- Natural Color (2): natural_color, natural_with_contrast
- False Color (2): false_color_infrared, false_color_nir
- Water-Specific (6): turbidity, chlorophyll, coastal, sediment, water_quality, atmospheric
- Research (10): NASA Ocean Color, HAB detection, coastal turbidity, deep water, riverine, CDOM, etc.

Spectral Index Categories:
- Water Quality (7): NDWI, MNDWI, NDTI, NDMI, AWEI, WI, WRI
- Chlorophyll & Algae (3): NDCI, CHL_RED_EDGE, GNDVI
- Turbidity & Sediment (2): TSI, NGRDI
- Advanced (3): SDD, CDOM, RDI (Relative Depth Index)

Note: PC, FAI, FUI removed - require spectral bands not available in Sentinel-2.
"""

import os
import logging
import shutil
import subprocess
import platform
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

from ..config import MarineVisualizationConfig
from ..utils.raster_io import RasterIO
from ..utils.output_structure import OutputStructure
from .snap_calculator import ProcessingResult

logger = logging.getLogger('sentinel2_tss_pipeline')


class RGBCompositeDefinitions:
    """Define RGB combinations and spectral indices for visualization."""

    def __init__(self):
        # Marine-optimized RGB combinations using Sentinel-2 bands
        self.rgb_combinations = {
            # =================================================================
            # NATURAL COLOR COMBINATIONS (2 combinations)
            # =================================================================
            'natural_color': {
                'red': 665, 'green': 560, 'blue': 490,  # B4, B3, B2
                'description': 'Natural color (True color)',
                'application': 'General visualization, publications',
                'priority': 'essential'
            },
            'natural_with_contrast': {
                'red': 665, 'green': 560, 'blue': 443,  # B4, B3, B1
                'description': 'Natural color with enhanced contrast',
                'application': 'Better water-land contrast',
                'priority': 'important'
            },

            # =================================================================
            # FALSE COLOR COMBINATIONS (2 combinations)
            # =================================================================
            'false_color_infrared': {
                'red': 842, 'green': 665, 'blue': 560,  # B8, B4, B3
                'description': 'False color infrared',
                'application': 'Vegetation (red), clear water (dark)',
                'priority': 'important'
            },
            'false_color_nir': {
                'red': 865, 'green': 665, 'blue': 560,  # B8A, B4, B3
                'description': 'False color NIR',
                'application': 'Enhanced vegetation/water contrast',
                'priority': 'important'
            },

            # =================================================================
            # WATER-SPECIFIC COMBINATIONS (3 combinations)
            # =================================================================
            'turbidity_enhanced': {
                'red': 865, 'green': 705, 'blue': 560,  # B8A, B5, B3
                'description': 'Turbidity and suspended sediment',
                'application': 'Sediment plumes, river discharge',
                'priority': 'marine'
            },
            'chlorophyll_enhanced': {
                'red': 705, 'green': 665, 'blue': 560,  # B5, B4, B3
                'description': 'Chlorophyll and algae detection',
                'application': 'Algal blooms, phytoplankton',
                'priority': 'marine'
            },
            'coastal_aerosol': {
                'red': 865, 'green': 665, 'blue': 443,  # B8A, B4, B1
                'description': 'Coastal aerosol enhanced',
                'application': 'Atmospheric correction, haze penetration',
                'priority': 'marine'
            },

            # =================================================================
            # SPECIALIZED MARINE COMBINATIONS (3 combinations)
            # =================================================================
            'sediment_transport': {
                'red': 2190, 'green': 865, 'blue': 665,  # B12, B8A, B4
                'description': 'Sediment transport visualization',
                'application': 'River plumes, coastal erosion',
                'priority': 'marine'
            },
            'water_quality': {
                'red': 842, 'green': 705, 'blue': 490,  # B8, B5, B2
                'description': 'Water quality assessment',
                'application': 'Turbidity, organic matter',
                'priority': 'marine'
            },
            'atmospheric_penetration': {
                'red': 2190, 'green': 1610, 'blue': 865,  # B12, B11, B8A
                'description': 'Atmospheric penetration',
                'application': 'Hazy conditions, atmospheric interference',
                'priority': 'optional'
            },

            # =================================================================
            # RESEARCH COMBINATIONS - EXISTING (2 combinations)
            # =================================================================
            'research_marine': {
                'red': 740, 'green': 705, 'blue': 665,  # B6, B5, B4
                'description': 'Marine research combination',
                'application': 'Red edge analysis, chlorophyll research',
                'priority': 'research'
            },
            'bathymetric': {
                'red': 560, 'green': 490, 'blue': 443,  # B3, B2, B1
                'description': 'Bathymetric analysis',
                'application': 'Shallow water depth estimation',
                'priority': 'research'
            },

            # =================================================================
            # RESEARCH COMBINATIONS - NEW ADDITIONS (8 combinations)
            # Based on latest 2023-2024 scientific literature
            # =================================================================

            # 1. Ocean Color Research Standard (NASA/ESA protocol)
            'ocean_color_standard': {
                'red': 490, 'green': 560, 'blue': 443,  # B2, B3, B1
                'description': 'NASA Ocean Color standard composite',
                'application': 'Ocean color research, international standard protocol',
                'priority': 'research',
                'reference': "O'Reilly et al. (1998) - Ocean color chlorophyll algorithms for SeaWiFS",
                'doi': '10.1029/98JC02160'
            },

            # 2. Enhanced Harmful Algal Bloom Detection
            'algal_bloom_enhanced': {
                'red': 705, 'green': 665, 'blue': 560,  # B5, B4, B3
                'description': 'Red edge enhanced for algal bloom detection',
                'application': 'Harmful algal bloom monitoring and early detection',
                'priority': 'research',
                'reference': 'Caballero et al. (2020) - New capabilities of Sentinel-2A/B for HAB monitoring',
                'doi': '10.1038/s41598-020-65600-1'
            },

            # 3. Coastal Turbidity Monitoring
            'coastal_turbidity': {
                'red': 865, 'green': 740, 'blue': 490,  # B8A, B6, B2
                'description': 'NIR-enhanced for coastal sediment monitoring',
                'application': 'Coastal sediment transport and turbidity assessment',
                'priority': 'research',
                'reference': 'Nechad et al. (2010) - Generic multisensor algorithm for TSM mapping',
                'doi': '10.1016/j.rse.2009.11.022'
            },

            # 4. Deep Water Clarity Analysis
            'deep_water_clarity': {
                'red': 560, 'green': 490, 'blue': 443,  # B3, B2, B1
                'description': 'Blue-enhanced for deep water penetration',
                'application': 'Deep water clarity and depth estimation',
                'priority': 'research',
                'reference': 'Lee et al. (2002) - Deriving inherent optical properties from water color',
                'doi': '10.1364/AO.41.005755'
            },

            # 5. River and Estuarine Waters
            'riverine_waters': {
                'red': 740, 'green': 705, 'blue': 665,  # B6, B5, B4
                'description': 'Red edge focus for riverine environments',
                'application': 'River discharge monitoring and estuarine mixing',
                'priority': 'research',
                'reference': 'Pahlevan et al. (2017) - Sentinel-2 MSI data processing for aquatic science',
                'doi': '10.1016/j.rse.2017.08.033'
            },

            # 6. CDOM Enhanced Visualization
            'cdom_enhanced': {
                'red': 490, 'green': 443, 'blue': 560,  # B2, B1, B3 (blue-shifted)
                'description': 'Blue-shifted for CDOM visualization',
                'application': 'Colored dissolved organic matter detection',
                'priority': 'research',
                'reference': 'Mannino et al. (2008) - Algorithm development for DOC and CDOM distributions',
                'doi': '10.1029/2007JC004493'
            },

            # 7. Multi-temporal Water Change Detection
            'water_change_detection': {
                'red': 865, 'green': 1610, 'blue': 560,  # B8A, B11, B3
                'description': 'SWIR-enhanced for change detection',
                'application': 'Water body change detection and temporal monitoring',
                'priority': 'research',
                'reference': 'Pekel et al. (2016) - High-resolution mapping of global surface water changes',
                'doi': '10.1038/nature20584'
            },

            # 8. Advanced Atmospheric Correction
            'advanced_atmospheric': {
                'red': 1375, 'green': 945, 'blue': 705,  # B10, B9, B5
                'description': 'Atmospheric correction enhanced',
                'application': 'Atmospheric interference reduction, water vapor correction',
                'priority': 'research',
                'reference': 'Vanhellemont & Ruddick (2018) - Atmospheric correction of metre-scale optical satellite data',
                'doi': '10.1016/j.rse.2018.02.047'
            }
        }


        # Spectral indices for marine applications
        self.spectral_indices = {
            # =================================================================
            # WATER QUALITY INDICES (7 indices)
            # =================================================================
            'NDWI': {
                'formula': '(B3 - B8A) / (B3 + B8A)',
                'required_bands': [560, 865],
                'description': 'Normalized Difference Water Index',
                'application': 'Water body delineation',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate water',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'essential'
            },
            'MNDWI': {
                'formula': '(B3 - B11) / (B3 + B11)',
                'required_bands': [560, 1610],
                'fallback_bands': [560, 865],
                'description': 'Modified Normalized Difference Water Index',
                'application': 'Enhanced water detection',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate water',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'important'
            },
            'NDTI': {
                'formula': '(B4 - B3) / (B4 + B3)',
                'required_bands': [665, 560],
                'description': 'Normalized Difference Turbidity Index',
                'application': 'Turbidity assessment',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate turbidity',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'essential'
            },
            'NDMI': {
                'formula': '(B8A - B11) / (B8A + B11)',
                'required_bands': [865, 1610],
                'fallback_bands': [842, 1610],
                'description': 'Normalized Difference Moisture Index',
                'application': 'Water vs non-water separation',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate more water/moisture',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'important'
            },
            'AWEI': {
                'formula': '4 * (B3 - B11) - (0.25 * B8A + 2.75 * B12)',
                'required_bands': [560, 1610, 865, 2190],
                'fallback_bands': [560, 1610, 842, 2190],
                'description': 'Automated Water Extraction Index',
                'application': 'Enhanced water body extraction',
                'range': '(-inf, inf)',
                'interpretation': 'Positive values indicate water',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'important'
            },
            'WI': {
                'formula': 'B2 / (B3 + B8A)',
                'required_bands': [490, 560, 865],
                'fallback_bands': [490, 560, 842],
                'description': 'Water Index',
                'application': 'Turbid water detection',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate clearer water',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'marine'
            },
            'WRI': {
                'formula': '(B3 + B4) / (B8A + B11)',
                'required_bands': [560, 665, 865, 1610],
                'fallback_bands': [560, 665, 842, 1610],
                'description': 'Water Ratio Index',
                'application': 'Water/land separation',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate water presence',
                'enabled_by': 'generate_water_quality_indices',
                'category': 'water_quality',
                'priority': 'marine'
            },

            # =================================================================
            # CHLOROPHYLL & ALGAE INDICES (5 indices)
            # =================================================================
            'NDCI': {
                'formula': '(B5 - B4) / (B5 + B4)',
                'required_bands': [705, 665],
                'description': 'Normalized Difference Chlorophyll Index',
                'application': 'Chlorophyll concentration',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate chlorophyll',
                'enabled_by': 'generate_chlorophyll_indices',
                'category': 'chlorophyll',
                'priority': 'marine'
            },
            'CHL_RED_EDGE': {
                'formula': '(B5 / B4) - 1',
                'required_bands': [705, 665],
                'description': 'Chlorophyll Red Edge',
                'application': 'Chlorophyll using red edge',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate chlorophyll',
                'enabled_by': 'generate_chlorophyll_indices',
                'category': 'chlorophyll',
                'priority': 'marine'
            },
            'GNDVI': {
                'formula': '(B8 - B3) / (B8 + B3)',
                'required_bands': [842, 560],
                'description': 'Green Normalized Difference Vegetation Index',
                'application': 'Aquatic vegetation',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate vegetation',
                'enabled_by': 'generate_chlorophyll_indices',
                'category': 'chlorophyll',
                'priority': 'important'
            },
            # PC (Phycocyanin Index) - REMOVED: Requires 620nm band (S2 lacks this band)
            # FAI (Floating Algae Index) - REMOVED: Requires SWIR at 20m (not available in resampled C2RCC)

            # =================================================================
            # TURBIDITY & SEDIMENT INDICES (2 indices)
            # =================================================================
            'TSI': {
                'formula': '(B4 + B3) / 2',
                'required_bands': [665, 560],
                'description': 'Turbidity Spectral Index',
                'application': 'Turbidity estimation',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate turbidity',
                'enabled_by': 'generate_turbidity_indices',
                'category': 'turbidity',
                'priority': 'essential'
            },
            'NGRDI': {
                'formula': '(B3 - B4) / (B3 + B4)',
                'required_bands': [560, 665],
                'description': 'Normalized Green Red Difference Index',
                'application': 'Water-vegetation separation',
                'range': '(-1, 1)',
                'interpretation': 'Higher values indicate vegetation',
                'enabled_by': 'generate_turbidity_indices',
                'category': 'turbidity',
                'priority': 'marine'
            },

            # =================================================================
            # ADVANCED WATER PROPERTIES (3 indices)
            # =================================================================
            # FUI (Forel-Ule Index) - REMOVED: Requires CIE chromaticity conversion (current formula scientifically incorrect)
            'SDD': {
                'formula': 'ln(0.14 / B4) / 1.7',
                'required_bands': [665],
                'description': 'Secchi Disk Depth proxy',
                'application': 'Water transparency',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate clearer water',
                'enabled_by': 'generate_advanced_indices',
                'category': 'advanced',
                'priority': 'advanced'
            },
            'CDOM': {
                'formula': 'B1 / B3',
                'required_bands': [443, 560],
                'description': 'Colored Dissolved Organic Matter proxy',
                'application': 'CDOM concentration',
                'range': '(0, inf)',
                'interpretation': 'Higher values indicate more CDOM',
                'enabled_by': 'generate_advanced_indices',
                'category': 'advanced',
                'priority': 'advanced'
            },
            'RDI': {
                'formula': 'ln(B2) / ln(B3)',
                'required_bands': [490, 560],
                'description': 'Relative Depth Index (Stumpf et al. 2003)',
                'application': 'Time-series bathymetric change detection, sediment deposition monitoring',
                'range': '(0, 1)',
                'interpretation': 'Higher values indicate deeper water (relative, not calibrated)',
                'limitations': 'Valid only in clear, shallow water (<20m). Users should mask high-TSS areas.',
                'reference': 'Stumpf et al. (2003) Limnol. Oceanogr. 48(1):547-556',
                'enabled_by': 'generate_advanced_indices',
                'category': 'advanced',
                'priority': 'advanced'
            }
        }


class VisualizationProcessor:
    """Process RGB composites and spectral indices from Sentinel-2 data."""

    def __init__(self, config: Optional[MarineVisualizationConfig] = None):
        """Initialize marine visualization processor with configuration and logging"""
        # Check GDAL availability
        if not GDAL_AVAILABLE:
            logger.warning("GDAL not available - visualization functionality will be limited")

        # Configuration setup
        self.config = config or MarineVisualizationConfig()

        # Initialize RGB generator
        self.rgb_generator = RGBCompositeDefinitions()

        # Use global logger (updated for centralized logging)
        self.logger = logger

        # Log initialization details
        self.logger.debug("Initialized S2 Marine Visualization Processor")
        self.logger.debug(f"Configuration settings:")
        self.logger.debug(f"  Natural color RGB: {self.config.generate_natural_color}")
        self.logger.debug(f"  False color RGB: {self.config.generate_false_color}")
        self.logger.debug(f"  Water-specific RGB: {self.config.generate_water_specific}")
        self.logger.debug(f"  Research RGB: {self.config.generate_research_combinations}")
        self.logger.debug(f"  Water quality indices: {self.config.generate_water_quality_indices}")
        self.logger.debug(f"  Chlorophyll indices: {self.config.generate_chlorophyll_indices}")
        self.logger.debug(f"  Turbidity indices: {self.config.generate_turbidity_indices}")
        self.logger.debug(f"  Advanced indices: {self.config.generate_advanced_indices}")
        self.logger.debug(f"  Output format: {self.config.rgb_format}")
        self.logger.debug(f"  Contrast enhancement: {self.config.apply_contrast_enhancement}")
        self.logger.debug(f"  Contrast method: {self.config.contrast_method}")
        self.logger.debug(f"  Export metadata: {self.config.export_metadata}")
        self.logger.debug(f"  Create overview images: {self.config.create_overview_images}")

        # Estimate expected products
        expected_rgb = 0
        if self.config.generate_natural_color: expected_rgb += 2
        if self.config.generate_false_color: expected_rgb += 2
        if self.config.generate_water_specific: expected_rgb += 6
        if self.config.generate_research_combinations: expected_rgb += 10

        expected_indices = 0
        if self.config.generate_water_quality_indices: expected_indices += 7
        if self.config.generate_chlorophyll_indices: expected_indices += 3  # PC, FAI removed
        if self.config.generate_turbidity_indices: expected_indices += 2
        if self.config.generate_advanced_indices: expected_indices += 3  # FUI removed, RDI added

        self.logger.debug(f"Expected products: ~{expected_rgb} RGB composites + ~{expected_indices} spectral indices")
        self.logger.debug("Marine visualization processor ready for processing")

    def process_marine_visualizations(self, c2rcc_path: str, output_folder: str,
                                    product_name: str, intermediate_paths: Optional[Dict[str, str]] = None) -> Dict[str, ProcessingResult]:
        """
        Generate RGB composites and spectral indices from geometric products

        Args:
            c2rcc_path: Path to C2RCC output directory (fallback only)
            output_folder: Output directory for visualizations
            product_name: Name of the product being processed
            intermediate_paths: Optional dictionary containing geometric_path

        Returns:
            Dictionary of visualization results (RGB composites + spectral indices)
        """
        try:
            self.logger.debug(f"Starting marine visualization processing for {product_name}")

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
                    rgb_results = self._generate_rgb_composites(bands_data, reference_metadata, rgb_output_dir, scene_name)
                    viz_results.update(rgb_results)
                    rgb_count = len([k for k in rgb_results.keys() if k.startswith('rgb_')])
                    self.logger.info(f"    RGB Composites... done ({rgb_count} products)")

                except Exception as rgb_error:
                    self.logger.error(f"Error generating RGB composites: {rgb_error}")
                    viz_results['rgb_error'] = ProcessingResult(False, "", None, str(rgb_error))

            # Generate spectral indices
            if bands_data and len(bands_data) >= 2:
                self.logger.debug("Generating spectral indices")

                try:
                    index_results = self._generate_spectral_indices(bands_data, reference_metadata, indices_output_dir, scene_name)
                    viz_results.update(index_results)
                    index_count = len([k for k in index_results.keys() if k.startswith('index_')])
                    self.logger.info(f"    Spectral Indices... done ({index_count} products)")

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

            rgb_combinations = {
                'natural_color': {
                    'red': 665, 'green': 560, 'blue': 490,
                    'description': 'Natural color (True color)',
                    'application': 'General visualization, publications',
                    'priority': 'essential',
                    'enabled': self.config.generate_natural_color
                },
                'natural_enhanced': {
                    'red': 665, 'green': 560, 'blue': 443,
                    'description': 'Natural color with enhanced contrast',
                    'application': 'Better water-land contrast',
                    'priority': 'important',
                    'enabled': self.config.generate_natural_color
                },
                'false_color_infrared': {
                    'red': 842, 'green': 665, 'blue': 560,
                    'description': 'False color infrared',
                    'application': 'Vegetation (red), clear water (dark)',
                    'priority': 'important',
                    'enabled': self.config.generate_false_color
                },
                'false_color_nir': {
                    'red': 865, 'green': 665, 'blue': 560,
                    'description': 'False color NIR',
                    'application': 'Enhanced vegetation/water contrast',
                    'priority': 'important',
                    'enabled': self.config.generate_false_color
                },
                'turbidity_enhanced': {
                    'red': 705, 'green': 665, 'blue': 560,
                    'description': 'Turbidity-enhanced RGB',
                    'application': 'Enhanced turbidity visualization',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'chlorophyll_enhanced': {
                    'red': 705, 'green': 665, 'blue': 490,
                    'description': 'Chlorophyll-enhanced RGB',
                    'application': 'Enhanced chlorophyll visualization',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'coastal_aerosol': {
                    'red': 665, 'green': 490, 'blue': 443,
                    'description': 'Coastal aerosol RGB',
                    'application': 'Coastal water analysis',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'water_quality': {
                    'red': 740, 'green': 665, 'blue': 560,
                    'description': 'Water quality RGB',
                    'application': 'General water quality assessment',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'sediment_transport': {
                    'red': 783, 'green': 705, 'blue': 665,
                    'description': 'Sediment transport RGB',
                    'application': 'Sediment plume visualization',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'atmospheric_penetration': {
                    'red': 865, 'green': 783, 'blue': 740,
                    'description': 'Atmospheric penetration RGB',
                    'application': 'Deep water analysis',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'ocean_color_standard': {
                    'red': 490, 'green': 560, 'blue': 443,
                    'description': 'NASA Ocean Color standard composite',
                    'application': 'Ocean color research, international standard protocol',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'algal_bloom_enhanced': {
                    'red': 705, 'green': 665, 'blue': 560,
                    'description': 'Red edge enhanced for algal bloom detection',
                    'application': 'Harmful algal bloom monitoring and early detection',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'coastal_turbidity': {
                    'red': 865, 'green': 740, 'blue': 490,
                    'description': 'NIR-enhanced for coastal sediment monitoring',
                    'application': 'Coastal sediment transport and turbidity assessment',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'deep_water_clarity': {
                    'red': 560, 'green': 490, 'blue': 443,
                    'description': 'Blue-enhanced for deep water penetration',
                    'application': 'Deep water clarity and depth estimation',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'riverine_waters': {
                    'red': 740, 'green': 705, 'blue': 665,
                    'description': 'Red edge focus for riverine environments',
                    'application': 'River discharge monitoring and estuarine mixing',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'cdom_enhanced': {
                    'red': 490, 'green': 443, 'blue': 560,
                    'description': 'Blue-shifted for CDOM visualization',
                    'application': 'Colored dissolved organic matter detection',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'water_change_detection': {
                    'red': 865, 'green': 1610, 'blue': 560,
                    'description': 'SWIR-enhanced for change detection',
                    'application': 'Water body change detection and temporal monitoring',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'advanced_atmospheric': {
                    'red': 1375, 'green': 945, 'blue': 705,
                    'description': 'Atmospheric correction enhanced',
                    'application': 'Atmospheric interference reduction, water vapor correction',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                }
            }

            active_combinations = {name: config for name, config in rgb_combinations.items()
                                if config['enabled'] and config['priority'] in ['essential', 'important', 'marine', 'research']}

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
                            'priority': config['priority'],
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

            # Note: Water masking is controlled by JiangTSSConfig and applied at save time
            # Marine viz no longer applies its own mask
            water_mask = None

            spectral_indices = {
                'NDWI': {
                    'formula': '(B3 - B8A) / (B3 + B8A)',
                    'required_bands': [560, 865],
                    'description': 'Normalized Difference Water Index',
                    'application': 'Water body delineation',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'MNDWI': {
                    'formula': '(B3 - B11) / (B3 + B11)',
                    'required_bands': [560, 1610],
                    'fallback_bands': [560, 865],
                    'description': 'Modified Normalized Difference Water Index',
                    'application': 'Enhanced water detection',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'NDTI': {
                    'formula': '(B4 - B3) / (B4 + B3)',
                    'required_bands': [665, 560],
                    'description': 'Normalized Difference Turbidity Index',
                    'application': 'Turbidity assessment',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'NDMI': {
                    'formula': '(B8A - B11) / (B8A + B11)',
                    'required_bands': [865, 1610],
                    'fallback_bands': [842, 1610],
                    'description': 'Normalized Difference Moisture Index',
                    'application': 'Water vs non-water separation, moisture content',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'AWEI': {
                    'formula': '4 * (B3 - B11) - (0.25 * B8A + 2.75 * B12)',
                    'required_bands': [560, 1610, 865, 2190],
                    'fallback_bands': [560, 1610, 842, 2190],
                    'description': 'Automated Water Extraction Index',
                    'application': 'Enhanced water body extraction',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'WI': {
                    'formula': 'B2 / (B3 + B8A)',
                    'required_bands': [490, 560, 865],
                    'fallback_bands': [490, 560, 842],
                    'description': 'Water Index',
                    'application': 'Turbid water detection and delineation',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'WRI': {
                    'formula': '(B3 + B4) / (B8A + B11)',
                    'required_bands': [560, 665, 865, 1610],
                    'fallback_bands': [560, 665, 842, 1610],
                    'description': 'Water Ratio Index',
                    'application': 'Water/land separation and water quality assessment',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'NDCI': {
                    'formula': '(B5 - B4) / (B5 + B4)',
                    'required_bands': [705, 665],
                    'description': 'Normalized Difference Chlorophyll Index',
                    'application': 'Chlorophyll concentration',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                'CHL_RED_EDGE': {
                    'formula': '(B5 / B4) - 1',
                    'required_bands': [705, 665],
                    'description': 'Chlorophyll Red Edge',
                    'application': 'Chlorophyll using red edge',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                'GNDVI': {
                    'formula': '(B8 - B3) / (B8 + B3)',
                    'required_bands': [842, 560],
                    'description': 'Green Normalized Difference Vegetation Index',
                    'application': 'Aquatic vegetation',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                # PC (Phycocyanin) REMOVED: Requires 620nm band for phycocyanin absorption - S2 lacks this
                # FAI (Floating Algae Index) REMOVED: Formula mathematically undefined for S2 band set
                'TSI': {
                    'formula': '(B4 + B3) / 2',
                    'required_bands': [665, 560],
                    'description': 'Turbidity Spectral Index',
                    'application': 'Turbidity estimation',
                    'enabled': self.config.generate_turbidity_indices,
                    'category': 'turbidity'
                },
                'NGRDI': {
                    'formula': '(B3 - B4) / (B3 + B4)',
                    'required_bands': [560, 665],
                    'description': 'Normalized Green Red Difference Index',
                    'application': 'Water-vegetation separation',
                    'enabled': self.config.generate_turbidity_indices,
                    'category': 'turbidity'
                },
                # FUI (Forel-Ule Index) REMOVED: Requires CIE chromaticity conversion, arctan2 formula scientifically invalid
                'SDD': {
                    'formula': 'ln(0.14 / B4) / 1.7',
                    'required_bands': [665],
                    'description': 'Secchi Disk Depth proxy (Gordon 1989)',
                    'application': 'Water transparency - Kd*SD=1.7 relationship',
                    'enabled': self.config.generate_advanced_indices,
                    'category': 'advanced'
                },
                'CDOM': {
                    'formula': 'B1 / B3',
                    'required_bands': [443, 560],
                    'description': 'Colored Dissolved Organic Matter proxy',
                    'application': 'CDOM concentration',
                    'enabled': self.config.generate_advanced_indices,
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
                b8a = bands_data[bands_to_use[1]]  # 865nm
                return (b3 - b8a) / (b3 + b8a + 1e-8)

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
            elif index_name == 'TSI':
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b4 + b3) / 2

            elif index_name == 'NGRDI':
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b3 - b4) / (b3 + b4 + 1e-8)

            # Advanced Water Properties
            # PC, FAI, FUI removed: require bands/methods not available in Sentinel-2

            elif index_name == 'SDD':
                # Secchi Disk Depth proxy using Gordon (1989) Kd*SD=1.7 relationship
                # 0.14 = empirical reflectance threshold for Secchi visibility
                # Reference: Gordon, H.R. (1989). Limnol. Oceanogr. 34(8):1389-1409
                b4 = bands_data[bands_to_use[0]]  # 665nm
                return np.log(0.14 / (b4 + 1e-8)) / 1.7

            elif index_name == 'CDOM':
                b1 = bands_data[bands_to_use[0]]  # 443nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return b1 / (b3 + 1e-8)

            elif index_name == 'RDI':
                # Relative Depth Index (Stumpf et al. 2003)
                # Without in-situ calibration, produces relative depth index (unitless)
                # Ideal for time-series analysis of bathymetric changes
                blue = bands_data[bands_to_use[0]]   # 490nm (B2)
                green = bands_data[bands_to_use[1]]  # 560nm (B3)
                valid_mask = (blue > 0) & (green > 0)
                rdi = np.full_like(blue, np.nan, dtype=np.float32)
                rdi[valid_mask] = np.log(blue[valid_mask]) / np.log(green[valid_mask])
                # Mask optically deep water where ratio breaks down (>1.0)
                rdi = np.where(rdi > 1.0, np.nan, rdi)
                return rdi

            else:
                self.logger.warning(f"Unknown index calculation for {index_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error calculating {index_name}: {e}")
            return None

    # Note: _create_water_mask() removed - water masking is now controlled by
    # JiangTSSConfig (shapefile mask or NIR threshold) and applied centrally

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

    def _apply_contrast_enhancement(self, rgb_array: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement based on configuration."""
        try:
            if self.config.contrast_method == 'percentile_stretch':
                for i in range(3):
                    channel = rgb_array[:, :, i]
                    valid_mask = channel > 0
                    valid_data = channel[valid_mask]

                    if len(valid_data) > 100:
                        p_low, p_high = self.config.percentile_range
                        p_low_val, p_high_val = np.percentile(valid_data, [p_low, p_high])

                        if p_high_val > p_low_val:
                            enhanced_channel = np.clip(
                                (channel - p_low_val) / (p_high_val - p_low_val),
                                0, 1
                            )
                            rgb_array[:, :, i] = enhanced_channel

            elif self.config.contrast_method == 'histogram_equalization':
                for i in range(3):
                    channel = rgb_array[:, :, i]
                    valid_mask = channel > 0

                    if np.any(valid_mask):
                        valid_data = channel[valid_mask]
                        if len(np.unique(valid_data)) > 10:
                            hist, bins = np.histogram(valid_data, bins=256, range=(0, 1))
                            cdf = hist.cumsum()
                            cdf = cdf / cdf[-1]
                            enhanced_data = np.interp(valid_data, bins[:-1], cdf)
                            channel[valid_mask] = enhanced_data
                            rgb_array[:, :, i] = channel

            return rgb_array

        except Exception as e:
            self.logger.warning(f"Error in additional contrast enhancement: {e}")
            return rgb_array

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

            if self.config.export_metadata:
                dataset.SetMetadataItem('PROCESSING_METHOD', 'Marine RGB Composite')
                dataset.SetMetadataItem('CONTRAST_ENHANCEMENT', str(self.config.apply_contrast_enhancement))
                dataset.SetMetadataItem('CONTRAST_METHOD', self.config.contrast_method)

            if self.config.create_overview_images:
                dataset.BuildOverviews('AVERAGE', [2, 4, 8, 16])

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

            nodata_value = metadata.get('nodata', -9999)
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

            if self.config.export_metadata:
                dataset.SetMetadataItem('PROCESSING_METHOD', 'Spectral Index Calculation')
                dataset.SetMetadataItem('UNITS', 'Dimensionless')

            if self.config.create_overview_images:
                dataset.BuildOverviews('AVERAGE', [2, 4, 8, 16])

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

                f.write("CONFIGURATION SETTINGS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  RGB Format: {self.config.rgb_format}\n")
                f.write(f"  Contrast Enhancement: {self.config.apply_contrast_enhancement}\n")
                f.write(f"  Contrast Method: {self.config.contrast_method}\n")
                f.write(f"  Percentile Range: {self.config.percentile_range}\n")
                f.write(f"  Export Metadata: {self.config.export_metadata}\n")
                f.write(f"  Create Overviews: {self.config.create_overview_images}\n\n")

                f.write("ENABLED FEATURES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Natural Color RGB: {self.config.generate_natural_color}\n")
                f.write(f"  False Color RGB: {self.config.generate_false_color}\n")
                f.write(f"  Water-specific RGB: {self.config.generate_water_specific}\n")
                f.write(f"  Research RGB: {self.config.generate_research_combinations}\n")
                f.write(f"  Water Quality Indices: {self.config.generate_water_quality_indices}\n")
                f.write(f"  Chlorophyll Indices: {self.config.generate_chlorophyll_indices}\n")
                f.write(f"  Turbidity Indices: {self.config.generate_turbidity_indices}\n")
                f.write(f"  Advanced Indices: {self.config.generate_advanced_indices}\n\n")

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
                    logger.warning(f"Partial cleanup: {len(remaining_resampled)} items remain")

            except Exception as e:
                logger.warning(f"Cleanup with shutil failed: {e}")

            # Method 2: System commands for remaining items
            try:
                logger.debug("Using system commands for cleanup...")

                current_items = os.listdir(geometric_folder) if os.path.exists(geometric_folder) else []
                current_resampled = [item for item in current_items if item.startswith('Resampled_')]

                if platform.system() == "Windows":
                    for item in current_resampled:
                        item_path = os.path.join(geometric_folder, item)
                        if os.path.exists(item_path):
                            try:
                                if os.path.isdir(item_path):
                                    subprocess.run(['rmdir', '/s', '/q', f'"{item_path}"'],
                                                shell=True, capture_output=True, timeout=30)
                                else:
                                    subprocess.run(['del', '/f', '/q', f'"{item_path}"'],
                                                shell=True, capture_output=True, timeout=30)
                            except subprocess.TimeoutExpired:
                                logger.warning(f"Timeout deleting {item}")
                            except Exception as e:
                                logger.warning(f"Command failed for {item}: {e}")
                else:
                    for item in current_resampled:
                        item_path = os.path.join(geometric_folder, item)
                        if os.path.exists(item_path):
                            try:
                                subprocess.run(['rm', '-rf', item_path],
                                            capture_output=True, timeout=30)
                            except subprocess.TimeoutExpired:
                                logger.warning(f"Timeout deleting {item}")
                            except Exception as e:
                                logger.warning(f"Command failed for {item}: {e}")

                final_remaining = os.listdir(geometric_folder) if os.path.exists(geometric_folder) else []
                final_resampled = [item for item in final_remaining if item.startswith('Resampled_')]

                if not final_resampled:
                    logger.debug(f"Cleanup successful (Method 2): All geometric products deleted ({total_size_mb:.1f} MB freed)")
                    return True
                else:
                    logger.error(f"Cleanup failed: {len(final_resampled)} items still remain: {final_resampled}")
                    return False

            except Exception as e:
                logger.error(f"System command cleanup failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            import traceback
            logger.error(f"Cleanup traceback: {traceback.format_exc()}")
            return False

    def _calculate_rgb_statistics(self, rgb_array: np.ndarray) -> Dict:
        """Calculate statistics for RGB composite"""
        try:
            stats = {}

            valid_pixels = np.sum(np.any(rgb_array > 0, axis=2))
            total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
            stats['coverage_percent'] = (valid_pixels / total_pixels) * 100

            for i, channel_name in enumerate(['red', 'green', 'blue']):
                channel = rgb_array[:, :, i]
                valid_data = channel[channel > 0]

                if len(valid_data) > 0:
                    stats[f'{channel_name}_min'] = float(np.min(valid_data))
                    stats[f'{channel_name}_max'] = float(np.max(valid_data))
                    stats[f'{channel_name}_mean'] = float(np.mean(valid_data))
                    stats[f'{channel_name}_std'] = float(np.std(valid_data))
                else:
                    stats[f'{channel_name}_min'] = 0.0
                    stats[f'{channel_name}_max'] = 0.0
                    stats[f'{channel_name}_mean'] = 0.0
                    stats[f'{channel_name}_std'] = 0.0

            return stats

        except Exception as e:
            self.logger.warning(f"Error calculating RGB statistics: {e}")
            return {'coverage_percent': 0.0}

    def _calculate_band_statistics(self, data: np.ndarray, band_name: str) -> Dict:
        """Calculate statistics for single band data"""
        try:
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                stats = {
                    'band_name': band_name,
                    'min_value': float(np.min(valid_data)),
                    'max_value': float(np.max(valid_data)),
                    'mean_value': float(np.mean(valid_data)),
                    'std_value': float(np.std(valid_data)),
                    'valid_pixels': len(valid_data),
                    'coverage_percent': (len(valid_data) / data.size) * 100
                }
            else:
                stats = {
                    'band_name': band_name,
                    'min_value': 0.0,
                    'max_value': 0.0,
                    'mean_value': 0.0,
                    'std_value': 0.0,
                    'valid_pixels': 0,
                    'coverage_percent': 0.0
                }

            return stats

        except Exception as e:
            self.logger.warning(f"Error calculating band statistics for {band_name}: {e}")
            return {'band_name': band_name, 'coverage_percent': 0.0}
