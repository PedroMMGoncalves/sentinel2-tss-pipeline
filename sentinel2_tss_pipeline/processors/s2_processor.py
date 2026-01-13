"""
Sentinel-2 Processor for SNAP-based processing pipeline.

Handles L1C → Resampling → Subset → C2RCC processing using SNAP GPT.
Generates processing graphs and manages intermediate products.

Reference:
    C2RCC (Case 2 Regional Coast Colour) processor in SNAP
    SNAP GPT (Graph Processing Tool) for batch processing
"""

import os
import sys
import time
import subprocess
import logging
from typing import Dict, Optional, List, NamedTuple

import numpy as np

from ..config import ProcessingConfig, ProcessingMode
from ..utils.raster_io import RasterIO
from ..utils.product_detector import ProductDetector, SystemMonitor
from .snap_calculator import TSMChlorophyllCalculator, ProcessingResult

logger = logging.getLogger('sentinel2_tss_pipeline')


class ProcessingStatus(NamedTuple):
    """Processing status information"""
    total_products: int
    processed: int
    failed: int
    skipped: int
    current_product: str
    current_stage: str
    progress_percent: float
    eta_minutes: float
    processing_speed: float


class S2Processor:
    """Enhanced S2 processor with complete pipeline"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = time.time()
        self.current_product = ""
        self.current_stage = ""

        # Track intermediate products
        self.intermediate_products = {}

        # System monitor (set externally by UnifiedS2TSSProcessor)
        self.system_monitor = None

        # Validate SNAP installation
        self.validate_snap_installation()

        # Create processing graphs
        self.setup_processing_graphs()

    def validate_snap_installation(self):
        """Enhanced SNAP validation"""
        snap_home = os.environ.get('SNAP_HOME')
        if not snap_home:
            logger.error("SNAP_HOME environment variable not set!")
            raise RuntimeError("SNAP installation not found")

        logger.info(f"SNAP_HOME: {snap_home}")

        gpt_cmd = self.get_gpt_command()
        if not os.path.exists(gpt_cmd):
            logger.error(f"GPT executable not found: {gpt_cmd}")
            raise RuntimeError("GPT executable not found")

        try:
            result = subprocess.run([gpt_cmd, '-h'],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"GPT validated: {gpt_cmd}")
            else:
                logger.error(f"GPT validation failed: {result.stderr}")
                raise RuntimeError("GPT validation failed")
        except subprocess.TimeoutExpired:
            logger.error("GPT validation timeout")
            raise RuntimeError("GPT validation timeout")
        except Exception as e:
            logger.error(f"GPT validation error: {e}")
            raise RuntimeError(f"GPT validation error: {e}")

    def get_gpt_command(self) -> str:
        """Get GPT command for the operating system"""
        snap_home = os.environ.get('SNAP_HOME')
        if sys.platform.startswith('win'):
            return os.path.join(snap_home, 'bin', 'gpt.exe')
        else:
            return os.path.join(snap_home, 'bin', 'gpt')

    def setup_processing_graphs(self):
        """Create processing graphs based on configuration"""
        mode = self.config.processing_mode

        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            if self.config.subset_config.geometry_wkt or self.config.subset_config.pixel_start_x is not None:
                self.main_graph_file = self.create_s2_graph_with_subset()
            else:
                self.main_graph_file = self.create_s2_graph_no_subset()

        logger.info(f"Processing graph created for mode: {mode.value}")

    def create_s2_graph_with_subset(self) -> str:
        """Create S2 processing graph with spatial subset"""
        subset_params = self._get_subset_parameters()

        graph_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<graph id="S2_Complete_Processing_WithSubset">
  <version>1.0</version>

  <!-- Step 1: Read Input Product -->
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{sourceProduct}}</file>
    </parameters>
  </node>

  <!-- Step 2: S2 Resampling -->
  <node id="S2Resampling">
    <operator>S2Resampling</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <resolution>{self.config.resampling_config.target_resolution}</resolution>
      <upsampling>{self.config.resampling_config.upsampling_method}</upsampling>
      <downsampling>{self.config.resampling_config.downsampling_method}</downsampling>
      <flagDownsampling>{self.config.resampling_config.flag_downsampling}</flagDownsampling>
      <resampleOnPyramidLevels>{str(self.config.resampling_config.resample_on_pyramid_levels).lower()}</resampleOnPyramidLevels>
    </parameters>
  </node>

  <!-- Step 3: Spatial Subset -->
  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="S2Resampling"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {subset_params}
      <subSamplingX>{self.config.subset_config.sub_sampling_x}</subSamplingX>
      <subSamplingY>{self.config.subset_config.sub_sampling_y}</subSamplingY>
      <fullSwath>{str(self.config.subset_config.full_swath).lower()}</fullSwath>
      <copyMetadata>{str(self.config.subset_config.copy_metadata).lower()}</copyMetadata>
    </parameters>
  </node>

  <!-- Step 3.5: SAVE GEOMETRIC PRODUCTS -->
<node id="WriteGeometric">
  <operator>Write</operator>
  <sources>
    <source refid="Subset"/>
  </sources>
  <parameters class="com.bc.ceres.binding.dom.XppDomElement">
    <file>${{geometricProduct}}</file>
    <formatName>BEAM-DIMAP</formatName>
  </parameters>
</node>

  <!-- Step 4: C2RCC Atmospheric Correction -->
  <node id="c2rcc_msi">
    <operator>c2rcc.msi</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {self._get_c2rcc_parameters()}
    </parameters>
  </node>

  <!-- Step 5: Write Output -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="c2rcc_msi"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{targetProduct}}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

</graph>'''

        graph_file = 's2_complete_processing_with_subset.xml'
        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write(graph_content)

        logger.info(f"Complete processing graph saved: {graph_file}")
        return graph_file

    def create_s2_graph_no_subset(self) -> str:
        """Create S2 processing graph without spatial subset"""
        graph_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<graph id="S2_Complete_Processing_NoSubset">
  <version>1.0</version>

  <!-- Step 1: Read Input Product -->
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{sourceProduct}}</file>
    </parameters>
  </node>

  <!-- Step 2: S2 Resampling -->
  <node id="S2Resampling">
    <operator>S2Resampling</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <resolution>{self.config.resampling_config.target_resolution}</resolution>
      <upsampling>{self.config.resampling_config.upsampling_method}</upsampling>
      <downsampling>{self.config.resampling_config.downsampling_method}</downsampling>
      <flagDownsampling>{self.config.resampling_config.flag_downsampling}</flagDownsampling>
      <resampleOnPyramidLevels>{str(self.config.resampling_config.resample_on_pyramid_levels).lower()}</resampleOnPyramidLevels>
    </parameters>
  </node>

  <!-- Step 2.5: SAVE GEOMETRIC PRODUCTS (for RGB composites and spectral indices) -->
  <node id="WriteGeometric">
    <operator>Write</operator>
    <sources>
      <source refid="S2Resampling"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{geometricProduct}}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

  <!-- Step 3: C2RCC Atmospheric Correction -->
  <node id="c2rcc_msi">
    <operator>c2rcc.msi</operator>
    <sources>
      <sourceProduct refid="S2Resampling"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {self._get_c2rcc_parameters()}
    </parameters>
  </node>

  <!-- Step 4: Write Output -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="c2rcc_msi"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{targetProduct}}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

</graph>'''

        graph_file = 's2_complete_processing_no_subset.xml'
        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write(graph_content)

        logger.info(f"Complete processing graph saved: {graph_file}")
        return graph_file

    def _get_subset_parameters(self) -> str:
        """Generate subset parameters for XML with proper escaping"""
        subset_config = self.config.subset_config

        if subset_config.geometry_wkt:
            escaped_wkt = subset_config.geometry_wkt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            return f"<geoRegion>{escaped_wkt}</geoRegion>"
        elif subset_config.pixel_start_x is not None:
            return f"<region>{subset_config.pixel_start_x},{subset_config.pixel_start_y},{subset_config.pixel_size_x},{subset_config.pixel_size_y}</region>"
        else:
            return ""

    def _get_c2rcc_parameters(self) -> str:
        """Generate complete C2RCC parameters with correct SNAP parameter names"""
        c2rcc = self.config.c2rcc_config

        # Escape XML special characters
        valid_pixel_expr = c2rcc.valid_pixel_expression.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Build complete parameter set using exact SNAP parameter names (CORRECT ORDER)
        params = f'''<validPixelExpression>{valid_pixel_expr}</validPixelExpression>
        <salinity>{c2rcc.salinity}</salinity>
        <temperature>{c2rcc.temperature}</temperature>
        <ozone>{c2rcc.ozone}</ozone>
        <press>{c2rcc.pressure}</press>
        <elevation>{c2rcc.elevation}</elevation>
        <TSMfac>{c2rcc.tsm_fac}</TSMfac>
        <TSMexp>{c2rcc.tsm_exp}</TSMexp>
        <CHLexp>{c2rcc.chl_exp}</CHLexp>
        <CHLfac>{c2rcc.chl_fac}</CHLfac>
        <thresholdRtosaOOS>{c2rcc.threshold_rtosa_oos}</thresholdRtosaOOS>
        <thresholdAcReflecOos>{c2rcc.threshold_ac_reflec_oos}</thresholdAcReflecOos>
        <thresholdCloudTDown865>{c2rcc.threshold_cloud_tdown865}</thresholdCloudTDown865>
        <netSet>{c2rcc.net_set}</netSet>
        <useEcmwfAuxData>{str(c2rcc.use_ecmwf_aux_data).lower()}</useEcmwfAuxData>
        <demName>{c2rcc.dem_name}</demName>
        <outputAsRrs>{str(c2rcc.output_as_rrs).lower()}</outputAsRrs>
        <deriveRwFromPathAndTransmittance>{str(c2rcc.derive_rw_from_path_and_transmittance).lower()}</deriveRwFromPathAndTransmittance>
        <outputRtoa>{str(c2rcc.output_rtoa).lower()}</outputRtoa>
        <outputRtosaGc>{str(c2rcc.output_rtosa_gc).lower()}</outputRtosaGc>
        <outputRtosaGcAann>{str(c2rcc.output_rtosa_gc_aann).lower()}</outputRtosaGcAann>
        <outputRpath>{str(c2rcc.output_rpath).lower()}</outputRpath>
        <outputTdown>{str(c2rcc.output_tdown).lower()}</outputTdown>
        <outputTup>{str(c2rcc.output_tup).lower()}</outputTup>
        <outputAcReflectance>{str(c2rcc.output_ac_reflectance).lower()}</outputAcReflectance>
        <outputRhown>{str(c2rcc.output_rhown).lower()}</outputRhown>
        <outputOos>{str(c2rcc.output_oos).lower()}</outputOos>
        <outputKd>{str(c2rcc.output_kd).lower()}</outputKd>
        <outputUncertainties>{str(c2rcc.output_uncertainties).lower()}</outputUncertainties>'''

        # Add optional paths if specified
        if c2rcc.atmospheric_aux_data_path:
            params += f'\n      <atmosphericAuxDataPath>{c2rcc.atmospheric_aux_data_path}</atmosphericAuxDataPath>'

        if c2rcc.alternative_nn_path:
            params += f'\n      <alternativeNNPath>{c2rcc.alternative_nn_path}</alternativeNNPath>'

        return params

    def _get_resampled_output_path(self, input_path: str, output_folder: str) -> str:
        """Get the resampled product output path."""
        product_name = os.path.basename(input_path)
        clean_name = self._extract_clean_product_name(product_name)

        # Check both possible geometric output names
        subset_path = os.path.join(output_folder, "Geometric_Products", f"Resampled_{clean_name}_Subset.dim")
        resample_path = os.path.join(output_folder, "Geometric_Products", f"Resampled_{clean_name}.dim")

        # Return the one that exists, or default to subset version
        if os.path.exists(subset_path):
            return subset_path
        elif os.path.exists(resample_path):
            return resample_path
        else:
            # Default assumption: subset is applied
            return subset_path

    def _extract_clean_product_name(self, product_name: str) -> str:
        """Extract clean product name from input path"""
        clean_name = os.path.basename(product_name)
        clean_name = clean_name.replace('.zip', '').replace('.SAFE', '')

        # Create cleaner name from Sentinel-2 product name
        if 'MSIL1C' in clean_name:
            parts = clean_name.split('_')
            if len(parts) >= 6:
                # S2A_MSIL1C_20190105T112441_N0500_R037_T29TNF_20221215T164753
                # becomes: S2A_20190105T112441_T29TNF
                clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
            else:
                clean_name = clean_name.replace('MSIL1C_', '')

        return clean_name

    def get_output_filename(self, input_path: str, output_dir: str, stage: str) -> str:
        """Generate output filename based on processing stage"""
        basename = os.path.basename(input_path)

        # Extract base product name
        if basename.endswith('.zip'):
            product_name = basename.replace('.zip', '')
        elif basename.endswith('.SAFE'):
            product_name = basename.replace('.SAFE', '')
        elif basename.endswith('.dim'):
            product_name = basename.replace('.dim', '')
        else:
            product_name = basename

        # Remove MSIL1C prefix for cleaner naming
        if 'MSIL1C' in product_name:
            # Extract key parts: S2A_MSIL1C_20230615T113321_N0509_R080_T29TNE_20230615T134426
            parts = product_name.split('_')
            if len(parts) >= 6:
                # Create cleaner name: S2A_20230615T113321_T29TNE
                clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
            else:
                clean_name = product_name.replace('MSIL1C_', '')
        else:
            clean_name = product_name

        # Stage-specific naming
        if stage == "geometric":
            output_name = f"Resampled_{clean_name}_Subset.dim"
            return os.path.join(output_dir, "Geometric_Products", output_name)
        elif stage == "c2rcc":
            output_name = f"Resampled_{clean_name}_Subset_C2RCC.dim"
            return os.path.join(output_dir, "C2RCC_Products", output_name)
        else:
            return os.path.join(output_dir, f"{clean_name}_{stage}.dim")

    def process_single_product(self, input_path: str, output_folder: str) -> Dict[str, ProcessingResult]:
        """
        Process single product through complete S2 pipeline

        Responsibilities:
        - L1C -> Resampling -> Subset -> C2RCC
        - Calculate SNAP TSM/CHL from IOPs
        - Verify C2RCC output
        """
        processing_start = time.time()
        results = {}

        try:
            product_name = os.path.basename(input_path)
            self.current_product = product_name

            logger.info(f"Processing: {product_name}")
            logger.info(f"  Mode: {self.config.processing_mode.value}")
            logger.info(f"  Resolution: {self.config.resampling_config.target_resolution}m")
            logger.info(f"  ECMWF: {self.config.c2rcc_config.use_ecmwf_aux_data}")

            # Check system health before processing
            if self.system_monitor:
                healthy, warnings = self.system_monitor.check_system_health()
                if not healthy:
                    logger.warning("System health issues detected:")
                    for warning in warnings:
                        logger.warning(f"  - {warning}")

            # Ensure output directories exist
            os.makedirs(os.path.join(output_folder, "Geometric_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "C2RCC_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "TSS_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "Logs"), exist_ok=True)

            # Step 1: S2 Processing (Resampling + Subset + C2RCC)
            self.current_stage = "S2 Processing (Complete)"
            c2rcc_output_path = self.get_output_filename(input_path, output_folder, "c2rcc")

            # Check if output already exists and is valid
            if self.config.skip_existing and os.path.exists(c2rcc_output_path):
                file_size = os.path.getsize(c2rcc_output_path)
                if file_size > 1024 * 1024:  # > 1MB
                    logger.info(f"C2RCC output exists ({file_size/1024/1024:.1f}MB), skipping S2 processing")
                    self.skipped_count += 1

                    # Verify required bands exist for potential TSS processing
                    data_folder = c2rcc_output_path.replace('.dim', '.data')
                    required_bands = ['conc_tsm.img', 'conc_chl.img', 'unc_tsm.img', 'unc_chl.img']
                    if self.config.jiang_config.enable_jiang_tss:
                        required_bands.extend(['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img',
                                            'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img'])

                    missing_bands = []
                    for band in required_bands:
                        if not os.path.exists(os.path.join(data_folder, band)):
                            missing_bands.append(band)

                    if missing_bands:
                        logger.warning(f"Missing bands for TSS processing: {missing_bands}")
                        logger.warning("Will reprocess to ensure all required bands are available")
                    else:
                        # Verify and return existing output
                        c2rcc_stats = self._verify_c2rcc_output(c2rcc_output_path)
                        if c2rcc_stats:
                            results['s2_processing'] = ProcessingResult(True, c2rcc_output_path,
                                                                    c2rcc_stats, None)

                            # S2Processor stops here - no TSS processing
                            # TSS processing is handled by UnifiedS2TSSProcessor/JiangTSSProcessor
                            logger.info("S2 processing completed - ready for TSS processing")

                            return results
                else:
                    logger.warning(f"Removing incomplete C2RCC output file ({file_size} bytes)")
                    os.remove(c2rcc_output_path)

            # Run S2 processing
            logger.info("Starting S2 processing with C2RCC...")
            s2_success = self._run_s2_processing(input_path, c2rcc_output_path)

            processing_time = time.time() - processing_start

            if s2_success:
                # Verify C2RCC output and extract SNAP TSM/CHL info
                c2rcc_stats = self._verify_c2rcc_output(c2rcc_output_path)

                if c2rcc_stats:
                    # Initial S2 processing success
                    logger.info(f"S2 processing SUCCESS: {product_name}")
                    logger.info(f"   Output: {os.path.basename(c2rcc_output_path)} ({c2rcc_stats['file_size_mb']:.1f}MB)")
                    logger.info(f"   Processing time: {processing_time/60:.1f} minutes")

                    self.processed_count += 1
                    results['s2_processing'] = ProcessingResult(True, c2rcc_output_path, c2rcc_stats, None)

                    # Calculate SNAP TSM/CHL from IOPs if missing
                    if not c2rcc_stats['has_tsm'] or not c2rcc_stats['has_chl']:
                        logger.info("Calculating missing SNAP TSM/CHL concentrations from IOP products...")
                        snap_calculator = TSMChlorophyllCalculator(
                            tsm_fac=self.config.c2rcc_config.tsm_fac,
                            tsm_exp=self.config.c2rcc_config.tsm_exp,
                            chl_fac=self.config.c2rcc_config.chl_fac,
                            chl_exp=self.config.c2rcc_config.chl_exp
                        )

                        snap_tsm_chl_results = snap_calculator.calculate_snap_tsm_chl(c2rcc_output_path)
                        results.update(snap_tsm_chl_results)

                        # Log SNAP TSM/CHL results
                        if 'snap_tsm' in snap_tsm_chl_results and snap_tsm_chl_results['snap_tsm'].success:
                            logger.info("SNAP TSM calculation successful!")

                        if 'snap_chl' in snap_tsm_chl_results and snap_tsm_chl_results['snap_chl'].success:
                            logger.info("SNAP CHL calculation successful!")

                        # Re-verify after calculating TSM/CHL
                        final_stats = self._verify_c2rcc_output(c2rcc_output_path)
                        if final_stats:
                            # Store intermediate paths for marine visualization
                            self.intermediate_products['geometric_path'] = self._get_resampled_output_path(input_path, output_folder)
                            self.intermediate_products['c2rcc_path'] = c2rcc_output_path

                            # Create enhanced ProcessingResult with intermediate paths
                            result = ProcessingResult(True, c2rcc_output_path, final_stats, None)
                            result.intermediate_paths = self.intermediate_products.copy()
                            results['s2_processing'] = result
                            logger.info("UPDATED SNAP PRODUCTS STATUS:")
                            logger.info(f"   TSM={final_stats['has_tsm']}, CHL={final_stats['has_chl']}, Uncertainties={final_stats['has_uncertainties']}")

                    # S2Processor only does S2 processing - TSS handled separately
                    logger.info("S2 processing completed - ready for TSS processing")
                    logger.info("TSS processing (Jiang + Marine Viz) handled by separate processors")

                    # Clean up memory after S2 processing
                    try:
                        import gc
                        gc.collect()
                        logger.debug("Memory cleanup completed after S2 processing")
                    except Exception as cleanup_error:
                        logger.debug(f"Memory cleanup warning: {cleanup_error}")

                else:
                    logger.error("C2RCC output verification failed")
                    self.failed_count += 1
                    results['s2_error'] = ProcessingResult(False, "", None, "C2RCC verification failed")
            else:
                logger.error(f"S2 processing failed: {product_name}")
                self.failed_count += 1
                results['s2_error'] = ProcessingResult(False, "", None, "S2 processing failed")

            return results

        except Exception as e:
            error_msg = f"Unexpected error processing {self.current_product}: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.failed_count += 1
            return {'s2_error': ProcessingResult(False, "", None, error_msg)}

    def _run_s2_processing(self, input_path: str, output_path: str) -> bool:
        """Run S2 processing using GPT"""
        try:
            # Prepare GPT command
            gpt_cmd = self.get_gpt_command()

            # Get the main output folder (parent of C2RCC_Products)
            # output_path is like: /path/to/output/C2RCC_Products/product.dim
            # We need: /path/to/output/ (the main output folder)
            output_folder = os.path.dirname(os.path.dirname(output_path))
            geometric_output_path = self._get_resampled_output_path(input_path, output_folder)

            # Ensure geometric products directory exists
            os.makedirs(os.path.dirname(geometric_output_path), exist_ok=True)

            cmd = [
                gpt_cmd,
                self.main_graph_file,
                f'-PsourceProduct={input_path}',
                f'-PtargetProduct={output_path}',
                f'-PgeometricProduct={geometric_output_path}',
                f'-c', f'{self.config.memory_limit_gb}G',
                f'-q', str(self.config.thread_count)
            ]

            logger.info(f"GPT processing paths:")
            logger.info(f"  Input: {os.path.basename(input_path)}")
            logger.info(f"  C2RCC Output: {os.path.basename(output_path)}")
            logger.info(f"  Geometric Output: {os.path.basename(geometric_output_path)}")
            logger.debug(f"GPT command: {' '.join(cmd)}")

            # Run GPT processing with timeout
            logger.info("Executing GPT processing...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Check processing results
            if result.returncode == 0:
                # Check main C2RCC output
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 1024 * 1024:  # > 1MB
                        logger.info(f"C2RCC output created: {file_size / (1024*1024):.1f} MB")

                        # Also check if geometric product was created
                        if os.path.exists(geometric_output_path):
                            geom_size = os.path.getsize(geometric_output_path)
                            logger.info(f"Geometric output created: {geom_size / (1024*1024):.1f} MB")
                        else:
                            logger.warning("Geometric product not created (but C2RCC succeeded)")

                        return True
                    else:
                        logger.error(f"C2RCC output file too small ({file_size} bytes)")
                        return False
                else:
                    logger.error(f"C2RCC output file not created")
                    return False
            else:
                logger.error(f"GPT processing failed")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"GPT stderr: {result.stderr[:1000]}...")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"GPT processing timeout")
            return False
        except Exception as e:
            logger.error(f"GPT processing error: {str(e)}")
            return False

    def _verify_c2rcc_output(self, c2rcc_path: str) -> Optional[Dict]:
        """Enhanced C2RCC output verification with automatic fixes"""
        try:
            if not os.path.exists(c2rcc_path):
                logger.error(f"C2RCC file not found: {c2rcc_path}")
                return None

            # Basic file info
            file_size = os.path.getsize(c2rcc_path)
            file_size_mb = file_size / (1024 * 1024)
            data_folder = c2rcc_path.replace('.dim', '.data')

            logger.info("=" * 60)
            logger.info("C2RCC OUTPUT VERIFICATION REPORT")
            logger.info("=" * 60)
            logger.info(f"C2RCC file: {os.path.basename(c2rcc_path)} ({file_size_mb:.1f} MB)")
            logger.info(f"Data folder: {os.path.basename(data_folder)}")

            # Check critical products
            critical_products = {
                'conc_tsm.img': 'TSM',
                'conc_chl.img': 'CHL',
                'unc_tsm.img': 'TSM uncertainties',
                'unc_chl.img': 'CHL uncertainties'
            }

            logger.info("\nSNAP TSM/CHL PRODUCTS (Critical for Analysis):")

            has_tsm = False
            has_chl = False
            has_uncertainties = True  # Start optimistic

            for filename, description in critical_products.items():
                file_path = os.path.join(data_folder, filename)
                exists, size_kb, is_valid = self._verify_file_integrity(file_path, min_size_kb=10)

                if exists and is_valid:
                    logger.info(f"   + {description}: {filename} ({size_kb:.1f} KB)")
                    if 'tsm' in filename and 'unc' not in filename:
                        has_tsm = True
                    elif 'chl' in filename and 'unc' not in filename:
                        has_chl = True
                elif exists and not is_valid:
                    logger.warning(f"   ! {description}: {filename} ({size_kb:.1f} KB) - File too small!")
                    if 'unc' in filename:
                        has_uncertainties = False
                else:
                    logger.warning(f"   - {description}: {filename} (missing/empty)")
                    if 'unc' in filename:
                        has_uncertainties = False
                    elif 'tsm' in filename:
                        has_tsm = False
                    elif 'chl' in filename:
                        has_chl = False

            # Success message or issues
            if has_tsm and has_chl:
                if has_uncertainties:
                    logger.info("SUCCESS: Both TSM and CHL products available with uncertainties!")
                else:
                    logger.info("SUCCESS: Both TSM and CHL products available!")
                    logger.info("   ! Uncertainties: Not available (calculated products)")
            else:
                logger.info("TSM/CHL products will be calculated from IOP products")
                logger.info("   Using SNAP formulas:")
                logger.info("   - TSM = TSMfac * (bpart + bwit)^TSMexp")
                logger.info("   - CHL = apig^CHLexp * CHLfac")

            # Check IOP products
            logger.info("\nSNAP IOP PRODUCTS:")
            iop_products = ['iop_apig.img', 'iop_adet.img', 'iop_agelb.img',
                        'iop_bpart.img', 'iop_bwit.img', 'iop_btot.img']

            iop_count = 0
            for iop_file in iop_products:
                file_path = os.path.join(data_folder, iop_file)
                exists, size_kb, is_valid = self._verify_file_integrity(file_path, min_size_kb=100)

                if exists and is_valid:
                    logger.info(f"   + {iop_file} ({size_kb:.1f} KB)")
                    iop_count += 1
                else:
                    if iop_file == 'iop_btot.img':
                        # Try to calculate missing btot
                        if self._calculate_missing_btot(c2rcc_path):
                            logger.info(f"   + {iop_file} (calculated from bpart + bwit)")
                            iop_count += 1
                        else:
                            logger.info(f"   - {iop_file} (missing/empty)")
                    else:
                        logger.info(f"   - {iop_file} (missing/empty)")

            # Check rhow bands for Jiang TSS
            logger.info("\nJIANG TSS READINESS:")
            rhow_bands = self._check_rhow_bands_availability(c2rcc_path)
            rhow_count = len(rhow_bands)

            if rhow_count == 8:
                logger.info(f"EXCELLENT: All {rhow_count}/8 rhow bands available")
                ready_for_jiang = True
            elif rhow_count >= 6:
                logger.info(f"GOOD: {rhow_count}/8 rhow bands available (sufficient for processing)")
                ready_for_jiang = True
            elif rhow_count > 0:
                logger.warning(f"PARTIAL: Only {rhow_count}/8 rhow bands available")
                missing_bands = []
                for wl in [443, 490, 560, 665, 705, 740, 783, 865]:
                    if wl not in rhow_bands:
                        band_names = ['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img',
                                    'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img']
                        wavelengths = [443, 490, 560, 665, 705, 740, 783, 865]
                        if wl in wavelengths:
                            missing_bands.append(band_names[wavelengths.index(wl)])
                logger.warning(f"   Missing bands: {missing_bands}")
                ready_for_jiang = False
            else:
                logger.warning("PARTIAL: Only 0/8 rhow bands available")
                logger.warning("   Missing bands: ['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img', 'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img']")
                ready_for_jiang = False

            # Overall assessment
            logger.info("\nOVERALL ASSESSMENT:")

            if has_tsm and has_chl and has_uncertainties and ready_for_jiang and iop_count >= 5:
                logger.info("EXCELLENT: Complete processing capabilities")
                logger.info("   - SNAP TSM/CHL: Available")
                logger.info("   - Jiang TSS: Ready")
                logger.info("   - Advanced algorithms: Can proceed")
                overall_status = "excellent"
            elif has_tsm and has_chl and ready_for_jiang:
                logger.info("GOOD: Jiang TSS ready, SNAP products will be calculated")
                overall_status = "good"
            elif ready_for_jiang:
                logger.info("ADEQUATE: Jiang TSS ready, some products missing")
                overall_status = "adequate"
            else:
                logger.warning("LIMITED: Missing critical components")
                overall_status = "limited"

            logger.info("=" * 60)

            # Return comprehensive stats
            stats = {
                'file_size_mb': file_size_mb,
                'has_tsm': has_tsm,
                'has_chl': has_chl,
                'has_uncertainties': has_uncertainties,
                'iop_count': iop_count,
                'rhow_bands_count': rhow_count,
                'ready_for_jiang_tss': ready_for_jiang,
                'overall_status': overall_status,
                'rhow_bands': rhow_bands
            }

            return stats

        except Exception as e:
            logger.error(f"Unexpected error verifying C2RCC output {c2rcc_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _verify_file_integrity(self, file_path: str, min_size_kb: int = 1) -> tuple:
        """Verify file exists and has reasonable size"""
        try:
            if not os.path.exists(file_path):
                return False, 0.0, False

            size_bytes = os.path.getsize(file_path)
            size_kb = size_bytes / 1024
            is_valid = size_kb >= min_size_kb

            return True, size_kb, is_valid

        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False, 0.0, False

    def _calculate_missing_btot(self, c2rcc_path: str) -> bool:
        """Calculate missing iop_btot from bpart + bwit"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            bpart_path = os.path.join(data_folder, 'iop_bpart.img')
            bwit_path = os.path.join(data_folder, 'iop_bwit.img')
            btot_path = os.path.join(data_folder, 'iop_btot.img')

            # Check if btot already exists and is valid
            if os.path.exists(btot_path) and os.path.getsize(btot_path) > 1024:
                logger.info("iop_btot.img already exists")
                return True

            # Check if source files exist
            if not os.path.exists(bpart_path):
                logger.error(f"Missing bpart file: {bpart_path}")
                return False

            if not os.path.exists(bwit_path):
                logger.error(f"Missing bwit file: {bwit_path}")
                return False

            logger.info("Calculating missing iop_btot from bpart + bwit...")

            # Load bpart and bwit
            bpart_data, bpart_meta = RasterIO.read_raster(bpart_path)
            bwit_data, _ = RasterIO.read_raster(bwit_path)

            # Validate data loaded correctly
            if not isinstance(bpart_data, np.ndarray):
                logger.error(f"Invalid bpart_data type: {type(bpart_data)}")
                return False

            if not isinstance(bwit_data, np.ndarray):
                logger.error(f"Invalid bwit_data type: {type(bwit_data)}")
                return False

            # Check shapes match
            if bpart_data.shape != bwit_data.shape:
                logger.error(f"Shape mismatch: bpart {bpart_data.shape} vs bwit {bwit_data.shape}")
                return False

            # Calculate btot = bpart + bwit
            btot_data = bpart_data + bwit_data

            # Validate metadata
            if not isinstance(bpart_meta, dict):
                logger.error(f"Invalid metadata type: {type(bpart_meta)}")
                return False

            logger.info(f"Calculated btot: shape={btot_data.shape}, valid_pixels={np.sum(~np.isnan(btot_data))}")

            # Write result
            success = RasterIO.write_raster(
                data=btot_data,
                output_path=btot_path,
                metadata=bpart_meta,
                description="Total backscattering coefficient (bpart + bwit) in m^-1",
                nodata=-9999
            )

            if success and os.path.exists(btot_path):
                file_size_kb = os.path.getsize(btot_path) / 1024
                logger.info(f"Successfully calculated and saved iop_btot.img ({file_size_kb:.1f} KB)")
                return True
            else:
                logger.error("Failed to save iop_btot.img")
                return False

        except Exception as e:
            logger.error(f"Error calculating btot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _get_file_size_kb(self, file_path: str) -> float:
        """Get file size in KB"""
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path) / 1024
            return 0.0
        except Exception:
            return 0.0

    def _check_rhow_bands_availability(self, c2rcc_path: str) -> Dict[int, str]:
        """Load water reflectance bands from SNAP C2RCC output"""
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"

        if not os.path.exists(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return {}

        # Include rrs_ bands as first priority since both rhow and rrs are equally valid
        # for the Jiang algorithm (both are water-leaving reflectance, just different units)
        band_mapping = {
            443: ['rhow_B1.img', 'rrs_B1.img', 'rhown_B1.img', 'rho_toa_B1.img', 'rtoa_B1.img'],
            490: ['rhow_B2.img', 'rrs_B2.img', 'rhown_B2.img', 'rho_toa_B2.img', 'rtoa_B2.img'],
            560: ['rhow_B3.img', 'rrs_B3.img', 'rhown_B3.img', 'rho_toa_B3.img', 'rtoa_B3.img'],
            665: ['rhow_B4.img', 'rrs_B4.img', 'rhown_B4.img', 'rho_toa_B4.img', 'rtoa_B4.img'],
            705: ['rhow_B5.img', 'rrs_B5.img', 'rhown_B5.img', 'rho_toa_B5.img', 'rtoa_B5.img'],
            740: ['rhow_B6.img', 'rrs_B6.img', 'rhown_B6.img', 'rho_toa_B6.img', 'rtoa_B6.img'],
            783: ['rhow_B7.img', 'rrs_B7.img', 'rho_toa_B7.img', 'rtoa_B7.img'],  # No rhown_B7
            865: ['rhow_B8A.img', 'rrs_B8A.img', 'rho_toa_B8A.img', 'rtoa_B8A.img']  # No rhown_B8A
        }

        rhow_bands = {}
        logger.info(f"Checking for water reflectance bands in: {data_folder}")

        found_band_types = {}  # Track which type of bands we're finding

        for wavelength, possible_names in band_mapping.items():
            for name in possible_names:
                file_path = os.path.join(data_folder, name)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # >1KB
                    rhow_bands[wavelength] = file_path

                    # Track band type for logging
                    if name.startswith('rhow_'):
                        band_type = 'rhow'
                    elif name.startswith('rrs_'):
                        band_type = 'rrs'
                    elif name.startswith('rhown_'):
                        band_type = 'rhown'
                    else:
                        band_type = 'toa'

                    found_band_types[wavelength] = band_type
                    logger.info(f"Found band {wavelength}nm: {name} ({band_type})")
                    break
            else:
                logger.warning(f"Missing band {wavelength}nm")

        # Summary of band types found
        type_counts = {}
        for band_type in found_band_types.values():
            type_counts[band_type] = type_counts.get(band_type, 0) + 1

        logger.info(f"Band type summary: {type_counts}")

        # Check if we have mixed band types (which might need unit conversion)
        unique_types = set(found_band_types.values())
        if len(unique_types) > 1:
            logger.warning(f"Mixed band types detected: {unique_types}")
            logger.warning("   This may require unit conversions in processing")

        return rhow_bands

    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status with division by zero protection"""
        total = self.processed_count + self.failed_count + self.skipped_count
        if total == 0:
            return ProcessingStatus(0, 0, 0, 0, "", "", 0.0, 0.0, 0.0)

        elapsed_time = time.time() - self.start_time

        # Calculate ETA with protection against division by zero
        if self.processed_count > 0 and elapsed_time > 0:
            avg_time_per_product = elapsed_time / self.processed_count
            # Estimate based on a typical batch size
            eta_minutes = avg_time_per_product / 60
            processing_speed = (self.processed_count / elapsed_time) * 60  # products per minute
        else:
            eta_minutes = 0.0
            processing_speed = 0.0

        # Fix: Ensure no division by zero in progress calculation
        progress_percent = (total / max(total, 1)) * 100 if total > 0 else 0.0

        return ProcessingStatus(
            total_products=total,
            processed=self.processed_count,
            failed=self.failed_count,
            skipped=self.skipped_count,
            current_product=self.current_product,
            current_stage=self.current_stage,
            progress_percent=progress_percent,
            eta_minutes=eta_minutes,
            processing_speed=processing_speed
        )

    def cleanup(self):
        """Cleanup resources"""
        # Clean up graph files
        graph_files = [getattr(self, 'main_graph_file', None)]
        for graph_file in graph_files:
            if graph_file and os.path.exists(graph_file):
                try:
                    os.remove(graph_file)
                except Exception:
                    pass
