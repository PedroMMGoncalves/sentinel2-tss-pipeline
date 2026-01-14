"""
Unified S2-TSS Processor.

Main processor that coordinates complete S2 processing and TSS estimation.
Orchestrates the pipeline: L1C -> C2RCC -> Jiang TSS.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

import os
import gc
import time
import logging
from typing import Dict, List

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ..config import ProcessingConfig, ProcessingMode
from ..utils.product_detector import ProductDetector, SystemMonitor
from ..utils.memory_manager import MemoryManager
from ..utils.output_structure import OutputStructure
from ..processors.snap_calculator import ProcessingResult
from ..processors.s2_processor import S2Processor, ProcessingStatus
from ..processors.jiang_processor import JiangTSSProcessor

logger = logging.getLogger('sentinel2_tss_pipeline')


class UnifiedS2TSSProcessor:
    """Main processor that coordinates complete S2 processing and TSS estimation"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.s2_processor = None
        self.jiang_processor = None

        # Processing statistics
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = time.time()

        # System monitoring
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()

        # Initialize processors based on mode
        self._initialize_processors()

        logger.info(f"Initialized Unified S2-TSS Processor - Mode: {config.processing_mode.value}")

    def _initialize_processors(self):
        """Initialize processors based on processing mode"""
        mode = self.config.processing_mode

        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            self.s2_processor = S2Processor(self.config)
            self.s2_processor.system_monitor = self.system_monitor

        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
            self.jiang_processor = JiangTSSProcessor(self.config.jiang_config)

    def process_batch(self) -> Dict[str, int]:
        """
        Process all products in the input folder based on selected mode

        Returns:
            Processing statistics
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING UNIFIED S2-TSS PROCESSING")
            logger.info("=" * 80)

            # Find and validate products
            products = self._find_products()
            if not products:
                logger.error("No compatible products found")
                return {'processed': 0, 'failed': 1, 'skipped': 0}

            logger.info(f"Found {len(products)} products to process")
            logger.info(f"Processing mode: {self.config.processing_mode.value}")

            # Process each product
            for i, product_path in enumerate(products, 1):
                self._process_single_product(product_path, i, len(products))

            # Final summary
            self._print_final_summary()

            return {
                'processed': self.processed_count,
                'failed': self.failed_count,
                'skipped': self.skipped_count
            }

        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            return {'processed': self.processed_count, 'failed': self.failed_count + 1, 'skipped': self.skipped_count}

    def _find_products(self) -> List[str]:
        """Find products based on processing mode"""
        products = ProductDetector.scan_input_folder(self.config.input_folder)
        mode = self.config.processing_mode

        # Validate products for current mode
        valid, message, product_list = ProductDetector.validate_processing_mode(products, mode)

        if not valid:
            logger.error(f"Product validation failed: {message}")
            return []

        logger.info(message)
        return sorted(product_list)

    def _process_single_product(self, product_path: str, current: int, total: int):
        """Process single product based on mode"""
        processing_start = time.time()

        try:
            product_name = self._extract_product_name(product_path)

            logger.info(f"Processing {current}/{total}: {product_name}")

            # Check if outputs already exist
            if self.config.skip_existing and self._check_outputs_exist(product_name):
                logger.info(f"Outputs exist, skipping: {product_name}")
                self.skipped_count += 1
                return

            # Process based on mode
            results = {}

            if self.config.processing_mode == ProcessingMode.COMPLETE_PIPELINE:
                # Complete pipeline with proper coordination
                logger.debug("S2 Processing (L1C -> C2RCC)")
                s2_results = self.s2_processor.process_single_product(product_path, self.config.output_folder)
                results.update(s2_results)

                # Check if S2 processing succeeded
                if 'error' in s2_results or 's2_error' in s2_results:
                    error_key = 'error' if 'error' in s2_results else 's2_error'
                    logger.error(f"S2 processing failed, skipping TSS: {s2_results[error_key].error_message}")
                    self.failed_count += 1
                    return

                # TSS Processing (if enabled and processor exists)
                if (self.config.jiang_config.enable_jiang_tss and
                    hasattr(self, 'jiang_processor') and
                    self.jiang_processor is not None):

                    logger.debug("TSS Processing (Jiang)")

                    # Get C2RCC output path from S2 results
                    if 's2_processing' in s2_results:
                        c2rcc_output_path = s2_results['s2_processing'].output_path
                    else:
                        # Find the C2RCC output file in Intermediate folder
                        c2rcc_folder = OutputStructure.get_intermediate_folder(
                            self.config.output_folder, OutputStructure.C2RCC_FOLDER
                        )
                        c2rcc_output_path = os.path.join(c2rcc_folder, f"Resampled_{product_name}_Subset_C2RCC.dim")

                    # Process TSS - use main output folder (jiang_processor creates scene subfolder)
                    try:
                        s2_result = s2_results.get('s2_processing')
                        tss_results = self.jiang_processor.process_jiang_tss(
                            c2rcc_output_path, self.config.output_folder, product_name, s2_result
                        )
                        results.update(tss_results)

                        if 'error' in tss_results:
                            logger.error(f"TSS processing failed: {tss_results['error'].error_message}")
                            # Don't mark as completely failed if S2 succeeded
                        else:
                            logger.info("TSS processing completed successfully")

                    except Exception as tss_error:
                        logger.error(f"TSS processing error: {str(tss_error)}")
                        results['tss_error'] = ProcessingResult(False, "", None, str(tss_error))

                elif self.config.jiang_config.enable_jiang_tss:
                    logger.warning("Jiang TSS enabled but processor not initialized")
                else:
                    logger.info("Jiang TSS disabled - using SNAP TSM/CHL products only")

            elif self.config.processing_mode == ProcessingMode.S2_PROCESSING_ONLY:
                # S2 processing only: L1C -> C2RCC (with SNAP TSM/CHL)
                results = self.s2_processor.process_single_product(product_path, self.config.output_folder)

            elif self.config.processing_mode == ProcessingMode.TSS_PROCESSING_ONLY:
                # TSS processing only: C2RCC -> Jiang TSS
                # jiang_processor creates scene-based subfolders
                if hasattr(self, 'jiang_processor') and self.jiang_processor is not None:
                    results = self.jiang_processor.process_jiang_tss(
                        product_path, self.config.output_folder, product_name
                    )
                else:
                    logger.error("Jiang processor not initialized for TSS_PROCESSING_ONLY mode")
                    results = {'error': ProcessingResult(False, "", None, "Jiang processor not initialized")}

            processing_time = time.time() - processing_start

            # Check results
            if any('error' in key for key in results.keys()):
                logger.error(f"Processing failed with errors")
                self.failed_count += 1
            else:
                success_count = 0
                for key, result in results.items():
                    if isinstance(result, ProcessingResult) and hasattr(result, 'success'):
                        if result.success:
                            success_count += 1
                    elif result is not None:
                        success_count += 1
                processing_time = time.time() - processing_start
                logger.info(f"Scene complete: {success_count} products ({processing_time/60:.1f} min)")
                self.processed_count += 1

            # Aggressive Memory cleanup between scenes
            try:
                # Clean up large variables from this processing cycle
                MemoryManager.cleanup_variables(results)

                # Force multiple GC passes (all generations)
                gc.collect(0)  # Generation 0 (youngest)
                gc.collect(1)  # Generation 1
                gc.collect(2)  # Generation 2 (full collection)

                # Enhanced memory monitoring
                if MemoryManager.monitor_memory():
                    logger.debug("Running aggressive memory cleanup...")
                    MemoryManager.cleanup_variables()

                    # Multiple full GC passes to ensure cleanup
                    for _ in range(3):
                        gc.collect()

                    # Check memory again after cleanup
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    logger.debug(f"Memory usage after cleanup: {current_memory:.1f}MB")

            except Exception as cleanup_error:
                logger.debug(f"Memory cleanup warning: {cleanup_error}")

            # Progress estimation and scene separator
            if self.processed_count > 0 and current < total:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / current
                remaining = total - current
                eta_minutes = (avg_time * remaining) / 60
                logger.info(f"Progress: {current}/{total} - ETA: {eta_minutes:.0f} min remaining")
            logger.info("-" * 60)

        except Exception as e:
            processing_time = time.time() - processing_start
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            self.failed_count += 1

    def _extract_product_name(self, product_path: str) -> str:
        """Extract clean product name from path"""
        basename = os.path.basename(product_path)

        if basename.endswith('.dim'):
            product_name = basename.replace('.dim', '')
        elif basename.endswith('.zip'):
            product_name = basename.replace('.zip', '')
        elif basename.endswith('.SAFE'):
            product_name = basename.replace('.SAFE', '')
        else:
            product_name = basename

        # Clean up common prefixes/suffixes
        if product_name.startswith('Resampled_'):
            product_name = product_name.replace('Resampled_', '')
        if '_Subset_C2RCC' in product_name:
            product_name = product_name.replace('_Subset_C2RCC', '')
        if '_C2RCC' in product_name:
            product_name = product_name.replace('_C2RCC', '')

        return product_name

    def _check_outputs_exist(self, product_name: str) -> bool:
        """Check if outputs already exist for this product"""
        try:
            mode = self.config.processing_mode

            # Extract clean scene name for new folder structure
            scene_name = OutputStructure.extract_clean_scene_name(product_name)

            if mode == ProcessingMode.COMPLETE_PIPELINE:
                # Check for C2RCC output in Intermediate folder
                c2rcc_folder = OutputStructure.get_intermediate_folder(
                    self.config.output_folder, OutputStructure.C2RCC_FOLDER
                )
                c2rcc_path = os.path.join(c2rcc_folder, f"Resampled_{product_name}_Subset_C2RCC.dim")
                if not os.path.exists(c2rcc_path):
                    return False

                # Check for TSS output if Jiang is enabled (in scene folder)
                if self.config.jiang_config.enable_jiang_tss:
                    scene_folder = os.path.join(self.config.output_folder, scene_name)
                    tss_path = os.path.join(scene_folder, OutputStructure.TSS_FOLDER, f"{scene_name}_TSS.tif")
                    if not os.path.exists(tss_path):
                        return False

                return True

            elif mode == ProcessingMode.S2_PROCESSING_ONLY:
                # Check for C2RCC output
                c2rcc_folder = OutputStructure.get_intermediate_folder(
                    self.config.output_folder, OutputStructure.C2RCC_FOLDER
                )
                c2rcc_path = os.path.join(c2rcc_folder, f"Resampled_{product_name}_Subset_C2RCC.dim")
                return os.path.exists(c2rcc_path)

            elif mode == ProcessingMode.TSS_PROCESSING_ONLY:
                # Check for TSS output in scene folder
                scene_folder = os.path.join(self.config.output_folder, scene_name)
                tss_path = os.path.join(scene_folder, OutputStructure.TSS_FOLDER, f"{scene_name}_TSS.tif")
                return os.path.exists(tss_path)

            return False

        except Exception as e:
            logger.debug(f"Error checking existing outputs: {e}")
            return False

    def _print_final_summary(self):
        """Print final processing summary"""
        total_time = (time.time() - self.start_time) / 60

        logger.info(f"\n{'=' * 80}")
        logger.info("UNIFIED S2-TSS PROCESSING SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Products processed successfully: {self.processed_count}")
        logger.info(f"Products skipped (existing): {self.skipped_count}")
        logger.info(f"Products with errors: {self.failed_count}")
        logger.info(f"Total processing time: {total_time:.2f} minutes")

        if self.processed_count > 0:
            avg_time = total_time / self.processed_count
            logger.info(f"Average time per product: {avg_time:.2f} minutes")

        # Output summary
        logger.info(f"\nOutput Structure:")
        logger.info(f"|-- {self.config.output_folder}/")
        if self.config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
            logger.info(f"    |-- <scene_name>/")
            logger.info(f"    |   |-- TSS/           (Jiang TSS products)")
            logger.info(f"    |   |-- SNAP/          (SNAP TSM/CHL)")
            logger.info(f"    |   |-- RGB/           (RGB composites)")
            logger.info(f"    |   |-- Indices/       (Spectral indices)")
            logger.info(f"    |   |-- Advanced/      (HAB, WaterClarity)")
        if self.config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            logger.info(f"    |-- Intermediate/")
            logger.info(f"    |   |-- Geometric/     (Resampled L1C)")
            logger.info(f"    |   |-- C2RCC/         (C2RCC products)")
        logger.info(f"    |-- Logs/")

    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status"""
        if self.s2_processor:
            return self.s2_processor.get_processing_status()
        else:
            total = self.processed_count + self.failed_count + self.skipped_count
            elapsed_time = time.time() - self.start_time

            return ProcessingStatus(
                total_products=total,
                processed=self.processed_count,
                failed=self.failed_count,
                skipped=self.skipped_count,
                current_product="",
                current_stage="",
                progress_percent=(total / max(total, 1)) * 100,
                eta_minutes=0.0,
                processing_speed=(self.processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0.0
            )

    def cleanup(self):
        """Cleanup resources"""
        self.system_monitor.stop_monitoring()

        if self.s2_processor:
            self.s2_processor.cleanup()
