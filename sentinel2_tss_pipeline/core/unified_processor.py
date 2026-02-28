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
import glob
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
from ..utils.logging_utils import StepTracker
from ..processors.tsm_chl_calculator import ProcessingResult
from ..processors.c2rcc_processor import C2RCCProcessor, ProcessingStatus
from ..processors.tss_processor import TSSProcessor

logger = logging.getLogger('sentinel2_tss_pipeline')


class UnifiedS2TSSProcessor:
    """Main processor that coordinates complete S2 processing and TSS estimation"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.c2rcc_processor = None
        self.tss_processor = None

        # Processing statistics
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_products = 0
        self.start_time = None

        # System monitoring
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()

        # Structured logging
        self.tracker = StepTracker(logger)

        # Initialize processors based on mode
        self._initialize_processors()

        logger.debug(f"Initialized Unified S2-TSS Processor - Mode: {config.processing_mode.value}")

    def _initialize_processors(self):
        """Initialize processors based on processing mode"""
        mode = self.config.processing_mode

        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            self.c2rcc_processor = C2RCCProcessor(self.config)
            self.c2rcc_processor.system_monitor = self.system_monitor

        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
            self.tss_processor = TSSProcessor(self.config.tss_config)

    def process_batch(self) -> Dict[str, int]:
        """
        Process all products in the input folder based on selected mode

        Returns:
            Processing statistics
        """
        try:
            # Opening banner
            mode_labels = {
                ProcessingMode.COMPLETE_PIPELINE: "Complete Pipeline (L1C \u2192 C2RCC \u2192 TSS)",
                ProcessingMode.S2_PROCESSING_ONLY: "S2 Processing Only (L1C \u2192 C2RCC)",
                ProcessingMode.TSS_PROCESSING_ONLY: "TSS Processing Only (C2RCC \u2192 TSS)",
            }
            from .. import __version__
            self.tracker.banner(
                f"SENTINEL-2 TSS PIPELINE v{__version__}",
                f"Mode: {mode_labels.get(self.config.processing_mode, self.config.processing_mode.value)}"
            )

            # Step 1: Scan input
            self.tracker.log_step("Step 1: Scanning input folder...")
            products = self._find_products()
            if not products:
                logger.error("  No compatible products found")
                return {'processed': 0, 'failed': 1, 'skipped': 0}

            logger.info(f"  Found: {len(products)} products")

            # Configuration box
            self._log_config_summary()

            # Track batch size and reset start time for accurate progress/ETA
            self.total_products = len(products)
            self.start_time = time.time()

            # Process each product
            for i, product_path in enumerate(products, 1):
                self.tracker.log_step(f"Step {i + 1}: Processing scene {i}/{len(products)}")
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

    def _log_config_summary(self):
        """Log processing configuration as a structured box."""
        c = self.config

        # Build subset description
        if c.subset_config.geometry_wkt:
            wkt = c.subset_config.geometry_wkt
            subset = f"WKT ({wkt[:50]}...)" if len(wkt) > 50 else f"WKT ({wkt})"
        elif c.subset_config.pixel_start_x is not None:
            subset = (f"Pixel ({c.subset_config.pixel_start_x},{c.subset_config.pixel_start_y} "
                      f"w={c.subset_config.pixel_size_x} h={c.subset_config.pixel_size_y})")
        else:
            subset = "Full scene"

        # Build options string
        opts = []
        if c.skip_existing:
            opts.append('skip_existing')
        if c.test_mode:
            opts.append('test_mode')
        if c.delete_intermediate_files:
            opts.append('delete_intermediate')

        kv = {
            "Input": c.input_folder,
            "Output": c.output_folder,
            "Resolution": (f"{c.resampling_config.target_resolution}m "
                           f"(up={c.resampling_config.upsampling_method}, "
                           f"down={c.resampling_config.downsampling_method})"),
            "Subset": subset,
            "C2RCC NN": c.c2rcc_config.net_set,
            "Water": f"salinity={c.c2rcc_config.salinity} PSU, temp={c.c2rcc_config.temperature}C",
            "GPT": f"memory={c.memory_limit_gb}G, threads={c.thread_count}",
        }
        if opts:
            kv["Options"] = ", ".join(opts)

        self.tracker.config_box(kv)

    def _process_single_product(self, product_path: str, current: int, total: int):
        """Process single product based on mode"""
        processing_start = time.time()

        try:
            product_name = self._extract_product_name(product_path)
            clean_product_name = product_name

            # Check if outputs already exist
            if self.config.skip_existing and self._check_outputs_exist(product_name):
                logger.info(f"  Skipped (outputs exist): {product_name}")
                self.skipped_count += 1
                return

            self.tracker.box_start(product_name)

            results = {}

            if self.config.processing_mode == ProcessingMode.COMPLETE_PIPELINE:
                # Determine sub-step count
                has_tss = (self.config.tss_config.enable_tss_processing and
                           hasattr(self, 'tss_processor') and
                           self.tss_processor is not None)
                total_steps = 2 if has_tss else 1

                # Sub-step 1: C2RCC
                with self.tracker.step(f"[1/{total_steps}] Resampling + C2RCC"):
                    s2_results = self.c2rcc_processor.process_single_product(
                        product_path, self.config.output_folder)
                    results.update(s2_results)

                clean_product_name = s2_results.get('clean_product_name', product_name)

                # Check if S2 processing succeeded
                if 'error' in s2_results or 's2_error' in s2_results:
                    error_key = 'error' if 'error' in s2_results else 's2_error'
                    self.tracker.box_line(
                        f"FAILED: {s2_results[error_key].error_message}")
                    self.tracker.box_end()
                    self.failed_count += 1
                    return

                # Sub-step 2: TSS + Visualization
                if has_tss:
                    # Get C2RCC output path
                    if 's2_processing' in s2_results:
                        c2rcc_output_path = s2_results['s2_processing'].output_path
                    else:
                        c2rcc_folder = OutputStructure.get_intermediate_folder(
                            self.config.output_folder, OutputStructure.C2RCC_FOLDER)
                        c2rcc_output_path = os.path.join(
                            c2rcc_folder, f"Resampled_{product_name}_Subset_C2RCC.dim")

                    try:
                        s2_result = s2_results.get('s2_processing')
                        with self.tracker.step(
                                f"[2/{total_steps}] TSS + Visualization (Jiang et al. 2021)"):
                            tss_results = self.tss_processor.process_tss(
                                c2rcc_output_path, self.config.output_folder,
                                product_name, s2_result)
                            results.update(tss_results)

                        if 'error' in tss_results:
                            self.tracker.box_line(
                                f"TSS failed: {tss_results['error'].error_message}")

                    except Exception as tss_error:
                        logger.error(f"TSS processing error: {str(tss_error)}")
                        results['tss_error'] = ProcessingResult(
                            False, "", None, str(tss_error))

                elif self.config.tss_config.enable_tss_processing:
                    logger.warning("TSS processing enabled but processor not initialized")

            elif self.config.processing_mode == ProcessingMode.S2_PROCESSING_ONLY:
                with self.tracker.step("[1/1] Resampling + C2RCC"):
                    results = self.c2rcc_processor.process_single_product(
                        product_path, self.config.output_folder)

            elif self.config.processing_mode == ProcessingMode.TSS_PROCESSING_ONLY:
                if hasattr(self, 'tss_processor') and self.tss_processor is not None:
                    with self.tracker.step(
                            "[1/1] TSS + Visualization (Jiang et al. 2021)"):
                        results = self.tss_processor.process_tss(
                            product_path, self.config.output_folder, product_name)
                else:
                    logger.error("TSS processor not initialized")
                    results = {'error': ProcessingResult(
                        False, "", None, "TSS processor not initialized")}

            processing_time = time.time() - processing_start

            # Check results and print scene summary
            if any('error' in key for key in results.keys()):
                self.tracker.box_line(
                    f"Scene FAILED ({processing_time/60:.1f} min)")
                self.failed_count += 1
            else:
                success_count = sum(
                    1 for r in results.values()
                    if (isinstance(r, ProcessingResult) and
                        hasattr(r, 'success') and r.success)
                    or (not isinstance(r, ProcessingResult) and r is not None))
                self.tracker.box_line(
                    f"Scene complete: {success_count} products "
                    f"({processing_time/60:.1f} min)")

                # Scene-level resource summary
                res = self.tracker.format_resources(
                    self.tracker._scene_cpu_samples,
                    self.tracker._scene_ram_samples)
                if res:
                    self.tracker.box_line(f"Scene resources \u2014 {res}")

                self.processed_count += 1

            self.tracker.box_end()

            # Memory cleanup between scenes
            try:
                MemoryManager.cleanup_variables(results)
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)

                if MemoryManager.monitor_memory():
                    logger.debug("Running aggressive memory cleanup...")
                    MemoryManager.cleanup_variables()
                    for _ in range(3):
                        gc.collect()
                    if HAS_PSUTIL:
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        logger.debug(f"Memory usage after cleanup: {current_memory:.1f}MB")

            except Exception as cleanup_error:
                logger.debug(f"Memory cleanup warning: {cleanup_error}")

            # Delete intermediate files if requested
            if getattr(self.config, 'delete_intermediate_files', False):
                self._cleanup_intermediate_files(self.config.output_folder, clean_product_name)

            # Progress ETA
            if self.processed_count > 0 and current < total:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / current
                remaining = total - current
                eta_minutes = (avg_time * remaining) / 60
                logger.info(
                    f"  Progress: {current}/{total} \u2014 "
                    f"ETA: {eta_minutes:.0f} min remaining")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
                # Check for C2RCC output in Intermediate folder (glob for flexible naming)
                c2rcc_folder = OutputStructure.get_intermediate_folder(
                    self.config.output_folder, OutputStructure.C2RCC_FOLDER
                )
                c2rcc_pattern = os.path.join(c2rcc_folder, f"*{product_name}*C2RCC*.dim")
                if not glob.glob(c2rcc_pattern):
                    return False

                # Check for TSS output if TSS processing is enabled (in scene folder)
                if self.config.tss_config.enable_tss_processing:
                    scene_folder = os.path.join(self.config.output_folder, scene_name)
                    tss_path = os.path.join(scene_folder, OutputStructure.TSS_FOLDER, f"{scene_name}_TSS.tif")
                    if not os.path.exists(tss_path):
                        return False

                return True

            elif mode == ProcessingMode.S2_PROCESSING_ONLY:
                # Check for C2RCC output (glob for flexible naming)
                c2rcc_folder = OutputStructure.get_intermediate_folder(
                    self.config.output_folder, OutputStructure.C2RCC_FOLDER
                )
                c2rcc_pattern = os.path.join(c2rcc_folder, f"*{product_name}*C2RCC*.dim")
                return len(glob.glob(c2rcc_pattern)) > 0

            elif mode == ProcessingMode.TSS_PROCESSING_ONLY:
                # Check for TSS output in scene folder
                scene_folder = os.path.join(self.config.output_folder, scene_name)
                tss_path = os.path.join(scene_folder, OutputStructure.TSS_FOLDER, f"{scene_name}_TSS.tif")
                return os.path.exists(tss_path)

            return False

        except Exception as e:
            logger.debug(f"Error checking existing outputs: {e}")
            return False

    @staticmethod
    def _safe_rmtree(path, retries=3, delay=1.0):
        """Remove directory tree with Windows file-lock retry."""
        import shutil
        for attempt in range(retries):
            try:
                shutil.rmtree(path)
                return True
            except PermissionError:
                if attempt < retries - 1:
                    logger.debug(f"File locked, retrying in {delay}s... ({attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    logger.warning(f"Could not delete {path} after {retries} attempts (file locked)")
                    return False

    def _cleanup_intermediate_files(self, output_folder: str, product_name: str):
        """
        Delete intermediate .dim/.data files after successful processing.

        Removes the Resampled and C2RCC intermediate files to save disk space.
        Only called when delete_intermediate_files config option is True.

        Args:
            output_folder: Base output directory
            product_name: Name of the processed product
        """
        try:
            # Get intermediate folders
            geometric_folder = OutputStructure.get_intermediate_folder(
                output_folder, OutputStructure.GEOMETRIC_FOLDER
            )
            c2rcc_folder = OutputStructure.get_intermediate_folder(
                output_folder, OutputStructure.C2RCC_FOLDER
            )

            deleted_count = 0

            # Delete Resampled product (.dim and .data folder)
            resampled_dim = os.path.join(geometric_folder, f"Resampled_{product_name}_Subset.dim")
            resampled_data = os.path.join(geometric_folder, f"Resampled_{product_name}_Subset.data")

            if os.path.exists(resampled_dim):
                os.remove(resampled_dim)
                deleted_count += 1
            if os.path.exists(resampled_data):
                self._safe_rmtree(resampled_data)
                deleted_count += 1

            # Delete C2RCC product (.dim and .data folder)
            # Try multiple naming patterns
            c2rcc_patterns = [
                f"Resampled_{product_name}_Subset_C2RCC",
                f"Resampled_{product_name}_C2RCC",
                f"{product_name}_C2RCC"
            ]

            for pattern in c2rcc_patterns:
                c2rcc_dim = os.path.join(c2rcc_folder, f"{pattern}.dim")
                c2rcc_data = os.path.join(c2rcc_folder, f"{pattern}.data")

                if os.path.exists(c2rcc_dim):
                    os.remove(c2rcc_dim)
                    deleted_count += 1
                if os.path.exists(c2rcc_data):
                    self._safe_rmtree(c2rcc_data)
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"  Cleanup: Deleted {deleted_count} intermediate files/folders")

        except Exception as e:
            logger.warning(f"  Cleanup warning: Could not delete intermediate files: {e}")

    def _print_final_summary(self):
        """Print final processing summary as a structured banner."""
        total_time = (time.time() - self.start_time) / 60

        kv = {}
        kv["Processed"] = (f"{self.processed_count} "
                           f"scene{'s' if self.processed_count != 1 else ''}")
        if self.skipped_count > 0:
            kv["Skipped"] = (f"{self.skipped_count} "
                             f"scene{'s' if self.skipped_count != 1 else ''}")
        if self.failed_count > 0:
            kv["Failed"] = str(self.failed_count)

        if self.processed_count > 0:
            avg = total_time / self.processed_count
            kv["Total time"] = f"{total_time:.1f} min (avg {avg:.1f} min/scene)"
        else:
            kv["Total time"] = f"{total_time:.1f} min"

        # Batch-level resource summary
        res = self.tracker.format_resources(
            self.tracker._batch_cpu_samples,
            self.tracker._batch_ram_samples)
        if res:
            kv["Resources"] = res

        kv["Output"] = self.config.output_folder

        # Banner title
        if self.failed_count > 0 and self.processed_count == 0:
            title = "FAILED"
        elif self.failed_count > 0:
            title = "COMPLETE (with errors)"
        else:
            title = "COMPLETE"

        self.tracker.summary_banner(kv, title=title)

    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status"""
        if self.c2rcc_processor:
            return self.c2rcc_processor.get_processing_status()
        else:
            completed = self.processed_count + self.failed_count + self.skipped_count
            total = max(self.total_products, completed)
            elapsed_time = time.time() - self.start_time

            return ProcessingStatus(
                total_products=total,
                processed=self.processed_count,
                failed=self.failed_count,
                skipped=self.skipped_count,
                current_product="",
                current_stage="",
                progress_percent=(completed / max(total, 1)) * 100,
                eta_minutes=0.0,
                processing_speed=(self.processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0.0
            )

    def cleanup(self):
        """Cleanup resources"""
        self.system_monitor.stop_monitoring()

        if self.c2rcc_processor:
            self.c2rcc_processor.cleanup()
