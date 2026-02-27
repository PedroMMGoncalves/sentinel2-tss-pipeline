"""
Main entry point for Sentinel-2 TSS Pipeline.

Provides both CLI and GUI interfaces for processing Sentinel-2 imagery
and estimating Total Suspended Solids (TSS).

Usage:
    GUI mode (default):
        python -m sentinel2_tss_pipeline

    CLI mode:
        python -m sentinel2_tss_pipeline -i <input_folder> -o <output_folder>

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

import os
import sys
import logging
import argparse
import tkinter as tk
from tkinter import messagebox

from .config import (
    ProcessingMode,
    ResamplingConfig,
    SubsetConfig,
    C2RCCConfig,
    TSSConfig,
    ProcessingConfig,
)
from .core import UnifiedS2TSSProcessor
from .utils import setup_enhanced_logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def cli_main():
    """Command line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="Unified S2 Processing & TSS Estimation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sentinel2_tss_pipeline -i D:/L1C_Products -o D:/Results
  python -m sentinel2_tss_pipeline -i /path/to/l1c -o /path/to/results --mode s2_processing_only
  python -m sentinel2_tss_pipeline -i ./c2rcc_products -o ./tss_results --mode tss_processing_only --enable-tss
        """
    )

    parser.add_argument("-i", "--input", required=True,
                       help="Input folder containing L1C products (.zip/.SAFE) or C2RCC products (.dim)")

    parser.add_argument("-o", "--output", required=True,
                       help="Output folder for results")

    parser.add_argument("--mode", choices=["complete_pipeline", "s2_processing_only", "tss_processing_only"],
                       default="complete_pipeline",
                       help="Processing mode (default: complete_pipeline)")

    parser.add_argument("--no-skip", action="store_true",
                       help="Process all products (don't skip existing outputs)")

    parser.add_argument("--enable-tss", action="store_true",
                       help="Enable Jiang TSS methodology (in addition to automatic SNAP TSM/CHL)")

    parser.add_argument("--no-ecmwf", action="store_true",
                       help="Disable ECMWF auxiliary data (reduces accuracy)")

    parser.add_argument("--test", action="store_true",
                       help="Test mode (process only 1 product)")

    parser.add_argument("--memory-limit", type=int, default=8,
                       help="Memory limit in GB (default: 8)")

    parser.add_argument("--threads", type=int, default=4,
                       help="Number of processing threads (default: 4)")

    parser.add_argument("--resolution", choices=["10", "20", "60"], default="10",
                       help="Target resolution in meters (default: 10)")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        print(f"ERROR: Input folder does not exist: {args.input}")
        return False

    # Create output folder
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create output folder: {e}")
        return False

    # Create configuration from arguments
    c2rcc_config = C2RCCConfig()
    c2rcc_config.use_ecmwf_aux_data = not args.no_ecmwf

    tss_config = TSSConfig()
    tss_config.enable_tss_processing = args.enable_tss

    resampling_config = ResamplingConfig()
    resampling_config.target_resolution = args.resolution

    config = ProcessingConfig(
        processing_mode=ProcessingMode(args.mode),
        input_folder=args.input,
        output_folder=args.output,
        resampling_config=resampling_config,
        subset_config=SubsetConfig(),
        c2rcc_config=c2rcc_config,
        tss_config=tss_config,
        skip_existing=not args.no_skip,
        test_mode=args.test,
        memory_limit_gb=args.memory_limit,
        thread_count=args.threads
    )

    # Print configuration
    print("=" * 80)
    print("UNIFIED S2 PROCESSING & TSS ESTIMATION PIPELINE")
    print("=" * 80)
    print(f"Input folder: {config.input_folder}")
    print(f"Output folder: {config.output_folder}")
    print(f"Processing mode: {config.processing_mode.value}")
    print(f"Resolution: {config.resampling_config.target_resolution}m")
    print(f"ECMWF: {'Enabled' if config.c2rcc_config.use_ecmwf_aux_data else 'Disabled'}")
    print(f"TSS Processing: {'Enabled' if config.tss_config.enable_tss_processing else 'Disabled'}")
    print(f"Skip existing: {config.skip_existing}")
    print(f"Test mode: {config.test_mode}")
    print(f"Memory limit: {config.memory_limit_gb} GB")
    print()

    # Run processing
    try:
        processor = UnifiedS2TSSProcessor(config)
        results = processor.process_batch()

        # Print results
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(f"Successfully processed: {results['processed']}")
        print(f"Skipped (existing): {results['skipped']}")
        print(f"Failed: {results['failed']}")

        processor.cleanup()
        return results['failed'] == 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return False
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        logger.error(f"CLI processing failed: {str(e)}")
        return False


def _check_snap_installation():
    """Check and configure SNAP installation."""
    import shutil

    # Set environment variable if not set
    if not os.environ.get('SNAP_HOME'):
        default_snap_paths = [
            # Windows paths
            r"C:\Program Files\esa-snap",
            r"C:\Program Files (x86)\esa-snap",
            r"D:\Program Files\esa-snap",
        ]

        # Add Linux/Mac paths
        if not sys.platform.startswith('win'):
            home = os.path.expanduser('~')
            default_snap_paths.extend([
                '/opt/snap',
                '/usr/local/snap',
                os.path.join(home, 'snap'),
                os.path.join(home, 'esa-snap'),
            ])

        snap_found = False
        for snap_path in default_snap_paths:
            if os.path.exists(snap_path):
                os.environ['SNAP_HOME'] = snap_path
                logger.info(f"Auto-detected SNAP_HOME: {snap_path}")
                snap_found = True
                break

        # Fallback: check if gpt is on PATH
        if not snap_found:
            gpt_name = 'gpt.exe' if sys.platform.startswith('win') else 'gpt'
            gpt_on_path = shutil.which(gpt_name)
            if gpt_on_path:
                snap_home = os.path.dirname(os.path.dirname(os.path.realpath(gpt_on_path)))
                os.environ['SNAP_HOME'] = snap_home
                logger.info(f"Auto-detected SNAP_HOME from PATH: {snap_home}")
                snap_found = True

        if not snap_found:
            return False, "SNAP_HOME not set and SNAP installation not found!"

    # Verify SNAP installation
    snap_home = os.environ.get('SNAP_HOME')
    gpt_path = os.path.join(snap_home, 'bin', 'gpt.exe' if sys.platform.startswith('win') else 'gpt')

    if not os.path.exists(gpt_path):
        return False, f"GPT not found at: {gpt_path}"

    return True, gpt_path


def _check_dependencies():
    """Check for required dependencies."""
    missing_deps = []

    try:
        import numpy as np
        print("NumPy available")
    except ImportError:
        missing_deps.append("numpy")

    try:
        from osgeo import gdal
        print("GDAL available")
    except ImportError:
        missing_deps.append("gdal")

    try:
        import psutil
        print("psutil available")
    except ImportError:
        missing_deps.append("psutil")

    return missing_deps


def main():
    """Main entry point with enhanced error handling."""
    try:
        # Check Python version
        if sys.version_info < (3, 6):
            print("ERROR: Python 3.6 or higher is required!")
            sys.exit(1)

        # Check for required dependencies
        missing_deps = _check_dependencies()

        if missing_deps:
            print(f"\nERROR: Missing required dependencies: {missing_deps}")
            print("Install with:")
            for dep in missing_deps:
                if dep == "gdal":
                    print(f"  conda install {dep}")
                else:
                    print(f"  pip install {dep}")
            sys.exit(1)

        # Check SNAP installation
        snap_ok, snap_info = _check_snap_installation()

        if not snap_ok:
            logger.error(snap_info)
            logger.error("Please install SNAP or set SNAP_HOME environment variable")

            # Show error dialog if running GUI
            if len(sys.argv) == 1:  # No command line arguments = GUI mode
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "SNAP Not Found",
                    f"{snap_info}\n\n"
                    "Please:\n"
                    "1. Install SNAP from https://step.esa.int/\n"
                    "2. Or set SNAP_HOME environment variable\n"
                    "3. Restart this application"
                )
            sys.exit(1)

        # Log startup information
        snap_home = os.environ.get('SNAP_HOME')
        logger.info("=" * 80)
        logger.info("UNIFIED S2 PROCESSING & TSS ESTIMATION PIPELINE v2.0")
        logger.info("=" * 80)
        logger.info(f"SNAP_HOME: {snap_home}")
        logger.info(f"GPT: {snap_info}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Working Directory: {os.getcwd()}")
        cwd_len = len(os.getcwd())
        if cwd_len > 200:
            logger.warning(f"Working directory path is very long ({cwd_len} chars). "
                           "This may cause issues on Windows with MAX_PATH limits.")
        logger.info("=" * 80)

        # Check if command line arguments provided
        if len(sys.argv) > 1:
            # Run CLI interface
            success = cli_main()
            sys.exit(0 if success else 1)
        else:
            # Run GUI interface
            logger.info("Starting GUI application...")
            from .gui import UnifiedS2TSSGUI
            app = UnifiedS2TSSGUI()
            app.run()

        logger.info("Application finished successfully")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Show critical error dialog for GUI mode
        try:
            if len(sys.argv) == 1:  # GUI mode
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "Critical Error",
                    f"A critical error occurred:\n\n{str(e)}\n\n"
                    "Check the log file for details."
                )
        except Exception as dialog_err:
            logger.debug(f"Could not show error dialog: {dialog_err}")

        sys.exit(1)


if __name__ == "__main__":
    main()
