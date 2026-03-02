"""
Main entry point for OceanRS SAR — SAR Bathymetry Toolkit.

Usage:
    GUI mode (default):
        python -m ocean_rs.sar

    CLI mode:
        python -m ocean_rs.sar --aoi "POLYGON(...)" --start 2024-01-01 --end 2024-06-01 -o results/
"""

import os
import sys
import logging
import argparse
import tkinter as tk
from tkinter import messagebox

from ocean_rs.shared import setup_enhanced_logging

logger = logging.getLogger('ocean_rs')


def _check_dependencies():
    """Check for required dependencies."""
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        from osgeo import gdal
    except ImportError:
        missing.append('gdal')
    return missing


def cli_main():
    """CLI interface for batch bathymetry processing."""
    parser = argparse.ArgumentParser(
        description="OceanRS SAR — SAR Bathymetry Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ocean_rs.sar --aoi "POLYGON((-9.5 38.5, -9.0 38.5, -9.0 39.0, -9.5 39.0, -9.5 38.5))" --start 2024-01-01 --end 2024-06-01 -o results/
  python -m ocean_rs.sar --help
        """
    )
    parser.add_argument("--aoi", required=True, help="AOI as WKT polygon")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--platform", default="Sentinel-1", help="SAR platform")
    parser.add_argument("--beam-mode", default="IW", help="Beam mode")
    parser.add_argument("--wave-period", type=float, help="Manual wave period (seconds)")
    parser.add_argument("--max-depth", type=float, default=100.0, help="Max depth (m)")

    args = parser.parse_args()

    from .config import SARProcessingConfig
    from .download import search_scenes, BatchDownloader, CredentialManager
    from .core import BathymetryPipeline

    config = SARProcessingConfig()
    config.search_config.aoi_wkt = args.aoi
    config.search_config.start_date = args.start
    config.search_config.end_date = args.end
    config.search_config.platform = args.platform
    config.search_config.beam_mode = args.beam_mode
    config.output_directory = args.output
    config.depth_config.max_depth_m = args.max_depth

    if args.wave_period:
        config.depth_config.wave_period_source = "manual"
        config.depth_config.manual_wave_period = args.wave_period

    os.makedirs(args.output, exist_ok=True)

    print(f"Searching for {args.platform} scenes...")
    scenes = search_scenes(
        aoi_wkt=args.aoi,
        start_date=args.start,
        end_date=args.end,
        platform=args.platform,
        beam_mode=args.beam_mode,
    )
    print(f"Found {len(scenes)} scenes")

    if not scenes:
        print("No scenes found. Adjust search parameters.")
        return False

    creds = CredentialManager()
    downloader = BatchDownloader(creds)
    download_dir = os.path.join(args.output, "Downloads")
    paths = downloader.download_scenes(scenes, download_dir)

    if not paths:
        print("No scenes downloaded successfully.")
        return False

    pipeline = BathymetryPipeline(config)
    result = pipeline.process_scenes(paths)

    return result is not None


def main():
    """Main entry point."""
    try:
        missing = _check_dependencies()
        if missing:
            print(f"Missing dependencies: {missing}")
            print("Install with: conda install " + " ".join(missing))
            sys.exit(1)

        setup_enhanced_logging()

        if len(sys.argv) > 1:
            success = cli_main()
            sys.exit(0 if success else 1)
        else:
            logger.info("Starting SAR Bathymetry Toolkit GUI...")
            from .gui import UnifiedSARGUI
            app = UnifiedSARGUI()
            app.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            if len(sys.argv) == 1:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Critical Error", f"{e}\n\nCheck log for details.")
        except Exception:
            print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
