"""
Bathymetry pipeline orchestrator.

Coordinates the full pipeline: preprocess -> FFT -> wave period -> depth inversion.
Follows the pattern of ocean_rs/optical/core/unified_processor.py.
"""

import gc
import time
import logging
from pathlib import Path
from typing import List, Optional, Callable

from ..core.data_models import OceanImage, BathymetryResult
from ..config.sar_config import SARProcessingConfig
from ..sensors.sentinel1 import Sentinel1Adapter
from ..bathymetry.fft_extractor import extract_swell
from ..bathymetry.wave_period import get_wave_period
from ..bathymetry.depth_inversion import invert_depth
from ..bathymetry.compositor import composite_bathymetry
from ocean_rs.shared import RasterIO

logger = logging.getLogger('ocean_rs')


class BathymetryPipeline:
    """Main orchestrator for SAR bathymetry processing.

    Pipeline: Preprocess -> FFT -> Wave Period -> Depth Inversion -> Export
    """

    def __init__(self, config: SARProcessingConfig):
        self.config = config
        self.adapter = Sentinel1Adapter()

        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None

        self._cancelled = False

    def cancel(self):
        """Request processing cancellation."""
        self._cancelled = True

    def process_scenes(self,
                      scene_paths: List[Path],
                      progress_callback: Optional[Callable] = None
                      ) -> Optional[BathymetryResult]:
        """Process multiple SAR scenes through the bathymetry pipeline.

        Args:
            scene_paths: Paths to downloaded SAR products
            progress_callback: Optional callback(step, total_steps, message)

        Returns:
            Composite BathymetryResult, or None if all failed
        """
        self._cancelled = False
        self.start_time = time.time()
        total = len(scene_paths)
        results = []

        output_dir = Path(self.config.output_directory)
        intermediate_dir = output_dir / "Intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        for i, scene_path in enumerate(scene_paths):
            if self._cancelled:
                logger.info("Processing cancelled by user")
                break

            scene_name = scene_path.stem
            logger.info(f"{'='*60}")
            logger.info(f"Processing [{i+1}/{total}]: {scene_name}")
            logger.info(f"{'='*60}")

            if progress_callback:
                progress_callback(i, total, f"Processing: {scene_name}")

            try:
                result = self._process_single_scene(scene_path, intermediate_dir)
                if result is not None:
                    results.append(result)
                    self.processed_count += 1
                else:
                    self.failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {scene_name}: {e}")
                self.failed_count += 1

            gc.collect()

        if not results:
            logger.warning("No scenes produced valid bathymetry results")
            return None

        if len(results) > 1 and self.config.compositing_config.enabled:
            logger.info(f"Compositing {len(results)} results...")
            final = composite_bathymetry(
                results,
                method=self.config.compositing_config.method
            )
        else:
            final = results[0]

        self._export_results(final, output_dir)

        elapsed = time.time() - self.start_time
        logger.info(f"Pipeline complete: {self.processed_count} processed, "
                   f"{self.failed_count} failed, {elapsed:.0f}s elapsed")

        if progress_callback:
            progress_callback(total, total, "Processing complete")

        return final

    def _process_single_scene(self, scene_path: Path,
                               intermediate_dir: Path) -> Optional[BathymetryResult]:
        """Process a single SAR scene through the pipeline."""
        # Step 1: Preprocess (no TC — for FFT analysis)
        logger.info("Step 1/4: Preprocessing (SNAP GPT, no TC for FFT)...")
        # Extract primary polarization (e.g. "VV" from "VV+VH")
        pol = self.config.search_config.polarization.split('+')[0]
        image = self.adapter.preprocess(
            scene_path,
            intermediate_dir,
            snap_gpt_path=self.config.snap_gpt_path or None,
            polarization=pol,
        )

        # Step 2: FFT swell extraction
        logger.info("Step 2/4: FFT swell extraction...")
        fft_cfg = self.config.fft_config
        swell = extract_swell(
            image,
            tile_size_m=fft_cfg.tile_size_m,
            overlap=fft_cfg.overlap,
            min_wavelength_m=fft_cfg.min_wavelength_m,
            max_wavelength_m=fft_cfg.max_wavelength_m,
            confidence_threshold=fft_cfg.confidence_threshold,
        )

        # Step 3: Wave period
        logger.info("Step 3/4: Wave period retrieval...")
        depth_cfg = self.config.depth_config
        if depth_cfg.wave_period_source == "manual":
            wave_period = depth_cfg.manual_wave_period
            period_source = "manual"
        else:
            try:
                # H-8: Convert center coordinates from image CRS to WGS84 for WW3 API
                lon, lat = self._get_wgs84_center(image)
                acq_time = image.metadata.get('datetime', '')
                wave_period = get_wave_period(lon, lat, acq_time)
                period_source = "wavewatch3"
            except Exception as e:
                logger.warning(f"WaveWatch III failed: {e}. Using manual period.")
                wave_period = depth_cfg.manual_wave_period
                period_source = "manual_fallback"

        # Step 4: Depth inversion
        logger.info("Step 4/4: Depth inversion...")
        result = invert_depth(
            swell,
            wave_period=wave_period,
            max_depth_m=depth_cfg.max_depth_m,
            gravity=depth_cfg.gravity,
            max_iterations=depth_cfg.max_iterations,
            convergence_tol=depth_cfg.convergence_tol,
        )
        result.wave_period_source = period_source

        return result

    @staticmethod
    def _get_wgs84_center(image: OceanImage) -> tuple:
        """Convert image center coordinates from image CRS to WGS84.

        H-8: The WaveWatch III API expects WGS84 (lon/lat) coordinates,
        but the image may be in a projected CRS (e.g. UTM).

        Args:
            image: OceanImage with geo transform and CRS.

        Returns:
            Tuple of (longitude, latitude) in WGS84.

        Raises:
            RuntimeError: If coordinate transformation fails.
        """
        from osgeo import osr

        cx = image.geo.origin_x + (image.geo.cols / 2) * image.geo.pixel_size_x
        cy = image.geo.origin_y + (image.geo.rows / 2) * image.geo.pixel_size_y

        crs_wkt = image.geo.crs_wkt
        if not crs_wkt:
            logger.warning(
                "No CRS defined for image — assuming coordinates are already WGS84"
            )
            return cx, cy

        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(crs_wkt)

        # Check if already geographic (lon/lat)
        if source_srs.IsGeographic():
            return cx, cy

        target_srs = osr.SpatialReference()
        target_srs.SetWellKnownGeogCS("WGS84")

        # Ensure consistent axis order (lon, lat) regardless of GDAL version
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transform = osr.CoordinateTransformation(source_srs, target_srs)
        try:
            lon, lat, _ = transform.TransformPoint(cx, cy)
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform center ({cx}, {cy}) to WGS84: {e}"
            ) from e

        # M2-11: Sanity check — transformed coordinates must be plausible WGS84
        if abs(lon) > 360 or abs(lat) > 90:
            raise ValueError(
                f"Center coordinates ({lon}, {lat}) not plausible WGS84 — "
                f"CRS may be projected. Ensure image has geographic CRS metadata."
            )

        logger.info(f"Image center: UTM ({cx:.0f}, {cy:.0f}) -> WGS84 ({lon:.4f}, {lat:.4f})")
        return lon, lat

    def _export_results(self, result: BathymetryResult, output_dir: Path):
        """Export bathymetry result as GeoTIFF."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # H2-6: Guard against 1D tile-level arrays — write_raster requires 2D
        if result.depth.ndim == 1:
            logger.warning(
                "Depth results are 1D tile-level arrays — gridding to 2D not yet implemented. "
                "Skipping GeoTIFF export. Use compositor results directly."
            )
            return None

        if self.config.export_geotiff and result.geo is not None:
            # M-24: Ensure pixel_size_y is negative (GeoTIFF convention:
            # origin is top-left, y increases downward in pixel space)
            pixel_size_y = result.geo.pixel_size_y
            origin_y = result.geo.origin_y
            if pixel_size_y > 0:
                # Flip to north-origin convention: move origin to north edge
                n_rows = result.depth.shape[0] if result.depth.ndim == 2 else 1
                origin_y = origin_y + pixel_size_y * n_rows
                pixel_size_y = -pixel_size_y
                logger.warning("Corrected pixel_size_y sign and adjusted origin_y to north edge")

            # Build GDAL-compatible metadata dict
            geo_metadata = {
                'geotransform': (
                    result.geo.origin_x,
                    result.geo.pixel_size_x,
                    0,
                    origin_y,
                    0,
                    pixel_size_y,
                ),
                'projection': result.geo.crs_wkt,
            }

            tiff_path = output_dir / "bathymetry_depth.tif"
            try:
                RasterIO.write_raster(
                    result.depth,
                    str(tiff_path),
                    geo_metadata,
                    description="SAR Bathymetry Depth (m)",
                    nodata=-9999.0,
                )
                logger.info(f"Exported bathymetry depth: {tiff_path}")
            except RuntimeError as e:
                raise RuntimeError(f"Failed to write bathymetry depth raster: {e}") from e

            unc_path = output_dir / "bathymetry_uncertainty.tif"
            try:
                RasterIO.write_raster(
                    result.uncertainty,
                    str(unc_path),
                    geo_metadata,
                    description="SAR Bathymetry Uncertainty (m)",
                    nodata=-9999.0,
                )
                logger.info(f"Exported bathymetry uncertainty: {unc_path}")
            except RuntimeError as e:
                raise RuntimeError(f"Failed to write bathymetry uncertainty raster: {e}") from e
