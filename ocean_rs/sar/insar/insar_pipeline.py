"""
End-to-end InSAR pipeline orchestrator.

Processes a pair of SAR SLC images through the complete InSAR chain:
    1. Sensor detection and SLC reading (with TOPS deburst if needed)
    2. Baseline computation from orbit state vectors
    3. SLC co-registration (orbit coarse + ESD/coherence fine)
    4. Interferogram formation with coherence estimation
    5. Topographic phase removal using DEM
    6. Goldstein adaptive phase filtering (Goldstein & Werner, 1998)
    7. Phase unwrapping (SNAPHU default)
    8. Geocoding to geographic coordinates
    9. Export GeoTIFF products

References:
    - Goldstein, R. M. & Werner, C. L. (1998). "Radar interferogram
      filtering for geophysical applications." Geophysical Research
      Letters, 25(21), 4035-4038.
    - Massonnet, D. & Feigl, K. L. (1998). "Radar interferometry and
      its application to changes in the Earth's surface." Reviews of
      Geophysics, 36(4), 441-500.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config.sar_config import InSARConfig, SARProcessingConfig
from ..core.data_models import (
    Interferogram, SLCImage, InSARPair, GeoTransform,
)
from ..sensors.base import SensorAdapter
from ..sensors.sentinel1 import Sentinel1Adapter
from ..sensors.nisar import NISARAdapter
from ..sensors.alos2 import ALOS2Adapter
from .baseline import compute_baseline
from .coregistration import coregister
from .interferogram import form_interferogram
from .phase_filter import goldstein_filter
from .phase_unwrap import unwrap_phase
from .topo_removal import remove_topographic_phase
from .geocoding import geocode, export_geocoded

logger = logging.getLogger('ocean_rs')


class InSARPipeline:
    """End-to-end InSAR processing pipeline.

    Orchestrates the complete interferometric processing chain from
    raw SLC pairs to geocoded interferograms and coherence maps.
    """

    def __init__(self, config: SARProcessingConfig):
        self.config = config
        self.insar_config = config.insar_config or InSARConfig()
        self._cancelled = False

        # Initialize sensor adapters
        self._adapters = [
            Sentinel1Adapter(),
            NISARAdapter(),
            ALOS2Adapter(),
        ]

    def cancel(self):
        """Request pipeline cancellation."""
        self._cancelled = True
        logger.info("InSAR pipeline cancellation requested")

    def process(
        self,
        primary_path: str,
        secondary_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[Interferogram]:
        """Run the complete InSAR pipeline.

        Args:
            primary_path: Path to primary (reference) SLC product.
            secondary_path: Path to secondary (repeat) SLC product.
            output_dir: Output directory for results.
            progress_callback: Optional callback(step, total, message).

        Returns:
            Interferogram with all products, or None if cancelled.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir = output_dir / 'Intermediate'
        intermediate_dir.mkdir(exist_ok=True)

        total_steps = 9
        step = 0

        def _progress(msg):
            nonlocal step
            step += 1
            logger.info(f"[Step {step}/{total_steps}] {msg}")
            if progress_callback:
                progress_callback(step, total_steps, msg)

        try:
            # Step 1: Detect sensor and read SLCs
            _progress("Reading SLC pair")
            if self._cancelled:
                return None

            adapter = self._detect_sensor(primary_path)
            primary_slc = adapter.read_slc(
                Path(primary_path), intermediate_dir
            )
            secondary_slc = adapter.read_slc(
                Path(secondary_path), intermediate_dir
            )

            # Step 2: Compute baseline
            _progress("Computing baseline")
            if self._cancelled:
                return None

            primary_orbits = primary_slc.metadata.get('orbit_state_vectors', [])
            secondary_orbits = secondary_slc.metadata.get('orbit_state_vectors', [])
            primary_time = primary_slc.metadata.get('acquisition_time', '')

            b_perp, b_par = compute_baseline(
                primary_orbits, secondary_orbits, primary_time
            )

            # Compute temporal baseline
            temporal_days = self._compute_temporal_baseline(
                primary_slc, secondary_slc
            )

            # Step 3: Co-registration
            _progress("Co-registering SLC pair")
            if self._cancelled:
                return None

            secondary_coreg = coregister(
                primary_slc, secondary_slc,
                method=self.insar_config.coregistration_method,
                patch_size=self.insar_config.coregistration_patch_size,
                oversample=self.insar_config.coregistration_oversample,
            )

            pair = InSARPair(
                primary=primary_slc,
                secondary=secondary_coreg,
                temporal_baseline_days=temporal_days,
                perpendicular_baseline_m=b_perp,
            )

            # Step 4: Interferogram formation
            _progress("Forming interferogram")
            if self._cancelled:
                return None

            ifg = form_interferogram(
                pair,
                coh_window_range=self.insar_config.coherence_window_range,
                coh_window_azimuth=self.insar_config.coherence_window_azimuth,
            )

            # Step 5: Topographic phase removal
            if self.insar_config.remove_topography:
                _progress("Removing topographic phase")
                if self._cancelled:
                    return None

                ifg = remove_topographic_phase(
                    ifg,
                    primary_orbits,
                    secondary_orbits,
                    dem_path=self.insar_config.dem_path or "auto",
                )
            else:
                step += 1

            # Step 6: Phase filtering
            _progress("Filtering phase (Goldstein)")
            if self._cancelled:
                return None

            ifg = goldstein_filter(
                ifg,
                alpha=self.insar_config.phase_filter_alpha,
                patch_size=self.insar_config.phase_filter_patch_size,
            )

            # Step 7: Phase unwrapping
            _progress("Unwrapping phase")
            if self._cancelled:
                return None

            unwrapped = unwrap_phase(
                ifg,
                method=self.insar_config.unwrapping_method,
            )
            ifg = Interferogram(
                phase=ifg.phase,
                coherence=ifg.coherence,
                unwrapped_phase=unwrapped,
                geo=ifg.geo,
                wavelength_m=ifg.wavelength_m,
                temporal_baseline_days=ifg.temporal_baseline_days,
                perpendicular_baseline_m=ifg.perpendicular_baseline_m,
                incidence_angle=ifg.incidence_angle,
                metadata=ifg.metadata,
            )

            # Step 8: Geocoding
            _progress("Geocoding to geographic coordinates")
            if self._cancelled:
                return None

            if ifg.geo:
                geo_metadata = {
                    'geotransform': [
                        ifg.geo.origin_x, ifg.geo.pixel_size_x, 0,
                        ifg.geo.origin_y, 0, ifg.geo.pixel_size_y,
                    ],
                    'projection': ifg.geo.crs_wkt if ifg.geo.crs_wkt else '',
                }

                geocoded_dir = output_dir / 'InSAR' / 'Geocoded'
                geocoded_dir.mkdir(parents=True, exist_ok=True)

                # Geocode coherence
                if self.insar_config.output_coherence:
                    coh_geo, coh_geotransform = geocode(ifg.coherence, ifg.geo)
                    export_geocoded(
                        coh_geo, coh_geotransform,
                        str(geocoded_dir / 'coherence_geo.tif'),
                    )

                # Geocode unwrapped phase
                if self.insar_config.output_unwrapped and ifg.unwrapped_phase is not None:
                    unw_geo, unw_geotransform = geocode(ifg.unwrapped_phase, ifg.geo)
                    export_geocoded(
                        unw_geo, unw_geotransform,
                        str(geocoded_dir / 'unwrapped_phase_geo.tif'),
                    )

                # Geocode interferogram phase
                if self.insar_config.output_interferogram:
                    ifg_geo, ifg_geotransform = geocode(ifg.phase, ifg.geo)
                    export_geocoded(
                        ifg_geo, ifg_geotransform,
                        str(geocoded_dir / 'interferogram_phase_geo.tif'),
                    )
            else:
                logger.warning("No geotransform available, skipping geocoding")

            # Step 9: Export results
            _progress("Exporting results")
            self._export_results(ifg, output_dir)

            logger.info("InSAR pipeline completed successfully")
            return ifg

        except Exception as e:
            logger.error(f"InSAR pipeline failed: {e}", exc_info=True)
            raise

    def _detect_sensor(self, path: str) -> SensorAdapter:
        """Auto-detect sensor from input file."""
        input_path = Path(path)
        for adapter in self._adapters:
            if adapter.can_process(input_path):
                logger.info(f"Detected sensor: {adapter.sensor_name}")
                return adapter

        raise ValueError(
            f"Cannot identify sensor for: {input_path.name}. "
            f"Supported: Sentinel-1, NISAR, ALOS-2"
        )

    def _compute_temporal_baseline(
        self, primary: SLCImage, secondary: SLCImage
    ) -> float:
        """Compute temporal baseline in days between two acquisitions."""
        from datetime import datetime

        primary_time = primary.metadata.get('acquisition_time', '')
        secondary_time = secondary.metadata.get('acquisition_time', '')

        if not primary_time or not secondary_time:
            logger.warning("Acquisition times not available. Temporal baseline unknown.")
            return 0.0

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

        dt_primary = None
        dt_secondary = None

        for fmt in formats:
            try:
                dt_primary = datetime.strptime(primary_time, fmt)
                break
            except ValueError:
                continue

        for fmt in formats:
            try:
                dt_secondary = datetime.strptime(secondary_time, fmt)
                break
            except ValueError:
                continue

        if dt_primary and dt_secondary:
            delta = abs((dt_secondary - dt_primary).total_seconds())
            return delta / 86400.0

        return 0.0

    def _export_results(self, ifg: Interferogram, output_dir: Path):
        """Export InSAR products to GeoTIFF."""
        from ocean_rs.shared.raster_io import RasterIO

        metadata = {
            'geotransform': [
                ifg.geo.origin_x, ifg.geo.pixel_size_x, 0,
                ifg.geo.origin_y, 0, ifg.geo.pixel_size_y,
            ] if ifg.geo else [0, 1, 0, 0, 0, -1],
            'projection': ifg.geo.crs_wkt if ifg.geo else '',
        }

        insar_dir = output_dir / 'InSAR'
        insar_dir.mkdir(exist_ok=True)

        # Export coherence
        if self.insar_config.output_coherence:
            coh_path = str(insar_dir / 'coherence.tif')
            RasterIO.write_raster(ifg.coherence, coh_path, metadata)
            logger.info(f"Exported: {coh_path}")

        # Export wrapped interferogram
        if self.insar_config.output_interferogram:
            ifg_path = str(insar_dir / 'interferogram_phase.tif')
            RasterIO.write_raster(ifg.phase, ifg_path, metadata)
            logger.info(f"Exported: {ifg_path}")

        # Export unwrapped phase
        if self.insar_config.output_unwrapped and ifg.unwrapped_phase is not None:
            unw_path = str(insar_dir / 'unwrapped_phase.tif')
            RasterIO.write_raster(ifg.unwrapped_phase, unw_path, metadata)
            logger.info(f"Exported: {unw_path}")

        # Export metadata
        meta_path = insar_dir / 'insar_metadata.txt'
        with open(meta_path, 'w') as f:
            f.write(f"Wavelength (m): {ifg.wavelength_m}\n")
            f.write(f"Temporal baseline (days): {ifg.temporal_baseline_days}\n")
            f.write(f"Perpendicular baseline (m): {ifg.perpendicular_baseline_m}\n")
            f.write(f"Mean coherence: {np.nanmean(ifg.coherence):.4f}\n")
            if ifg.unwrapped_phase is not None:
                f.write(f"Unwrapped phase range: "
                        f"[{ifg.unwrapped_phase.min():.2f}, "
                        f"{ifg.unwrapped_phase.max():.2f}] rad\n")
            for key, value in ifg.metadata.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Exported: {meta_path}")
