"""
Displacement analysis pipeline orchestrator.

Routes DInSAR (single-pair) and SBAS (time-series) workflows,
managing the complete chain from interferograms to displacement maps.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..config.sar_config import DisplacementConfig, SARProcessingConfig
from ..core.data_models import Interferogram, DisplacementField
from .dinsar import compute_dinsar
from .sbas import build_network, compute_sbas

logger = logging.getLogger('ocean_rs')


class DisplacementPipeline:
    """End-to-end displacement analysis pipeline.

    Supports:
        - DInSAR: Single interferogram → LOS/vertical displacement
        - SBAS: Network of interferograms → time-series displacement
    """

    def __init__(self, config: SARProcessingConfig):
        self.config = config
        self.disp_config = config.displacement_config or DisplacementConfig()
        self._cancelled = False

    def cancel(self):
        """Request pipeline cancellation."""
        self._cancelled = True
        logger.info("Displacement pipeline cancellation requested")

    def process_dinsar(
        self,
        interferogram: Interferogram,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[DisplacementField]:
        """Run DInSAR displacement estimation.

        Args:
            interferogram: Unwrapped interferogram (topo-corrected).
            output_dir: Output directory for results.
            progress_callback: Optional callback(step, total, message).

        Returns:
            DisplacementField or None if cancelled.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting DInSAR displacement analysis")

        if self._cancelled:
            return None

        # Derive nlooks from InSAR config if available
        insar_config = self.config.insar_config
        nlooks = 1
        if insar_config:
            nlooks = insar_config.coherence_window_range * insar_config.coherence_window_azimuth

        displacement = compute_dinsar(
            interferogram,
            output_vertical=self.disp_config.output_quasi_vertical,
            nlooks=nlooks,
        )

        # Export results
        self._export_displacement(displacement, output_dir, 'DInSAR')

        logger.info("DInSAR displacement analysis complete")
        return displacement

    def process_sbas(
        self,
        interferograms: List[Interferogram],
        pair_indices: list,
        dates: List[str],
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[List[DisplacementField]]:
        """Run SBAS time-series displacement estimation.

        Args:
            interferograms: List of unwrapped interferograms.
            pair_indices: List of (primary_idx, secondary_idx) pairs.
            dates: Acquisition dates.
            output_dir: Output directory for results.
            progress_callback: Optional callback(step, total, message).

        Returns:
            List of DisplacementField (one per date) or None if cancelled.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting SBAS time-series analysis")

        if self._cancelled:
            return None

        # Determine reference pixel by converting (lon, lat) to (row, col)
        reference_pixel = None
        if self.disp_config.reference_point is not None:
            lon, lat = self.disp_config.reference_point
            first_ifg = interferograms[0]
            # Convert geographic (lon, lat) to pixel (col, row) using geo transform
            if first_ifg.geo and first_ifg.geo.pixel_size_x != 0:
                col = int((lon - first_ifg.geo.origin_x) / first_ifg.geo.pixel_size_x)
                row = int((lat - first_ifg.geo.origin_y) / first_ifg.geo.pixel_size_y)
                # Validate bounds
                if 0 <= row < first_ifg.phase.shape[0] and 0 <= col < first_ifg.phase.shape[1]:
                    reference_pixel = (row, col)
                    logger.info(f"Reference point ({lon}, {lat}) mapped to pixel ({row}, {col})")
                else:
                    logger.warning(
                        f"Reference point ({lon}, {lat}) maps to pixel ({row}, {col}) "
                        f"which is outside the image bounds. Using automatic reference pixel."
                    )
                    reference_pixel = None
            else:
                logger.warning(
                    "Cannot convert reference point from geographic to pixel coordinates: "
                    "no valid GeoTransform. Using automatic reference pixel."
                )
                reference_pixel = None

        results = compute_sbas(
            interferograms,
            pair_indices,
            dates,
            temporal_coherence_threshold=self.disp_config.temporal_coherence_threshold,
            reference_pixel=reference_pixel,
            atmospheric_filter=self.disp_config.atmospheric_filter,
        )

        if self._cancelled:
            return None

        # Export results
        for i, disp in enumerate(results):
            self._export_displacement(
                disp, output_dir, f'SBAS_{i:03d}',
                date_label=disp.measurement_date
            )

        logger.info(f"SBAS analysis complete: {len(results)} displacement maps")
        return results

    def _export_displacement(
        self,
        displacement: DisplacementField,
        output_dir: Path,
        prefix: str,
        date_label: str = "",
    ):
        """Export displacement products to GeoTIFF."""
        from ocean_rs.shared.raster_io import RasterIO

        disp_dir = output_dir / 'Displacement'
        disp_dir.mkdir(exist_ok=True)

        metadata = {
            'geotransform': [
                displacement.geo.origin_x, displacement.geo.pixel_size_x, 0,
                displacement.geo.origin_y, 0, displacement.geo.pixel_size_y,
            ] if displacement.geo else [0, 1, 0, 0, 0, -1],
            'projection': displacement.geo.crs_wkt if displacement.geo else '',
        }

        # Displacement map
        suffix = f"_{date_label}" if date_label else ""
        disp_path = str(disp_dir / f'{prefix}_displacement{suffix}.tif')
        RasterIO.write_raster(
            displacement.displacement_m,
            disp_path,
            metadata,
            nodata=float('nan'),
        )
        logger.info(f"Exported: {disp_path}")

        # Uncertainty map
        unc_path = str(disp_dir / f'{prefix}_uncertainty{suffix}.tif')
        RasterIO.write_raster(
            displacement.uncertainty_m,
            unc_path,
            metadata,
            nodata=float('nan'),
        )

        # Metadata text
        meta_path = disp_dir / f'{prefix}_metadata{suffix}.txt'
        with open(meta_path, 'w') as f:
            f.write(f"Component: {displacement.component}\n")
            f.write(f"Reference date: {displacement.reference_date}\n")
            f.write(f"Measurement date: {displacement.measurement_date}\n")
            valid = np.isfinite(displacement.displacement_m)
            if np.any(valid):
                f.write(
                    f"Displacement range: "
                    f"[{np.nanmin(displacement.displacement_m)*1000:.2f}, "
                    f"{np.nanmax(displacement.displacement_m)*1000:.2f}] mm\n"
                )
            for key, value in displacement.metadata.items():
                f.write(f"{key}: {value}\n")
