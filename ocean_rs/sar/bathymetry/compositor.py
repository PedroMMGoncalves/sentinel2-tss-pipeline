"""
Multi-temporal bathymetry compositing.

Combines multiple BathymetryResult from different SAR acquisitions
into a single robust depth estimate, weighted by confidence.
"""

import logging
import numpy as np
from typing import List

from ..core.data_models import BathymetryResult

logger = logging.getLogger('ocean_rs')


def composite_bathymetry(results: List[BathymetryResult],
                         method: str = "weighted_median") -> BathymetryResult:
    """Combine multiple temporal bathymetry observations.

    Args:
        results: List of BathymetryResult from different acquisitions
        method: "weighted_median" or "weighted_mean"

    Returns:
        Composite BathymetryResult with reduced uncertainty
    """
    if len(results) == 0:
        raise ValueError("No bathymetry results to composite")

    if len(results) == 1:
        logger.info("Single result, no compositing needed")
        return results[0]

    logger.info(f"Compositing {len(results)} bathymetry results using {method}")

    # L-9: Validate all results have same shape before stacking
    ref_shape = results[0].depth.shape
    for i, r in enumerate(results):
        if r.depth.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: result[0] has shape {ref_shape}, "
                f"but result[{i}] has shape {r.depth.shape}. "
                "All results must have identical spatial dimensions for compositing."
            )
        if r.uncertainty.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: result[0].depth has shape {ref_shape}, "
                f"but result[{i}].uncertainty has shape {r.uncertainty.shape}."
            )

    # M2-10: Verify all results share the same CRS
    if len(results) > 1:
        ref_crs = results[0].geo.crs_wkt if results[0].geo else ''
        for i, r in enumerate(results[1:], 1):
            r_crs = r.geo.crs_wkt if r.geo else ''
            if r_crs != ref_crs:
                logger.warning(
                    f"CRS mismatch: result[0] vs result[{i}] — composite may be spatially incoherent"
                )

    all_depths = np.array([r.depth for r in results])
    all_uncertainties = np.array([r.uncertainty for r in results])

    n_obs = len(results)

    # H-6: Minimum uncertainty regularization (0.5m^2) — assumes independent errors
    epsilon = 0.25
    weights = 1.0 / (all_uncertainties**2 + epsilon)

    if method == "weighted_mean":
        weight_sum = np.sum(weights, axis=0)
        depth = np.sum(all_depths * weights, axis=0) / weight_sum
        uncertainty = 1.0 / np.sqrt(weight_sum)
    else:
        depth = _weighted_median(all_depths, weights)
        residuals = np.abs(all_depths - depth[np.newaxis, :])

        # C-4: MAD with N<3 underestimates uncertainty — use half-range instead
        if n_obs < 3:
            logger.warning(
                f"Only N={n_obs} observations: using half-range (max-min)/2 "
                "as uncertainty (MAD requires N>=3)"
            )
            uncertainty = (np.max(all_depths, axis=0) - np.min(all_depths, axis=0)) / 2.0
        else:
            if n_obs < 5:
                logger.warning(
                    f"Only N={n_obs} observations: MAD estimate may be unreliable "
                    "(recommend N>=5 for robust statistics)"
                )
            # M-5: Uses MAD * 1.4826 as robust sigma (assumes Gaussian distribution).
            # Alternative: IQR/1.349 for non-Gaussian residuals.
            uncertainty = 1.4826 * _weighted_median(residuals, weights)

    logger.info(f"Composite depth range: {np.nanmin(depth):.1f} - {np.nanmax(depth):.1f}m "
                f"(mean uncertainty: {np.nanmean(uncertainty):.1f}m)")

    # L-7: Use median instead of mean for composite wave_period (robust to outliers)
    composite_wave_period = float(np.median([r.wave_period for r in results]))

    return BathymetryResult(
        depth=depth,
        uncertainty=uncertainty,
        method=f"composite_{method}",
        wave_period=composite_wave_period,
        wave_period_source="composite",
        geo=results[0].geo,
        metadata={
            'n_observations': n_obs,
            'compositing_method': method,
        },
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted median along first axis.

    Iterates per-point (O(N) per point where N = n_observations).
    For large grids consider vectorizing with np.apply_along_axis,
    though the bottleneck is typically n_observations << n_points.

    Args:
        values: Array of shape (n_obs, n_points)
        weights: Array of shape (n_obs, n_points)

    Returns:
        Weighted median of shape (n_points,)
    """
    n_obs, n_points = values.shape
    result = np.zeros(n_points)

    for j in range(n_points):
        v = np.asarray(values[:, j], dtype=np.float64)
        w = np.asarray(weights[:, j], dtype=np.float64)
        # H2-2: Filter NaN and non-positive weights to prevent cumsum corruption
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            result[j] = np.nan
            continue
        v = v[valid]
        w = w[valid]
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]
        w_sorted = w[sorted_idx]
        cumsum = np.cumsum(w_sorted)
        cutoff = cumsum[-1] / 2.0
        idx = np.searchsorted(cumsum, cutoff)
        result[j] = v_sorted[min(idx, len(v_sorted) - 1)]

    return result
