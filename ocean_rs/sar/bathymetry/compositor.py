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

    all_depths = np.array([r.depth for r in results])
    all_uncertainties = np.array([r.uncertainty for r in results])

    weights = 1.0 / (all_uncertainties**2 + 1e-10)

    if method == "weighted_mean":
        weight_sum = np.sum(weights, axis=0)
        depth = np.sum(all_depths * weights, axis=0) / weight_sum
        uncertainty = 1.0 / np.sqrt(weight_sum)
    else:
        depth = _weighted_median(all_depths, weights)
        residuals = np.abs(all_depths - depth[np.newaxis, :])
        uncertainty = 1.4826 * _weighted_median(residuals, weights)

    logger.info(f"Composite depth range: {depth.min():.1f} - {depth.max():.1f}m "
               f"(mean uncertainty: {uncertainty.mean():.1f}m)")

    return BathymetryResult(
        depth=depth,
        uncertainty=uncertainty,
        method=f"composite_{method}",
        wave_period=np.mean([r.wave_period for r in results]),
        wave_period_source="composite",
        geo=results[0].geo,
        metadata={
            'n_observations': len(results),
            'compositing_method': method,
        },
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted median along first axis.

    Args:
        values: Array of shape (n_obs, n_points)
        weights: Array of shape (n_obs, n_points)

    Returns:
        Weighted median of shape (n_points,)
    """
    n_obs, n_points = values.shape
    result = np.zeros(n_points)

    for j in range(n_points):
        v = values[:, j]
        w = weights[:, j]
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]
        w_sorted = w[sorted_idx]
        cumsum = np.cumsum(w_sorted)
        cutoff = cumsum[-1] / 2.0
        idx = np.searchsorted(cumsum, cutoff)
        result[j] = v_sorted[min(idx, len(v_sorted) - 1)]

    return result
