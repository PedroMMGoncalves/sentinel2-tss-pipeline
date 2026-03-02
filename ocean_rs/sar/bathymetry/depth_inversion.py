"""
Depth inversion using the linear wave dispersion relation.

Solves omega^2 = g * k * tanh(k * h) for depth h using Newton-Raphson.

Reference:
    The dispersion relation relates wavelength to depth:
    waves slow down and shorten as they enter shallow water.
"""

import logging
import numpy as np

from ..core.data_models import SwellField, BathymetryResult

logger = logging.getLogger('ocean_rs')


def invert_depth(swell: SwellField,
                 wave_period: float,
                 max_depth_m: float = 100.0,
                 gravity: float = 9.81,
                 max_iterations: int = 50,
                 convergence_tol: float = 1e-6) -> BathymetryResult:
    """Invert depth from swell wavelength using linear dispersion relation.

    Solves: omega^2 = g * k * tanh(k * h)
    Where: omega = 2*pi/T, k = 2*pi/L, h = depth

    Newton-Raphson iteration:
        f(h) = omega^2 - g*k*tanh(k*h) = 0
        f'(h) = -g*k^2 / cosh^2(k*h)
        h_new = h - f(h)/f'(h)

    Warning:
        Single wave period used for entire scene. For large scenes (>50km),
        consider processing in smaller AOIs or using spatially varying wave period.
    """
    # H-20: Validate wave_period is positive (prevents division by zero)
    if wave_period <= 0:
        raise ValueError(f"Wave period must be positive, got {wave_period}")

    wavelengths = swell.wavelength.copy()

    # H-20: Filter non-positive wavelengths
    positive_mask = wavelengths > 0
    if not np.all(positive_mask):
        n_invalid = np.sum(~positive_mask)
        logger.warning(f"Filtering {n_invalid} non-positive wavelength values")
        wavelengths = wavelengths[positive_mask]
        if len(wavelengths) == 0:
            raise ValueError("No positive wavelength values after filtering")

    omega = 2 * np.pi / wave_period
    k = 2 * np.pi / wavelengths

    # H-9: Log warning about single wave period for large scenes
    if swell.geo is not None:
        scene_extent_x = abs(swell.geo.pixel_size_x * swell.geo.cols)
        scene_extent_y = abs(swell.geo.pixel_size_y * swell.geo.rows)
        scene_extent_km = max(scene_extent_x, scene_extent_y) / 1000.0
        if scene_extent_km > 50:
            logger.warning(
                f"Scene extent is {scene_extent_km:.0f}km. Single wave period "
                "T=%.1fs used for entire scene — consider processing in smaller "
                "AOIs or using spatially varying wave period.", wave_period
            )

    logger.info(f"Depth inversion: {len(wavelengths)} points, T={wave_period:.1f}s, "
                f"wavelength range: {wavelengths.min():.0f}-{wavelengths.max():.0f}m")

    L_deep = gravity * wave_period**2 / (2 * np.pi)
    logger.info(f"Deep water wavelength: {L_deep:.0f}m")

    h = wavelengths / 2.0

    # L-5: Initialize iteration variable before loop (prevents NameError if max_iterations=0)
    iteration = 0

    for iteration in range(max_iterations):
        kh = np.clip(k * h, 0, 20)

        tanh_kh = np.tanh(kh)
        cosh_kh = np.cosh(kh)

        f = omega**2 - gravity * k * tanh_kh
        f_prime = -gravity * k**2 / (cosh_kh**2)

        valid = np.abs(f_prime) > 1e-12
        delta = np.zeros_like(h)
        delta[valid] = f[valid] / f_prime[valid]

        h -= delta
        h = np.maximum(h, 0.1)

        max_delta = np.max(np.abs(delta))
        if max_delta < convergence_tol:
            logger.info(f"Converged after {iteration + 1} iterations "
                       f"(max delta: {max_delta:.2e}m)")
            break
    else:
        logger.warning(f"Newton-Raphson did not converge after {max_iterations} "
                      f"iterations (max delta: {max_delta:.2e}m)")

    # M-6: Check for deep water (kh > 10) — waves don't interact with bottom
    kh_final = k * h
    deep_water_mask = kh_final > 10
    n_deep = np.sum(deep_water_mask)
    if n_deep > 0:
        logger.warning(
            f"{n_deep} points in deep water (kh > 10) — waves don't sense bottom. "
            "Setting these depths to NaN."
        )
        h[deep_water_mask] = np.nan

    h = np.clip(h, 0, max_depth_m)

    # H-5: Scale wavelength uncertainty by FFT confidence
    # base_uncertainty = dh/dL * assumed_wavelength_uncertainty
    wavelength_uncertainty = 0.1 * wavelengths
    dh_dL = h / wavelengths
    base_uncertainty = np.abs(dh_dL * wavelength_uncertainty)

    # Scale by confidence: high-confidence tiles get tighter bounds
    confidence = swell.confidence
    if not np.all(positive_mask):
        confidence = confidence[positive_mask]
    uncertainty = base_uncertainty / np.maximum(confidence, 0.1)
    uncertainty = np.clip(uncertainty, 0.5, max_depth_m * 0.5)

    logger.info(f"Depth range: {np.nanmin(h):.1f} - {np.nanmax(h):.1f}m "
                f"(mean uncertainty: {np.nanmean(uncertainty):.1f}m)")

    return BathymetryResult(
        depth=h,
        uncertainty=uncertainty,
        method="linear_dispersion",
        wave_period=wave_period,
        wave_period_source="",
        geo=swell.geo,
        metadata={
            'n_points': len(h),
            'wave_period': wave_period,
            'deep_water_wavelength': L_deep,
            'iterations': min(iteration + 1, max_iterations),
            'n_deep_water_masked': int(n_deep),
        },
    )
