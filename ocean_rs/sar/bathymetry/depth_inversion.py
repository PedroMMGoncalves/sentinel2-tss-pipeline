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
    """
    wavelengths = swell.wavelength
    omega = 2 * np.pi / wave_period
    k = 2 * np.pi / wavelengths

    logger.info(f"Depth inversion: {len(wavelengths)} points, T={wave_period:.1f}s, "
                f"wavelength range: {wavelengths.min():.0f}-{wavelengths.max():.0f}m")

    L_deep = gravity * wave_period**2 / (2 * np.pi)
    logger.info(f"Deep water wavelength: {L_deep:.0f}m")

    h = wavelengths / 2.0

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

    h = np.clip(h, 0, max_depth_m)

    wavelength_uncertainty = 0.1 * wavelengths
    dh_dL = h / wavelengths
    depth_uncertainty = np.abs(dh_dL * wavelength_uncertainty)
    depth_uncertainty = np.clip(depth_uncertainty, 0.5, max_depth_m * 0.5)

    logger.info(f"Depth range: {h.min():.1f} - {h.max():.1f}m "
               f"(mean uncertainty: {depth_uncertainty.mean():.1f}m)")

    return BathymetryResult(
        depth=h,
        uncertainty=depth_uncertainty,
        method="linear_dispersion",
        wave_period=wave_period,
        wave_period_source="",
        geo=swell.geo,
        metadata={
            'n_points': len(h),
            'wave_period': wave_period,
            'deep_water_wavelength': L_deep,
            'iterations': min(iteration + 1, max_iterations),
        },
    )
