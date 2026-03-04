"""
Differential InSAR (DInSAR) displacement estimation.

Converts unwrapped interferometric phase to line-of-sight (LOS) displacement
and optionally decomposes to quasi-vertical displacement.

Displacement formula:
    d_LOS = -(λ / 4π) · φ_unwrapped

Sign convention (for ifg = primary * conj(secondary)):
    Positive LOS = increased sensor-to-target distance
    (subsidence / motion away from sensor)

References:
    Massonnet, D. & Feigl, K. (1998). Radar interferometry and its
    application to changes in the Earth's surface. Reviews of Geophysics,
    36(4), 441-500.
"""

import logging

import numpy as np

from ..core.data_models import Interferogram, DisplacementField

logger = logging.getLogger('ocean_rs')


def compute_dinsar(
    interferogram: Interferogram,
    output_vertical: bool = True,
    nlooks: int = 1,
) -> DisplacementField:
    """Compute DInSAR displacement from unwrapped interferogram.

    Requires:
        - Unwrapped phase (after topographic phase removal)
        - Radar wavelength
        - Incidence angle (for vertical decomposition)

    Args:
        interferogram: Interferogram with unwrapped_phase and wavelength.
        output_vertical: If True, also compute quasi-vertical displacement.
        nlooks: Number of independent looks used in coherence estimation.
            If not provided, extracted from interferogram metadata
            (coherence_window_range * coherence_window_azimuth).

    Returns:
        DisplacementField with LOS or quasi-vertical displacement.

    Raises:
        ValueError: If interferogram has no unwrapped phase.
    """
    if interferogram.unwrapped_phase is None:
        raise ValueError(
            "Interferogram must have unwrapped phase for DInSAR. "
            "Run phase unwrapping first."
        )

    wavelength = interferogram.wavelength_m
    if wavelength <= 0:
        raise ValueError("Interferogram wavelength must be positive")

    unwrapped = interferogram.unwrapped_phase
    rows, cols = unwrapped.shape

    logger.info(f"Computing DInSAR displacement: {rows}×{cols} pixels")

    # LOS displacement (m)
    # d_LOS = -(λ / 4π) · φ
    d_los = -(wavelength / (4 * np.pi)) * unwrapped
    d_los = d_los.astype(np.float32)

    # Uncertainty estimation
    # Based on coherence: lower coherence → higher uncertainty
    # σ_phase = √((1 - γ²) / (2·N·γ²)) for N looks (Touzi et al., 1999)
    # σ_d = (λ/4π) · σ_phase
    coherence = interferogram.coherence

    # Determine effective number of looks from metadata if not explicitly provided
    if nlooks <= 1:
        nlooks = (
            interferogram.metadata.get('nlooks', 0)
            or interferogram.metadata.get('coherence_window_range', 1)
            * interferogram.metadata.get('coherence_window_azimuth', 1)
        )
        if nlooks < 1:
            nlooks = 1

    coherence_safe = np.where(coherence > 0.1, coherence, 0.1)
    phase_std = np.sqrt((1 - coherence_safe ** 2) / (2 * nlooks * coherence_safe ** 2))
    uncertainty_los = (wavelength / (4 * np.pi)) * phase_std
    uncertainty_los = uncertainty_los.astype(np.float32)

    logger.info(
        f"LOS displacement range: [{np.nanmin(d_los)*1000:.1f}, "
        f"{np.nanmax(d_los)*1000:.1f}] mm"
    )

    # Quasi-vertical decomposition
    if output_vertical and interferogram.incidence_angle is not None:
        incidence = interferogram.incidence_angle
        cos_theta = np.cos(incidence)
        cos_theta = np.where(np.abs(cos_theta) > 0.01, cos_theta, 0.01)

        d_vertical = d_los / cos_theta
        uncertainty_vertical = uncertainty_los / np.abs(cos_theta)

        logger.info(
            f"Quasi-vertical displacement range: [{np.nanmin(d_vertical)*1000:.1f}, "
            f"{np.nanmax(d_vertical)*1000:.1f}] mm"
        )
        logger.warning(
            "Quasi-vertical decomposition assumes purely vertical motion. "
            "Horizontal motion will introduce errors."
        )

        return DisplacementField(
            displacement_m=d_vertical.astype(np.float32),
            uncertainty_m=uncertainty_vertical.astype(np.float32),
            component="quasi_vertical",
            reference_date=interferogram.metadata.get('primary_time', ''),
            measurement_date=interferogram.metadata.get('secondary_time', ''),
            geo=interferogram.geo,
            metadata={
                'method': 'DInSAR',
                'wavelength_m': wavelength,
                'temporal_baseline_days': interferogram.temporal_baseline_days,
                'perpendicular_baseline_m': interferogram.perpendicular_baseline_m,
                'sign_convention': 'positive = subsidence / away from sensor',
                'decomposition': 'quasi_vertical (assumes no horizontal motion)',
                'los_displacement_range_mm': (
                    float(np.nanmin(d_los) * 1000),
                    float(np.nanmax(d_los) * 1000),
                ),
            },
        )
    else:
        return DisplacementField(
            displacement_m=d_los,
            uncertainty_m=uncertainty_los,
            component="LOS",
            reference_date=interferogram.metadata.get('primary_time', ''),
            measurement_date=interferogram.metadata.get('secondary_time', ''),
            geo=interferogram.geo,
            metadata={
                'method': 'DInSAR',
                'wavelength_m': wavelength,
                'temporal_baseline_days': interferogram.temporal_baseline_days,
                'perpendicular_baseline_m': interferogram.perpendicular_baseline_m,
                'sign_convention': 'positive = subsidence / away from sensor',
            },
        )
