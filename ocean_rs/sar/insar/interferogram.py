"""
Interferogram formation and coherence estimation.

Forms the interferogram from co-registered SLC pair and estimates
spatial coherence using a rectangular sliding window.

References:
    Just, D. & Bamler, R. (1994). Phase statistics of interferograms
    with applications to synthetic aperture radar. Applied Optics, 33(20).
"""

import logging
from typing import Optional

import numpy as np

from ..core.data_models import SLCImage, InSARPair, Interferogram

logger = logging.getLogger('ocean_rs')


def form_interferogram(
    pair: InSARPair,
    coh_window_range: int = 15,
    coh_window_azimuth: int = 3,
) -> Interferogram:
    """Form interferogram and estimate coherence from co-registered SLC pair.

    Interferogram: ifg = primary * conj(secondary)
    Phase: φ = arg(ifg) ∈ [-π, π]
    Coherence: γ = |<primary · conj(secondary)>| / √(<|primary|²> · <|secondary|²>)

    Note: The sample coherence estimator has a positive bias of ~0.02-0.05
    for low true coherence with a 15x3 window (Touzi et al., 1999). No bias
    correction is applied.

    The coherence window is rectangular because range and azimuth pixel
    spacings differ significantly in SLC data (e.g., S1 IW: ~2.3m range
    vs ~13.9m azimuth). A 15×3 window provides ~same ground resolution
    in both directions.

    Args:
        pair: InSARPair with co-registered primary and secondary SLCs.
        coh_window_range: Coherence window size in range direction (pixels).
        coh_window_azimuth: Coherence window size in azimuth direction (pixels).

    Returns:
        Interferogram with wrapped phase and coherence.
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        raise ImportError(
            "scipy is required for coherence estimation.\n"
            "Install with: conda install -c conda-forge scipy"
        )

    primary = pair.primary.data
    secondary = pair.secondary.data

    if primary.shape != secondary.shape:
        raise ValueError(
            f"SLC shape mismatch: primary={primary.shape}, "
            f"secondary={secondary.shape}. Co-registration required first."
        )

    rows, cols = primary.shape
    logger.info(
        f"Forming interferogram: {rows}×{cols} pixels, "
        f"coherence window: {coh_window_range}×{coh_window_azimuth}"
    )

    # Interferogram formation
    ifg = primary * np.conj(secondary)
    phase = np.angle(ifg).astype(np.float32)

    # Coherence estimation with rectangular window
    window_size = (coh_window_azimuth, coh_window_range)

    # Numerator: |<primary · conj(secondary)>|
    cross_mean = uniform_filter(ifg.real, size=window_size) + \
                 1j * uniform_filter(ifg.imag, size=window_size)
    numerator = np.abs(cross_mean)

    # Denominator: √(<|primary|²> · <|secondary|²>)
    primary_power = uniform_filter(
        (np.abs(primary) ** 2).astype(np.float64), size=window_size
    )
    secondary_power = uniform_filter(
        (np.abs(secondary) ** 2).astype(np.float64), size=window_size
    )
    denominator = np.sqrt(primary_power * secondary_power)

    # Avoid division by zero
    coherence = np.where(
        denominator > 1e-10,
        numerator / denominator,
        0.0
    ).astype(np.float32)

    # Clip to [0, 1]
    np.clip(coherence, 0.0, 1.0, out=coherence)

    logger.info(
        f"Interferogram formed. Mean coherence: {np.nanmean(coherence):.3f}, "
        f"Phase range: [{phase.min():.2f}, {phase.max():.2f}] rad"
    )

    return Interferogram(
        phase=phase,
        coherence=coherence,
        geo=pair.primary.geo,
        wavelength_m=pair.primary.wavelength_m,
        temporal_baseline_days=pair.temporal_baseline_days,
        perpendicular_baseline_m=pair.perpendicular_baseline_m,
        metadata={
            'coherence_window_range': coh_window_range,
            'coherence_window_azimuth': coh_window_azimuth,
            'primary_time': pair.primary.metadata.get('acquisition_time', ''),
            'secondary_time': pair.secondary.metadata.get('acquisition_time', ''),
        },
    )
