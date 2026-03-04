"""
Goldstein adaptive phase filter for InSAR interferograms.

Reduces phase noise while preserving phase fringes by adaptively
weighting the spectrum based on its own magnitude.

References:
    Goldstein, R.M. & Werner, C.L. (1998). Radar interferogram filtering
    for geophysical applications. Geophysical Research Letters, 25(21), 4035-4038.
"""

import logging

import numpy as np

from ..core.data_models import Interferogram

logger = logging.getLogger('ocean_rs')


def goldstein_filter(
    interferogram: Interferogram,
    alpha: float = 0.5,
    patch_size: int = 32,
    overlap: int = 8,
) -> Interferogram:
    """Apply Goldstein adaptive phase filter to interferogram.

    The filter works on overlapping patches:
        1. Extract patch of complex interferogram (exp(j·φ))
        2. 2D FFT of patch
        3. Compute adaptive weight: H = |S|^α
        4. Apply filter: S_filtered = S · H
        5. IFFT → filtered complex
        6. Overlap-add reconstruction

    Higher α = stronger filtering (more smoothing).
    α = 0: no filtering. α = 1: maximum filtering.

    Args:
        interferogram: Input interferogram with wrapped phase.
        alpha: Filter strength [0, 1]. Default 0.5.
        patch_size: FFT patch size in pixels. Default 32.
        overlap: Overlap between adjacent patches. Default 8.

    Returns:
        Filtered interferogram (new Interferogram with filtered phase).
    """
    phase = interferogram.phase
    rows, cols = phase.shape

    logger.info(
        f"Goldstein filter: α={alpha}, patch={patch_size}×{patch_size}, "
        f"overlap={overlap}"
    )

    if alpha <= 0.0:
        logger.info("Alpha=0, skipping Goldstein filter")
        return interferogram

    if alpha > 1.0:
        logger.warning(f"Goldstein alpha={alpha:.2f} > 1.0 may produce artifacts. Clamping to 1.0.")
        alpha = 1.0

    # Convert phase to complex unit phasor
    complex_ifg = np.exp(1j * phase.astype(np.float64))

    # Use coherence as additional weight if available
    coherence = interferogram.coherence
    if coherence is not None:
        complex_ifg *= coherence

    # Output accumulator and weight map
    filtered = np.zeros_like(complex_ifg)
    weight_map = np.zeros((rows, cols), dtype=np.float64)

    step = patch_size - overlap

    for row_start in range(0, rows, step):
        for col_start in range(0, cols, step):
            row_end = min(row_start + patch_size, rows)
            col_end = min(col_start + patch_size, cols)

            # Extract patch
            patch = complex_ifg[row_start:row_end, col_start:col_end]
            pr, pc = patch.shape

            # Pad to patch_size if necessary
            if pr < patch_size or pc < patch_size:
                padded = np.zeros((patch_size, patch_size), dtype=np.complex128)
                padded[:pr, :pc] = patch
            else:
                padded = patch.copy()

            # 2D FFT
            spectrum = np.fft.fft2(padded)

            # Adaptive filter: H = |S|^alpha (Goldstein & Werner 1998)
            spec_mag = np.abs(spectrum)
            H = spec_mag ** alpha

            # Apply filter
            filtered_spectrum = spectrum * H

            # IFFT
            filtered_patch = np.fft.ifft2(filtered_spectrum)

            # Overlap-add (using raised cosine window for smooth blending)
            window = _raised_cosine_2d(patch_size, patch_size, overlap)
            windowed = filtered_patch * window

            # Accumulate
            filtered[row_start:row_end, col_start:col_end] += windowed[:pr, :pc]
            weight_map[row_start:row_end, col_start:col_end] += window[:pr, :pc].real

    # Normalize by weight
    valid = weight_map > 1e-10
    filtered[valid] /= weight_map[valid]

    # Extract filtered phase
    filtered_phase = np.angle(filtered).astype(np.float32)

    logger.info("Goldstein filter applied")

    return Interferogram(
        phase=filtered_phase,
        coherence=interferogram.coherence,
        unwrapped_phase=interferogram.unwrapped_phase,
        geo=interferogram.geo,
        wavelength_m=interferogram.wavelength_m,
        temporal_baseline_days=interferogram.temporal_baseline_days,
        perpendicular_baseline_m=interferogram.perpendicular_baseline_m,
        incidence_angle=interferogram.incidence_angle,
        metadata={
            **interferogram.metadata,
            'goldstein_alpha': alpha,
            'goldstein_patch_size': patch_size,
        },
    )


def _raised_cosine_2d(rows: int, cols: int, taper: int) -> np.ndarray:
    """Create 2D raised cosine (Tukey) window for overlap-add blending.

    Args:
        rows: Window height.
        cols: Window width.
        taper: Number of pixels to taper at edges.

    Returns:
        2D window array with values in [0, 1].
    """
    def _tukey_1d(n, taper_len):
        if n <= 2 * taper_len:
            return np.hanning(n)
        w = np.ones(n)
        if taper_len > 0:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
            w[:taper_len] = ramp
            w[-taper_len:] = ramp[::-1]
        return w

    w_row = _tukey_1d(rows, taper)
    w_col = _tukey_1d(cols, taper)

    return np.outer(w_row, w_col)
