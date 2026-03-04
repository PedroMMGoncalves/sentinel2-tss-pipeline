"""
Goldstein adaptive phase filter for InSAR interferograms.

Reduces phase noise while preserving phase fringes by adaptively
weighting the spectrum based on its own magnitude.

References:
    Goldstein, R.M. & Werner, C.L. (1998). Radar interferogram filtering
    for geophysical applications. Geophysical Research Letters, 25(21), 4035-4038.

    Baran, I., Stewart, M.P., Kampes, B.M., Perski, Z., & Lilly, P. (2003).
    A modification to the Goldstein radar interferogram filter.
    IEEE TGRS, 41(9), 2114-2118.
"""

import logging

import numpy as np

from ..core.data_models import Interferogram

logger = logging.getLogger('ocean_rs')

# --- Numba optional import ---
try:
    from numba import njit
    HAS_NUMBA = True
    logger.debug("Numba available — JIT-accelerated Goldstein filter enabled")
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity decorator fallback when Numba is not installed."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


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

    When coherence is available, uses the coherence-weighted variant
    (Baran et al., 2003) for better noise suppression.

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
    if phase is None:
        raise ValueError("Interferogram phase is None — cannot filter")
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

    step = patch_size - overlap
    if step < 1:
        logger.warning(
            f"overlap ({overlap}) >= patch_size ({patch_size}), clamping step to 1"
        )
        step = 1

    # Build patch grid coordinates
    row_starts = list(range(0, rows, step))
    col_starts = list(range(0, cols, step))
    n_patches = len(row_starts) * len(col_starts)

    logger.info(f"Goldstein filter: {n_patches} patches")

    from ocean_rs.shared.raster_io import check_memory_for_array
    # 3 complex128 arrays: patches_3d + spectra + filtered_patches
    check_memory_for_array(
        n_patches, patch_size * patch_size,
        bytes_per_pixel=16 * 3,  # 3 × complex128
        description="Goldstein filter batch FFT"
    )

    if HAS_NUMBA:
        logger.info("Numba JIT accumulation enabled (first call compiles — may take a few seconds)")

    # --- Extract all patches into a 3D stack for batch FFT ---
    patches_3d = np.zeros((n_patches, patch_size, patch_size), dtype=np.complex128)
    patch_rows = np.empty(n_patches, dtype=np.int32)
    patch_cols = np.empty(n_patches, dtype=np.int32)
    patch_pr = np.empty(n_patches, dtype=np.int32)
    patch_pc = np.empty(n_patches, dtype=np.int32)

    idx = 0
    for rs in row_starts:
        for cs in col_starts:
            re = min(rs + patch_size, rows)
            ce = min(cs + patch_size, cols)
            pr = re - rs
            pc = ce - cs
            patches_3d[idx, :pr, :pc] = complex_ifg[rs:re, cs:ce]
            patch_rows[idx] = rs
            patch_cols[idx] = cs
            patch_pr[idx] = pr
            patch_pc[idx] = pc
            idx += 1

    # Batch FFT → spectral weighting → batch IFFT (leverages threaded FFTW/MKL)
    spectra = np.fft.fft2(patches_3d, axes=(1, 2))
    H = np.abs(spectra) ** alpha
    filtered_patches = np.fft.ifft2(spectra * H, axes=(1, 2))

    # Compute window once
    window = _raised_cosine_2d(patch_size, patch_size, overlap)

    # Overlap-add accumulation (Numba JIT if available, else plain NumPy)
    if HAS_NUMBA:
        filtered, weight_map = _accumulate_patches_jit(
            filtered_patches, window,
            patch_rows, patch_cols, patch_pr, patch_pc,
            rows, cols, patch_size,
        )
    else:
        filtered = np.zeros((rows, cols), dtype=np.complex128)
        weight_map = np.zeros((rows, cols), dtype=np.float64)
        for i in range(n_patches):
            rs = patch_rows[i]
            cs = patch_cols[i]
            pr = patch_pr[i]
            pc = patch_pc[i]
            if pr < patch_size or pc < patch_size:
                w = _raised_cosine_2d(pr, pc, overlap)
                filtered[rs:rs+pr, cs:cs+pc] += (filtered_patches[i, :pr, :pc] * w)
                weight_map[rs:rs+pr, cs:cs+pc] += w
            else:
                windowed = filtered_patches[i] * window
                filtered[rs:rs+pr, cs:cs+pc] += windowed
                weight_map[rs:rs+pr, cs:cs+pc] += window

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
            **(interferogram.metadata or {}),
            'goldstein_alpha': alpha,
            'goldstein_patch_size': patch_size,
        },
    )


@njit(cache=True)
def _accumulate_patches_jit(
    filtered_patches, window,
    patch_rows, patch_cols, patch_pr, patch_pc,
    rows, cols, patch_size,
):
    """Overlap-add accumulation — Numba JIT compiled.

    Serial @njit (not parallel) because overlap-add writes to shared
    output locations. The speedup comes from eliminating Python dispatch
    overhead per patch (~174k iterations for 10k×10k images).
    """
    output_re = np.zeros((rows, cols), dtype=np.float64)
    output_im = np.zeros((rows, cols), dtype=np.float64)
    weight = np.zeros((rows, cols), dtype=np.float64)

    n_patches = len(patch_rows)
    for i in range(n_patches):
        rs = patch_rows[i]
        cs = patch_cols[i]
        pr = patch_pr[i]
        pc = patch_pc[i]
        # Note: boundary patches use top-left portion of window.
        # Weight normalization compensates for edge taper asymmetry.
        for r in range(pr):
            for c in range(pc):
                w = window[r, c]
                fp = filtered_patches[i, r, c]
                output_re[rs + r, cs + c] += fp.real * w
                output_im[rs + r, cs + c] += fp.imag * w
                weight[rs + r, cs + c] += w

    # Reconstruct complex output
    output = np.empty((rows, cols), dtype=np.complex128)
    for r in range(rows):
        for c in range(cols):
            output[r, c] = output_re[r, c] + 1j * output_im[r, c]

    return output, weight


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
