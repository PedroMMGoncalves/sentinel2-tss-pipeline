"""
SLC co-registration for InSAR processing.

Two-stage approach:
    1. Coarse: orbit-based or cross-correlation (~1 pixel accuracy)
    2. Fine: coherence-based cross-correlation or ESD (<0.001 pixel accuracy)

For Sentinel-1 TOPS (IW) mode, Enhanced Spectral Diversity (ESD) is used
for fine azimuth refinement to handle the azimuth phase ramp.

References:
    Scheiber, R. & Moreira, A. (2000). Coregistration of interferometric
    SAR images using spectral diversity. IEEE TGRS, 38(5), 2179-2191.

    Prats-Iraola, P. et al. (2012). TOPS Interferometry with TerraSAR-X.
    IEEE TGRS, 50(8), 3179-3188.
"""

import logging
from typing import Optional, Union

import numpy as np

from ..core.data_models import SLCImage, GeoTransform

logger = logging.getLogger('ocean_rs')


def coregister(
    primary: SLCImage,
    secondary: SLCImage,
    method: str = "auto",
    patch_size: int = 128,
    oversample: int = 2,
    grid_spacing: int = 64,
) -> SLCImage:
    """Co-register secondary SLC to primary SLC geometry.

    Two-stage process:
        1. Coarse alignment via cross-correlation of amplitude
        2. Fine alignment:
           - 'coherence': Dense cross-correlation with sub-pixel fitting
           - 'esd': Enhanced Spectral Diversity for TOPS azimuth refinement
           - 'auto': ESD for TOPS data, coherence for Stripmap

    Args:
        primary: Reference SLC image.
        secondary: Secondary SLC image to be resampled.
        method: Co-registration method ('auto', 'esd', 'coherence').
        patch_size: Size of cross-correlation patches (pixels).
        oversample: Oversampling factor for sub-pixel estimation.
        grid_spacing: Spacing between offset estimation points (pixels).

    Returns:
        Co-registered secondary SLCImage aligned to primary geometry.
    """
    try:
        from scipy.ndimage import map_coordinates
        from scipy.signal import correlate2d
    except ImportError:
        raise ImportError(
            "scipy is required for co-registration.\n"
            "Install with: conda install -c conda-forge scipy"
        )

    # Validate SLC shapes match; crop/pad secondary if needed
    if primary.data.shape != secondary.data.shape:
        logger.warning(
            f"SLC shape mismatch: primary={primary.data.shape}, "
            f"secondary={secondary.data.shape}. "
            f"Secondary will be cropped/padded to match primary."
        )
        min_rows = min(primary.data.shape[0], secondary.data.shape[0])
        min_cols = min(primary.data.shape[1], secondary.data.shape[1])
        secondary_data = np.zeros_like(primary.data)
        secondary_data[:min_rows, :min_cols] = secondary.data[:min_rows, :min_cols]
        secondary = SLCImage(
            data=secondary_data,
            geo=secondary.geo,
            metadata=secondary.metadata,
            wavelength_m=secondary.wavelength_m,
            pixel_spacing_range=secondary.pixel_spacing_range,
            pixel_spacing_azimuth=secondary.pixel_spacing_azimuth,
            is_debursted=secondary.is_debursted,
        )

    rows, cols = primary.data.shape
    logger.info(
        f"Co-registering SLC pair: {rows}×{cols} pixels, method={method}"
    )

    # Determine method
    if method == "auto":
        is_tops = primary.is_debursted or _detect_tops(primary)
        method = "esd" if is_tops else "coherence"
        logger.info(f"Auto-detected co-registration method: {method}")

    # Stage 1: Coarse alignment via amplitude cross-correlation
    coarse_offset_r, coarse_offset_a = _coarse_coregistration(
        primary.data, secondary.data, patch_size
    )
    logger.info(
        f"Coarse offset: range={coarse_offset_r:.1f}, azimuth={coarse_offset_a:.1f} pixels"
    )

    # Stage 2: Fine alignment
    # Both methods return 2D offset grids (rows x cols) for per-pixel warping
    if method == "esd":
        fine_offset_r, fine_offset_a = _esd_refinement(
            primary.data, secondary.data,
            coarse_offset_r, coarse_offset_a,
            patch_size, oversample, grid_spacing
        )
    else:
        fine_offset_r, fine_offset_a = _coherence_refinement(
            primary.data, secondary.data,
            coarse_offset_r, coarse_offset_a,
            patch_size, oversample, grid_spacing
        )

    # Combine coarse (scalar) and fine (grid) offsets
    total_offset_r = coarse_offset_r + fine_offset_r
    total_offset_a = coarse_offset_a + fine_offset_a

    # Log representative offset at image center
    if isinstance(total_offset_r, np.ndarray):
        center_r = total_offset_r[rows // 2, cols // 2]
        center_a = total_offset_a[rows // 2, cols // 2]
        mean_r = float(np.mean(total_offset_r))
        mean_a = float(np.mean(total_offset_a))
    else:
        center_r = total_offset_r
        center_a = total_offset_a
        mean_r = float(total_offset_r)
        mean_a = float(total_offset_a)
    logger.info(
        f"Total offset at center: range={center_r:.4f}, azimuth={center_a:.4f} pixels"
    )

    # Resample secondary to primary geometry
    resampled = _resample_slc(secondary.data, total_offset_r, total_offset_a)

    return SLCImage(
        data=resampled,
        geo=primary.geo,  # Aligned to primary geometry
        metadata={
            **secondary.metadata,
            'coregistration_method': method,
            'coregistration_offset_range_mean': mean_r,
            'coregistration_offset_azimuth_mean': mean_a,
        },
        wavelength_m=secondary.wavelength_m,
        pixel_spacing_range=primary.pixel_spacing_range,
        pixel_spacing_azimuth=primary.pixel_spacing_azimuth,
        is_debursted=secondary.is_debursted,
    )


def _coarse_coregistration(
    primary: np.ndarray,
    secondary: np.ndarray,
    patch_size: int,
) -> tuple:
    """Coarse co-registration via amplitude cross-correlation.

    Estimates integer pixel offset using a 3x3 grid of patches across the
    image. The search region in the secondary image is larger than the
    template patch by a search margin of template_size // 2 on each side,
    allowing detection of offsets up to that many pixels. The median of
    valid offsets across all patches is returned for robustness.

    Returns:
        (offset_range, offset_azimuth) in pixels.
    """
    from scipy.signal import correlate2d

    rows, cols = primary.shape
    half = patch_size // 2
    search_margin = patch_size // 2
    corr_threshold = 0.0  # Minimum correlation peak to accept an offset

    # Use 3x3 grid of patches instead of single center
    positions = [
        (rows // 4, cols // 4), (rows // 4, cols // 2), (rows // 4, 3 * cols // 4),
        (rows // 2, cols // 4), (rows // 2, cols // 2), (rows // 2, 3 * cols // 4),
        (3 * rows // 4, cols // 4), (3 * rows // 4, cols // 2), (3 * rows // 4, 3 * cols // 4),
    ]

    offsets_r = []
    offsets_a = []

    for cy, cx in positions:
        # Extract template patch from primary (amplitude)
        r0 = max(0, cy - half)
        r1 = min(rows, cy + half)
        c0 = max(0, cx - half)
        c1 = min(cols, cx + half)
        primary_patch = np.abs(primary[r0:r1, c0:c1])

        if primary_patch.size == 0:
            continue

        # Extract larger search region from secondary (template + margin)
        sr0 = max(0, cy - half - search_margin)
        sr1 = min(rows, cy + half + search_margin)
        sc0 = max(0, cx - half - search_margin)
        sc1 = min(cols, cx + half + search_margin)
        secondary_search = np.abs(secondary[sr0:sr1, sc0:sc1])

        if secondary_search.size == 0:
            continue

        # Cross-correlation (mode='same' relative to larger search region)
        corr = correlate2d(
            secondary_search, primary_patch, mode='same', boundary='fill'
        )

        # Find peak — offset is relative to search region center
        peak_val = np.max(np.abs(corr))
        if peak_val <= corr_threshold:
            continue

        peak_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        center = (corr.shape[0] // 2, corr.shape[1] // 2)

        offsets_a.append(peak_idx[0] - center[0])
        offsets_r.append(peak_idx[1] - center[1])

    # Take median of valid offsets
    if offsets_r:
        coarse_r = float(np.median(offsets_r))
        coarse_a = float(np.median(offsets_a))
    else:
        logger.warning("No valid coarse offset patches found, defaulting to zero offset")
        coarse_r = 0.0
        coarse_a = 0.0

    logger.info(
        f"Coarse registration: {len(offsets_r)}/9 patches yielded valid offsets"
    )

    return coarse_r, coarse_a


def _coherence_refinement(
    primary: np.ndarray,
    secondary: np.ndarray,
    coarse_r: float,
    coarse_a: float,
    patch_size: int,
    oversample: int,
    grid_spacing: int,
) -> tuple:
    """Fine co-registration via coherence-based cross-correlation.

    Estimates sub-pixel offsets on a dense grid of patches, then fits a
    first-order affine (6-parameter) warp model via weighted least squares.
    If fewer than 6 valid points are available, falls back to a constant
    (weighted-average) offset.

    Returns:
        Tuple of (range_offset_grid, azimuth_offset_grid) as 2D arrays
        with the same shape as primary, representing per-pixel fine offsets.
    """
    rows, cols = primary.shape

    from ocean_rs.shared.raster_io import check_memory_for_array
    check_memory_for_array(rows, cols, bytes_per_pixel=16,
                           description="coregistration offset grids")
    half = patch_size // 2

    offsets_r = []
    offsets_a = []
    weights_list = []
    row_positions = []
    col_positions = []

    # Apply coarse shift to secondary for fine estimation
    shifted = _apply_integer_shift(secondary, int(round(coarse_r)), int(round(coarse_a)))

    # Sample offset at grid points
    for y in range(half, rows - half, grid_spacing):
        for x in range(half, cols - half, grid_spacing):
            p_patch = primary[y - half:y + half, x - half:x + half]
            s_patch = shifted[y - half:y + half, x - half:x + half]

            if p_patch.shape != s_patch.shape or p_patch.size == 0:
                continue

            dr, da, coh = _subpixel_offset(p_patch, s_patch, oversample)
            if coh > 0.3:
                offsets_r.append(dr)
                offsets_a.append(da)
                weights_list.append(coh)
                row_positions.append(y)
                col_positions.append(x)

    if not offsets_r:
        logger.warning("No valid offset points found. Fine refinement skipped.")
        return (
            np.zeros((rows, cols), dtype=np.float64),
            np.zeros((rows, cols), dtype=np.float64),
        )

    offsets_r = np.array(offsets_r)
    offsets_a = np.array(offsets_a)
    weights = np.array(weights_list)
    row_positions = np.array(row_positions, dtype=np.float64)
    col_positions = np.array(col_positions, dtype=np.float64)

    n_valid = len(offsets_r)

    if n_valid < 6:
        # Not enough points for affine model, fall back to constant offset
        logger.warning(
            f"Too few valid points ({n_valid}) for affine model, using constant offset"
        )
        fine_r = float(np.average(offsets_r, weights=weights))
        fine_a = float(np.average(offsets_a, weights=weights))
        logger.info(
            f"Fine refinement (constant): range={fine_r:.4f}, azimuth={fine_a:.4f} pixels "
            f"(from {n_valid} valid points)"
        )
        return (
            np.full((rows, cols), fine_r, dtype=np.float64),
            np.full((rows, cols), fine_a, dtype=np.float64),
        )

    # Fit affine polynomial: offset = a0 + a1*row + a2*col
    # Build design matrix [1, row, col]
    A_design = np.column_stack([
        np.ones(n_valid),
        row_positions,
        col_positions,
    ])
    W = np.diag(weights)

    # Weighted least squares: (A^T W A) coeffs = A^T W offsets
    AtWA = A_design.T @ W @ A_design
    AtW = A_design.T @ W

    range_coeffs = np.linalg.lstsq(AtWA, AtW @ offsets_r, rcond=None)[0]
    azimuth_coeffs = np.linalg.lstsq(AtWA, AtW @ offsets_a, rcond=None)[0]

    logger.info(f"Affine range offset model: {range_coeffs}")
    logger.info(f"Affine azimuth offset model: {azimuth_coeffs}")

    # Generate full offset grids from affine model (broadcasting saves ~4.8 GB)
    row_vec = np.arange(rows, dtype=np.float64)[:, np.newaxis]
    col_vec = np.arange(cols, dtype=np.float64)[np.newaxis, :]

    range_offset_grid = (
        range_coeffs[0]
        + range_coeffs[1] * row_vec
        + range_coeffs[2] * col_vec
    )
    azimuth_offset_grid = (
        azimuth_coeffs[0]
        + azimuth_coeffs[1] * row_vec
        + azimuth_coeffs[2] * col_vec
    )

    # Log representative offset at image center
    cr = range_offset_grid[rows // 2, cols // 2]
    ca = azimuth_offset_grid[rows // 2, cols // 2]
    logger.info(
        f"Fine refinement (affine): center range={cr:.4f}, center azimuth={ca:.4f} pixels "
        f"(from {n_valid} valid points)"
    )

    return range_offset_grid, azimuth_offset_grid


def _esd_refinement(
    primary: np.ndarray,
    secondary: np.ndarray,
    coarse_r: float,
    coarse_a: float,
    patch_size: int,
    oversample: int,
    grid_spacing: int = 64,
) -> tuple:
    """Enhanced Spectral Diversity (ESD) refinement for TOPS data.

    ESD exploits the spectral overlap in azimuth between adjacent bursts
    to achieve sub-0.001 pixel azimuth accuracy. However, the correct
    phase-to-pixel conversion requires Doppler centroid frequency difference
    (delta_f_DC) and burst cycle time (T_burst) from SAR metadata:

        fine_a = mean_phase_diff / (2 * pi * delta_f_DC * T_burst)

    For S1 IW TOPS, delta_f_DC ~4900 Hz and T_burst ~2.7s, making the
    denominator ~83,000 rather than 2*pi (~6.28). Without these parameters
    the conversion is off by ~1000x.

    Since we do not currently extract delta_f_DC and T_burst from metadata,
    ESD azimuth refinement is disabled and falls back to coherence-based
    cross-correlation for the azimuth component. Range refinement still
    uses standard sub-pixel cross-correlation.

    Returns:
        (fine_offset_range, fine_offset_azimuth) in pixels.
    """
    logger.warning(
        "ESD azimuth refinement requires Doppler centroid frequency difference "
        "(delta_f_DC) and burst cycle time (T_burst) from SAR metadata, which "
        "are not available in the current architecture. Falling back to "
        "coherence-based cross-correlation for azimuth refinement."
    )

    # Fall back to coherence-based refinement for both range and azimuth
    return _coherence_refinement(
        primary, secondary,
        coarse_r, coarse_a,
        patch_size, oversample, grid_spacing
    )


def _subpixel_offset(
    primary_patch: np.ndarray,
    secondary_patch: np.ndarray,
    oversample: int,
) -> tuple:
    """Estimate sub-pixel offset between two patches.

    Uses oversampled FFT-based cross-correlation with parabolic peak fitting.

    Note: Uses amplitude cross-correlation. Complex cross-correlation would
    provide better accuracy for InSAR but requires coherent SLC data.

    Returns:
        (offset_range, offset_azimuth, coherence).
    """
    # Cross-correlation via FFT (much faster than spatial)
    p = np.abs(primary_patch).astype(np.float64)
    s = np.abs(secondary_patch).astype(np.float64)

    # Normalize
    p_norm = p - p.mean()
    s_norm = s - s.mean()

    p_std = p_norm.std()
    s_std = s_norm.std()

    if p_std < 1e-10 or s_std < 1e-10:
        return 0.0, 0.0, 0.0

    # FFT cross-correlation
    fft_p = np.fft.fft2(p_norm)
    fft_s = np.fft.fft2(s_norm)
    cross = np.fft.ifft2(fft_p * np.conj(fft_s))
    cross_abs = np.abs(cross)

    # Find peak
    peak_idx = np.unravel_index(np.argmax(cross_abs), cross_abs.shape)
    coherence = cross_abs[peak_idx] / (p_std * s_std * p.size)

    # Integer offset (wrap-around aware)
    offset_a = peak_idx[0]
    offset_r = peak_idx[1]
    if offset_a > cross_abs.shape[0] // 2:
        offset_a -= cross_abs.shape[0]
    if offset_r > cross_abs.shape[1] // 2:
        offset_r -= cross_abs.shape[1]

    # Parabolic sub-pixel refinement
    py, px = peak_idx
    ny, nx = cross_abs.shape

    sub_r = 0.0
    sub_a = 0.0

    if 0 < px < nx - 1:
        left = cross_abs[py, px - 1]
        center = cross_abs[py, px]
        right = cross_abs[py, px + 1]
        denom = 2 * (2 * center - left - right)
        if abs(denom) > 1e-10:
            sub_r = (left - right) / denom

    if 0 < py < ny - 1:
        top = cross_abs[py - 1, px]
        center = cross_abs[py, px]
        bottom = cross_abs[py + 1, px]
        denom = 2 * (2 * center - top - bottom)
        if abs(denom) > 1e-10:
            sub_a = (top - bottom) / denom

    return float(offset_r + sub_r), float(offset_a + sub_a), float(coherence)


def _apply_integer_shift(
    data: np.ndarray,
    shift_range: int,
    shift_azimuth: int,
) -> np.ndarray:
    """Apply integer pixel shift to an array."""
    result = np.zeros_like(data)

    src_r0 = max(0, -shift_range)
    src_r1 = min(data.shape[1], data.shape[1] - shift_range)
    dst_r0 = max(0, shift_range)
    dst_r1 = min(data.shape[1], data.shape[1] + shift_range)

    src_a0 = max(0, -shift_azimuth)
    src_a1 = min(data.shape[0], data.shape[0] - shift_azimuth)
    dst_a0 = max(0, shift_azimuth)
    dst_a1 = min(data.shape[0], data.shape[0] + shift_azimuth)

    if dst_a1 > dst_a0 and dst_r1 > dst_r0:
        result[dst_a0:dst_a1, dst_r0:dst_r1] = data[src_a0:src_a1, src_r0:src_r1]

    return result


def _resample_slc(
    data: np.ndarray,
    offset_range: Union[float, np.ndarray],
    offset_azimuth: Union[float, np.ndarray],
) -> np.ndarray:
    """Resample SLC using sub-pixel shifts via scipy.

    Uses cubic spline interpolation (scipy.ndimage.map_coordinates with
    order=3). For complex SLC data, real and imaginary parts are resampled
    separately.

    Args:
        data: Input SLC array (2D, real or complex).
        offset_range: Range offset — either a scalar (constant shift) or a
            2D array of per-pixel offsets (same shape as data).
        offset_azimuth: Azimuth offset — either a scalar or a 2D array.
    """
    from scipy.ndimage import map_coordinates

    rows, cols = data.shape

    # Build base coordinate grids (broadcasting saves ~4.8 GB)
    row_vec = np.arange(rows, dtype=np.float64)[:, np.newaxis]
    col_vec = np.arange(cols, dtype=np.float64)[np.newaxis, :]

    # Add offsets (works for both scalar and array)
    row_grid = row_vec + offset_azimuth
    col_grid = col_vec + offset_range

    # Resample real and imaginary parts separately
    coords = np.array([row_grid.ravel(), col_grid.ravel()])

    if np.iscomplexobj(data):
        real_resampled = map_coordinates(
            data.real.astype(np.float64), coords, order=3, mode='constant'
        ).reshape(rows, cols)
        imag_resampled = map_coordinates(
            data.imag.astype(np.float64), coords, order=3, mode='constant'
        ).reshape(rows, cols)
        return (real_resampled + 1j * imag_resampled).astype(np.complex64)
    else:
        return map_coordinates(
            data.astype(np.float64), coords, order=3, mode='constant'
        ).reshape(rows, cols).astype(data.dtype)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    mid = cumsum[-1] / 2.0
    idx = np.searchsorted(cumsum, mid)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def _detect_tops(slc: SLCImage) -> bool:
    """Detect if SLC is TOPS (IW) mode from metadata."""
    beam_mode = slc.metadata.get('beam_mode', '')
    return beam_mode.upper() in ('IW', 'EW')
