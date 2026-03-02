"""
FFT-based swell wavelength and direction extraction from SAR imagery.

Tiles the image, computes 2D FFT per tile, and finds the dominant spectral
peak to determine swell wavelength and propagation direction.
"""

import logging
import numpy as np
from typing import Optional

from ..core.data_models import OceanImage, SwellField, GeoTransform

logger = logging.getLogger('ocean_rs')


def extract_swell(image: OceanImage,
                  tile_size_m: float = 512.0,
                  overlap: float = 0.5,
                  min_wavelength_m: float = 50.0,
                  max_wavelength_m: float = 600.0,
                  confidence_threshold: float = 0.3) -> SwellField:
    """Extract dominant swell wavelength and direction from SAR image.

    Algorithm:
        1. Tile image with configurable overlap
        2. Per tile: apply Hanning window, compute 2D FFT
        3. Compute power spectrum, mask to wavelength range
        4. Find dominant peak -> wavelength + direction
        5. Compute confidence from spectral peak SNR

    Args:
        image: OceanImage to analyze
        tile_size_m: Tile size in meters (default 512)
        overlap: Tile overlap fraction (0-1, default 0.5)
        min_wavelength_m: Minimum wavelength to detect (default 50m)
        max_wavelength_m: Maximum wavelength to detect (default 600m)
        confidence_threshold: Minimum confidence to keep (default 0.3)

    Returns:
        SwellField with wavelength, direction, and confidence per tile
    """
    data = image.data
    pixel_m = image.pixel_spacing_m

    tile_px = int(tile_size_m / pixel_m)
    tile_px = _next_power_of_2(tile_px)
    step_px = int(tile_px * (1 - overlap))

    rows, cols = data.shape
    logger.info(f"FFT extraction: image={rows}x{cols}px, tile={tile_px}px, "
                f"step={step_px}px, pixel={pixel_m}m")

    row_starts = list(range(0, rows - tile_px + 1, step_px))
    col_starts = list(range(0, cols - tile_px + 1, step_px))

    if not row_starts or not col_starts:
        raise ValueError(
            f"Image too small ({rows}x{cols}px) for tile size {tile_px}px"
        )

    n_tiles = len(row_starts) * len(col_starts)
    logger.info(f"Processing {n_tiles} tiles ({len(row_starts)}x{len(col_starts)})")

    window = np.outer(np.hanning(tile_px), np.hanning(tile_px))

    freqs = np.fft.fftfreq(tile_px, d=pixel_m)
    fx, fy = np.meshgrid(freqs, freqs)
    freq_magnitude = np.sqrt(fx**2 + fy**2)
    wavelength_map = np.zeros_like(freq_magnitude)
    nonzero = freq_magnitude > 0
    wavelength_map[nonzero] = 1.0 / freq_magnitude[nonzero]

    valid_wavelength = (wavelength_map >= min_wavelength_m) & \
                       (wavelength_map <= max_wavelength_m)

    wavelengths = []
    directions = []
    confidences = []
    centers_x = []
    centers_y = []

    for r0 in row_starts:
        for c0 in col_starts:
            tile = data[r0:r0+tile_px, c0:c0+tile_px].astype(np.float64)

            valid_frac = np.sum(np.isfinite(tile) & (tile != 0)) / tile.size
            if valid_frac < 0.5:
                continue

            tile_mean = np.nanmean(tile)
            tile = np.where(np.isfinite(tile), tile, tile_mean)

            tile -= tile_mean
            tile *= window

            fft2 = np.fft.fft2(tile)
            power = np.abs(np.fft.fftshift(fft2))**2

            wl_shifted = np.fft.fftshift(wavelength_map)
            valid_shifted = np.fft.fftshift(valid_wavelength)
            fx_shifted = np.fft.fftshift(fx)
            fy_shifted = np.fft.fftshift(fy)

            masked_power = power * valid_shifted

            if np.max(masked_power) == 0:
                continue

            peak_idx = np.unravel_index(np.argmax(masked_power), power.shape)
            peak_power = masked_power[peak_idx]

            wl = wl_shifted[peak_idx]

            peak_fx = fx_shifted[peak_idx]
            peak_fy = fy_shifted[peak_idx]
            direction = np.degrees(np.arctan2(peak_fx, peak_fy)) % 360

            mean_power = np.mean(masked_power[masked_power > 0])
            snr = peak_power / mean_power if mean_power > 0 else 0
            confidence = min(1.0, snr / 10.0)

            if confidence >= confidence_threshold:
                wavelengths.append(wl)
                directions.append(direction)
                confidences.append(confidence)

                cx = image.geo.origin_x + (c0 + tile_px/2) * image.geo.pixel_size_x
                cy = image.geo.origin_y + (r0 + tile_px/2) * image.geo.pixel_size_y
                centers_x.append(cx)
                centers_y.append(cy)

    n_valid = len(wavelengths)
    logger.info(f"FFT complete: {n_valid}/{n_tiles} tiles above confidence threshold")

    if n_valid == 0:
        raise ValueError(
            "No valid swell detected. Try adjusting wavelength range "
            "or lowering confidence threshold."
        )

    return SwellField(
        wavelength=np.array(wavelengths),
        direction=np.array(directions),
        confidence=np.array(confidences),
        tile_centers_x=np.array(centers_x),
        tile_centers_y=np.array(centers_y),
        tile_size_m=tile_size_m,
        geo=image.geo,
    )


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    p = 1
    while p < n:
        p *= 2
    return p
