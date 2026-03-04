"""
SAR Processing Configuration.

Master configuration dataclass following OceanRS pattern.
Supports bathymetry, InSAR, and displacement processing modes.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from .download_config import DownloadConfig, SearchConfig

logger = logging.getLogger('ocean_rs')


@dataclass
class FFTConfig:
    """FFT swell extraction parameters."""
    tile_size_m: float = 1024.0  # Must be >= 2x max_wavelength_m for Nyquist
    overlap: float = 0.5
    min_wavelength_m: float = 50.0
    max_wavelength_m: float = 600.0
    confidence_threshold: float = 0.3
    window_function: str = "hann"

    def __post_init__(self):
        if self.overlap < 0.0 or self.overlap >= 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        if self.tile_size_m < self.max_wavelength_m * 2:
            logger.warning(
                f"tile_size_m ({self.tile_size_m}) < 2*max_wavelength_m "
                f"({self.max_wavelength_m * 2}). May violate Nyquist."
            )


@dataclass
class DepthInversionConfig:
    """Depth inversion parameters."""
    max_depth_m: float = 100.0
    wave_period_source: str = "wavewatch3"
    manual_wave_period: float = 10.0
    gravity: float = 9.81
    max_iterations: int = 50
    convergence_tol: float = 1e-6


@dataclass
class CompositingConfig:
    """Multi-temporal compositing parameters."""
    enabled: bool = True
    method: str = "weighted_median"


@dataclass
class InSARConfig:
    """InSAR processing parameters."""
    coregistration_method: str = "auto"      # "auto" | "esd" | "coherence"
    coregistration_patch_size: int = 128
    coregistration_oversample: int = 2
    coherence_window_range: int = 15         # Rectangular window (range direction)
    coherence_window_azimuth: int = 3        # Rectangular window (azimuth direction)
    phase_filter_alpha: float = 0.5          # Goldstein filter strength [0, 1]
    phase_filter_patch_size: int = 32
    unwrapping_method: str = "auto"          # "auto" | "snaphu" | "quality_guided"
    remove_topography: bool = True
    dem_path: str = ""                       # Empty = auto-download SRTM
    output_coherence: bool = True
    output_interferogram: bool = True
    output_unwrapped: bool = True


@dataclass
class DisplacementConfig:
    """Displacement analysis parameters."""
    mode: str = "dinsar"                           # "dinsar" | "sbas"
    max_temporal_baseline_days: int = 180
    max_perpendicular_baseline_m: float = 200.0
    atmospheric_filter: bool = False
    temporal_coherence_threshold: float = 0.7
    reference_point: Optional[tuple] = None        # (lon, lat) for reference
    output_quasi_vertical: bool = True
    output_los: bool = True


@dataclass
class SARProcessingConfig:
    """Complete SAR processing configuration."""
    config_version: str = "1.0"
    processing_mode: str = "bathymetry"  # "bathymetry" | "insar" | "displacement"
    search_config: Optional[SearchConfig] = None
    download_config: Optional[DownloadConfig] = None
    fft_config: Optional[FFTConfig] = None
    depth_config: Optional[DepthInversionConfig] = None
    compositing_config: Optional[CompositingConfig] = None
    insar_config: Optional[InSARConfig] = None
    displacement_config: Optional[DisplacementConfig] = None
    snap_gpt_path: str = ""
    output_directory: str = ""
    export_geotiff: bool = True
    export_png: bool = True
    memory_limit_gb: int = 8

    def __post_init__(self):
        if self.config_version != "1.0":
            raise ValueError(
                f"Unsupported SAR config version: {self.config_version}. "
                f"Expected '1.0'."
            )
        if self.search_config is None:
            self.search_config = SearchConfig()
        if self.download_config is None:
            self.download_config = DownloadConfig()
        if self.fft_config is None:
            self.fft_config = FFTConfig()
        if self.depth_config is None:
            self.depth_config = DepthInversionConfig()
        if self.compositing_config is None:
            self.compositing_config = CompositingConfig()
        if self.insar_config is None:
            self.insar_config = InSARConfig()
        if self.displacement_config is None:
            self.displacement_config = DisplacementConfig()
