"""
SAR Processing Configuration.

Master configuration dataclass following OceanRS pattern.
"""

from dataclasses import dataclass
from .download_config import DownloadConfig, SearchConfig


@dataclass
class FFTConfig:
    """FFT swell extraction parameters."""
    tile_size_m: float = 512.0
    overlap: float = 0.5
    min_wavelength_m: float = 50.0
    max_wavelength_m: float = 600.0
    confidence_threshold: float = 0.3
    window_function: str = "hann"


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
class SARProcessingConfig:
    """Complete SAR processing configuration."""
    search_config: SearchConfig = None
    download_config: DownloadConfig = None
    fft_config: FFTConfig = None
    depth_config: DepthInversionConfig = None
    compositing_config: CompositingConfig = None
    snap_gpt_path: str = ""
    output_directory: str = ""
    export_geotiff: bool = True
    export_png: bool = True
    memory_limit_gb: int = 8

    def __post_init__(self):
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
