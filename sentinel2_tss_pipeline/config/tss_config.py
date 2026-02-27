"""
TSS Processing Configuration

Based on: Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
"Remotely Estimating Total Suspended Solids Concentration in Clear to
Extremely Turbid Waters Using a Novel Semi-Analytical Method."
Remote Sensing of Environment, 258, 112386.
DOI: https://doi.org/10.1016/j.rse.2021.112386

Water Type Classification: Type I-IV based on Rrs(490), Rrs(560), Rrs(620), Rrs(754)

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass, field
from typing import Optional

from .output_categories import OutputCategoryConfig


@dataclass
class TSSConfig:
    """TSS processing and output configuration.

    Replaces the old JiangTSSConfig. Controls TSS algorithm settings,
    water masking, visualization, water quality sub-processing,
    and output category selection.
    """
    enable_tss_processing: bool = True
    output_intermediates: bool = True
    tss_valid_range: tuple = (0.01, 10000)  # g/m³
    output_comparison_stats: bool = True

    # Water masking - auto-detect enabled by default
    # NIR mask: water = NIR(865nm) < water_mask_threshold
    auto_water_mask: bool = True  # ENABLED by default (fixes land contamination)
    water_mask_shapefile: Optional[str] = None  # User override for exact coastline
    water_mask_threshold: float = 0.03  # NIR Rrs threshold for water detection

    # Water quality sub-processing (HAB, clarity, trophic state)
    enable_water_quality: bool = True
    water_quality_config: Optional[object] = None  # WaterQualityConfig, lazy init

    # Visualization sub-processing (RGB composites + spectral indices)
    enable_visualization: bool = True

    # Output categories (replaces 13 sub-toggles with 6 clear categories)
    output_categories: OutputCategoryConfig = field(default_factory=OutputCategoryConfig)

    def __post_init__(self):
        """Initialize water quality config if enabled and not provided."""
        if self.water_quality_config is None and self.enable_water_quality:
            from .water_quality_config import WaterQualityConfig
            self.water_quality_config = WaterQualityConfig()
