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
    water masking, and output category selection.
    """
    enable_tss_processing: bool = True
    output_intermediates: bool = True
    tss_valid_range: tuple = (0.01, 10000)  # g/m³
    output_comparison_stats: bool = True

    # Water masking - auto-detect enabled by default
    # NDWI+NIR mask: water = (NDWI > 0) AND (NIR < 0.03)
    auto_water_mask: bool = True  # ENABLED by default (fixes land contamination)
    water_mask_shapefile: Optional[str] = None  # User override for exact coastline

    # Output categories (replaces 13 sub-toggles with 6 clear categories)
    output_categories: OutputCategoryConfig = field(default_factory=OutputCategoryConfig)
