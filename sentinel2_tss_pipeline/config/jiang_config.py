"""
Jiang TSS Methodology Configuration

Based on: Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
"Remotely Estimating Total Suspended Solids Concentration in Clear to
Extremely Turbid Waters Using a Novel Semi-Analytical Method."
Remote Sensing of Environment, 258, 112386.
DOI: https://doi.org/10.1016/j.rse.2021.112386

Water Type Classification: Type I-IV based on Rrs(490), Rrs(560), Rrs(620), Rrs(754)

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .water_quality_config import WaterQualityConfig
    from .marine_config import MarineVisualizationConfig


@dataclass
class JiangTSSConfig:
    """Jiang TSS methodology configuration"""
    enable_jiang_tss: bool = True  # Enable by default
    output_intermediates: bool = True
    water_mask_threshold: float = 0.01
    tss_valid_range: tuple = (0.01, 10000)  # g/m³
    output_comparison_stats: bool = True

    # Advanced algorithms configuration
    enable_advanced_algorithms: bool = True
    water_quality_config: Optional['WaterQualityConfig'] = None

    # Marine visualization configuration
    enable_marine_visualization: bool = True  # ENABLED BY DEFAULT
    marine_viz_config: Optional['MarineVisualizationConfig'] = None

    def __post_init__(self):
        """Initialize advanced and marine visualization configs"""
        # Import here to avoid circular imports at module load time
        from .water_quality_config import WaterQualityConfig
        from .marine_config import MarineVisualizationConfig

        if self.enable_advanced_algorithms and self.water_quality_config is None:
            self.water_quality_config = WaterQualityConfig()

        if self.enable_marine_visualization and self.marine_viz_config is None:
            self.marine_viz_config = MarineVisualizationConfig()
