"""
Marine Visualization Configuration

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass


@dataclass
class MarineVisualizationConfig:
    """Configuration for marine visualization products"""

    # RGB options
    generate_natural_color: bool = True
    generate_false_color: bool = True
    generate_water_specific: bool = True
    generate_research_combinations: bool = False

    # Spectral indices options
    generate_water_quality_indices: bool = True
    generate_chlorophyll_indices: bool = True
    generate_turbidity_indices: bool = True
    generate_advanced_indices: bool = False

    # Output format options
    rgb_format: str = 'GeoTIFF'
    export_metadata: bool = True
    create_overview_images: bool = True

    # Enhancement options
    apply_contrast_enhancement: bool = True
    contrast_method: str = 'percentile_stretch'
    percentile_range: tuple = (2, 98)

    # Note: Water masking is controlled by JiangTSSConfig
    # Marine viz receives mask from calling code if enabled
