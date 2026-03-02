"""
Advanced Aquatic Algorithms Configuration

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WaterQualityConfig:
    """Configuration for advanced aquatic algorithms"""

    # Trophic state calculation
    enable_trophic_state: bool = False
    tsi_include_secchi: bool = False
    tsi_include_phosphorus: bool = False

    # Water clarity calculation
    enable_water_clarity: bool = True
    solar_zenith_angle: float = 30.0

    # HAB detection
    enable_hab_detection: bool = True
    hab_biomass_threshold: float = 20.0
    hab_extreme_threshold: float = 100.0

    # Upwelling detection
    enable_upwelling_detection: bool = True
    upwelling_chl_threshold: float = 10.0

    # River plume tracking
    enable_river_plume_tracking: bool = True
    plume_tss_threshold: float = 15.0
    plume_distance_threshold: float = 10000

    # Particle size estimation
    enable_particle_size: bool = True
    particle_size_wavelengths: Optional[List[int]] = None

    # Primary productivity
    enable_primary_productivity: bool = True
    productivity_model: str = 'vgpm'
    day_length: float = 12.0

    # Output options
    save_intermediate_products: bool = True
    create_classification_maps: bool = True
    generate_statistics: bool = True

    def __post_init__(self):
        if self.particle_size_wavelengths is None:
            self.particle_size_wavelengths = [443, 490, 560, 665, 705]
