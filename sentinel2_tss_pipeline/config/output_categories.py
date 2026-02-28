"""
Output Category Configuration

Replaces the previous 13 sub-toggles (MarineVisualizationConfig + scattered flags)
with 6 clear category ON/OFF switches. Each category generates ALL its products
when enabled - no individual product selection.

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass


@dataclass
class OutputCategoryConfig:
    """
    Controls which output product categories are generated.

    Each category is a simple ON/OFF toggle. When enabled, ALL products
    in that category are generated. No individual product selection.

    Categories:
        TSS:          Core Jiang TSS products (7 products)
        RGB:          RGB composite visualizations (15 unique composites)
        Indices:      Spectral indices (13 indices)
        WaterClarity: IOP-derived water clarity products (6 products)
        HAB:          Harmful Algal Bloom detection (9 products)
        TrophicState: Carlson Trophic State Index (3 products)
    """
    enable_tss: bool = True           # 7 products: TSS, Absorption, Backscattering, ReferenceBand, WaterTypes, ValidMask, Legend
    enable_rgb: bool = True           # 15 unique RGB composites (deduplicated)
    enable_indices: bool = True       # 13 spectral indices (NDWI, NDTI, NDCI, pSDB, CDOM, etc.)
    enable_water_clarity: bool = False  # 6 products: SecchiDepth, Kd, ClarityIndex, EuphoticDepth, BeamAttenuation, RelativeTurbidity
    enable_hab: bool = False          # 9 products: NDCI/MCI Values+Bloom, Probability, RiskLevel, PotentialBloom, Biomass Alerts
    enable_trophic_state: bool = False  # 3 products: TSI_Chlorophyll, TSI_Secchi, TrophicClass
