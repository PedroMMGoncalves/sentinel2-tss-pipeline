"""
Sentinel-2 Processing Configuration Classes

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResamplingConfig:
    """S2 Resampling configuration"""
    target_resolution: str = "10"
    upsampling_method: str = "Bilinear"
    downsampling_method: str = "Mean"
    flag_downsampling: str = "First"
    resample_on_pyramid_levels: bool = True


@dataclass
class SubsetConfig:
    """Spatial subset configuration"""
    geometry_wkt: Optional[str] = None
    sub_sampling_x: int = 1
    sub_sampling_y: int = 1
    full_swath: bool = False
    copy_metadata: bool = True
    pixel_start_x: Optional[int] = None
    pixel_start_y: Optional[int] = None
    pixel_size_x: Optional[int] = None
    pixel_size_y: Optional[int] = None


@dataclass
class C2RCCConfig:
    """Enhanced C2RCC atmospheric correction configuration with SNAP defaults"""

    # Basic water parameters
    salinity: float = 35.0
    temperature: float = 15.0
    ozone: float = 330.0
    pressure: float = 1000.0  # SNAP default
    elevation: float = 0.0

    # Neural network configuration
    net_set: str = "C2RCC-Nets"

    # DEM configuration
    dem_name: str = "Copernicus 30m Global DEM"

    # Auxiliary data - ECMWF enabled by default
    use_ecmwf_aux_data: bool = True
    atmospheric_aux_data_path: str = ""
    alternative_nn_path: str = ""

    # Essential output products (SNAP defaults + uncertainties)
    output_as_rrs: bool = True
    output_rhown: bool = True          # Required for TSS
    output_kd: bool = True
    output_uncertainties: bool = True  # Ensures unc_tsm.img and unc_chl.img
    output_ac_reflectance: bool = True
    output_rtoa: bool = True

    # Advanced atmospheric products (SNAP defaults)
    output_rtosa_gc: bool = False
    output_rtosa_gc_aann: bool = False
    output_rpath: bool = False
    output_tdown: bool = False
    output_tup: bool = False
    output_oos: bool = False

    # Advanced parameters
    derive_rw_from_path_and_transmittance: bool = False
    valid_pixel_expression: str = "B8 > 0 && B8 < 0.1"

    # Thresholds
    threshold_rtosa_oos: float = 0.05
    threshold_ac_reflec_oos: float = 0.1
    threshold_cloud_tdown865: float = 0.955

    # TSM and CHL parameters (SNAP defaults)
    # Reference: SNAP C2RCC documentation
    # TSM = TSMfac × iop_btot^TSMexp
    tsm_fac: float = 1.06
    tsm_exp: float = 0.942
    chl_fac: float = 21.0
    chl_exp: float = 1.04

    def apply_nn_presets(self):
        """Auto-adjust thresholds based on selected neural network.

        C2X-COMPLEX-Nets is designed for extreme/turbid waters and
        benefits from relaxed thresholds to include surf zone pixels
        that standard C2RCC-Nets would reject.
        """
        presets = {
            'C2RCC-Nets': {'b8_max': 0.1, 'rtosa_oos': 0.05, 'ac_oos': 0.1},
            'C2X-Nets': {'b8_max': 0.15, 'rtosa_oos': 0.08, 'ac_oos': 0.15},
            'C2X-COMPLEX-Nets': {'b8_max': 0.2, 'rtosa_oos': 0.1, 'ac_oos': 0.2},
        }
        if self.net_set not in presets:
            raise ValueError(f"Unknown neural network set: {self.net_set}. "
                           f"Valid options: {list(presets.keys())}")
        p = presets[self.net_set]
        self.valid_pixel_expression = f"B8 > 0 && B8 < {p['b8_max']}"
        self.threshold_rtosa_oos = p['rtosa_oos']
        self.threshold_ac_reflec_oos = p['ac_oos']
