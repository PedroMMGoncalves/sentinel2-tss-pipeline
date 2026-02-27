"""
Water Quality Processor.

Advanced aquatic algorithms processor with complete scientific implementations.

Algorithms implemented:
- Water clarity indices (Secchi depth, euphotic depth, diffuse attenuation)
- Harmful Algal Bloom (HAB) detection (NDCI, MCI)
- Trophic State Index (Carlson 1977)

Note: FLH (Fluorescence Line Height) removed - Sentinel-2 lacks 681nm band for true
fluorescence peak detection.

References:
- Kirk, J.T.O. (2011). Light and photosynthesis in aquatic ecosystems.
- Lee, Z. et al. (2002). Deriving inherent optical properties from water color.
- Tyler, J.E. (1968). The Secchi disc.
- Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index.
- Gower, J. et al. (2005). Maximum Chlorophyll Index.
- Carlson, R.E. (1977). A trophic state index for lakes.
"""

import logging
from typing import Dict, Optional

import numpy as np

from ..config import WaterQualityConfig

logger = logging.getLogger('sentinel2_tss_pipeline')


class WaterQualityConstants:
    """Constants for advanced aquatic algorithms based on scientific literature"""

    # OC3 algorithm coefficients (O'Reilly et al., 1998)
    OC3_COEFFS = [0.3272, -2.9940, 2.7218, -1.2259, -0.5683]

    # VGPM coefficients (Behrenfeld & Falkowski, 1997)
    VGPM_PBOPT_MAX = 4.0  # mg C (mg Chl)^-1 h^-1
    VGPM_SST_OPT = 20.0   # Optimal temperature (C)

    # Particle size coefficients (Boss et al., 2001)
    PARTICLE_SIZE_THRESHOLDS = {
        'small': 1.5,    # eta > 1.5: Small particles (<2 um)
        'medium': 0.5,   # 0.5 < eta < 1.5: Medium particles (2-20 um)
        'large': 0.0     # eta < 0.5: Large particles (>20 um)
    }


class WaterQualityProcessor:
    """
    Advanced aquatic algorithms processor with complete scientific implementations
    """

    def __init__(self):
        self.constants = WaterQualityConstants()
        logger.info("Initialized Advanced Aquatic Processor with scientific algorithms")


    def calculate_water_clarity(self, absorption: np.ndarray,
                               backscattering: np.ndarray,
                               solar_zenith: float = 30.0) -> Dict[str, np.ndarray]:
        """
        Calculate water clarity indices from bio-optical properties

        References:
        - Kirk, J.T.O. (2011). Light and photosynthesis in aquatic ecosystems. Cambridge University Press.
        - Lee, Z. et al. (2002). Deriving inherent optical properties from water color. Applied Optics, 41(27), 5755-5772.
        - Tyler, J.E. (1968). The Secchi disc. Limnology and Oceanography, 13(1), 1-6.
        - Preisendorfer, R.W. (1986). Secchi disk science: Visual optics of natural waters.
          Limnology and Oceanography, 31(5), 909-926.

        Args:
            absorption: Absorption coefficient at 443nm (m^-1)
            backscattering: Backscattering coefficient at 443nm (m^-1)
            solar_zenith: Solar zenith angle (degrees)

        Returns:
            Dictionary with clarity indices
        """
        try:
            logger.debug("Calculating water clarity indices")

            # Convert solar zenith to cosine
            mu0 = np.cos(np.radians(solar_zenith))

            # Gordon equation for diffuse attenuation coefficient (Gordon, 1989)
            kd = absorption + backscattering * (1 + 0.425 * mu0) / mu0

            # Tyler (1968) Secchi depth approximation
            secchi_depth = 1.7 / kd

            # Water clarity index (0-1 scale)
            clarity_index = 1 / (1 + kd)

            # Euphotic depth (1% light level)
            euphotic_depth = 4.605 / kd  # ln(100) / kd

            # Beam attenuation coefficient (approximate)
            beam_attenuation = absorption + backscattering

            # Relative turbidity index (dimensionless, 0-1 scale)
            # Note: This is NOT calibrated NTU - use for relative comparison only
            relative_turbidity_index = np.clip(backscattering * 100, 0, 1)

            results = {
                'diffuse_attenuation': kd,
                'secchi_depth': secchi_depth,
                'clarity_index': clarity_index,
                'euphotic_depth': euphotic_depth,
                'beam_attenuation': beam_attenuation,
                'relative_turbidity_index': relative_turbidity_index
            }

            # Calculate statistics
            valid_pixels = np.sum(~np.isnan(kd))
            if valid_pixels > 0:
                logger.debug(f"Water clarity: {valid_pixels} pixels, Secchi={np.nanmean(secchi_depth):.1f}m")

            return results

        except Exception as e:
            logger.error(f"Error calculating water clarity: {e}")
            return {}

    def detect_harmful_algal_blooms(self, chlorophyll: Optional[np.ndarray],
                                phycocyanin: Optional[np.ndarray],
                                rrs_bands: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        HAB detection using Sentinel-2 spectral bands (NDCI, MCI)

        Note: FLH removed - S2 lacks 681nm band for true fluorescence detection.

        References:
        - Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index.
        - Gower, J. et al. (2005). Maximum Chlorophyll Index.
        """
        try:
            logger.debug("Detecting harmful algal blooms")

            results = {}

            if not rrs_bands:
                logger.warning("No Rrs bands available for HAB detection")
                return {}

            # Get reference shape from any available band
            ref_band = list(rrs_bands.values())[0]
            shape = ref_band.shape

            # Initialize result arrays
            hab_probability = np.zeros(shape, dtype=np.float32)
            ndci_values = np.full(shape, np.nan, dtype=np.float32)
            # flh_values removed: S2 lacks 681nm band for true FLH
            mci_values = np.full(shape, np.nan, dtype=np.float32)

            algorithms_applied = []

            # Method 1: Normalized Difference Chlorophyll Index (NDCI)
            if 705 in rrs_bands and 665 in rrs_bands:
                logger.debug("Calculating NDCI")

                band_705 = rrs_bands[705]
                band_665 = rrs_bands[665]

                valid_mask = (
                    (~np.isnan(band_705)) &
                    (~np.isnan(band_665)) &
                    (band_705 > 0) &
                    (band_665 > 0)
                )

                denominator = band_705 + band_665
                valid_mask = valid_mask & (denominator > 1e-8)

                if np.any(valid_mask):
                    ndci_values[valid_mask] = ((band_705[valid_mask] - band_665[valid_mask]) /
                                            denominator[valid_mask])

                    # NDCI bloom threshold (Mishra & Mishra, 2012)
                    ndci_bloom = (ndci_values > 0.05).astype(np.float32)
                    results['ndci_bloom'] = ndci_bloom
                    results['ndci_values'] = ndci_values

                    # Add to probability calculation
                    hab_probability += ndci_bloom * 0.3
                    algorithms_applied.append("NDCI")

                    logger.debug(f"NDCI calculated for {np.sum(valid_mask)} pixels")

            # FLH (Fluorescence Line Height) removed: S2 lacks the 681nm band required for
            # true fluorescence peak detection. The 705nm band captures red-edge reflectance,
            # not the chlorophyll fluorescence emission at ~685nm.

            # Method 2: Maximum Chlorophyll Index (MCI) - Gower et al. (2005)
            if all(band in rrs_bands for band in [665, 705, 740, 865]):
                logger.debug("Calculating MCI")

                band_665 = rrs_bands[665]
                band_705 = rrs_bands[705]
                band_740 = rrs_bands[740]
                band_865 = rrs_bands[865]

                valid_mask = (
                    (~np.isnan(band_665)) &
                    (~np.isnan(band_705)) &
                    (~np.isnan(band_740)) &
                    (~np.isnan(band_865))
                )

                if np.any(valid_mask):
                    # MCI calculation (simplified for S2 bands)
                    slope = (740 - 665) / (865 - 665)
                    mci_values[valid_mask] = (
                        band_705[valid_mask] - band_665[valid_mask] -
                        slope * (band_865[valid_mask] - band_665[valid_mask])
                    )

                    # MCI bloom threshold
                    mci_bloom = (mci_values > 0.004).astype(np.float32)
                    results['mci_bloom'] = mci_bloom
                    results['mci_values'] = mci_values

                    # Add to probability calculation
                    hab_probability += mci_bloom * 0.3
                    algorithms_applied.append("MCI")

                    logger.debug(f"MCI calculated for {np.sum(valid_mask)} pixels")

            # Calculate combined HAB probability and risk levels
            if algorithms_applied:
                # Normalize probability to 0-1 range
                hab_probability = np.clip(hab_probability, 0, 1)
                results['hab_probability'] = hab_probability

                # Create risk level classification
                hab_risk = np.zeros(shape, dtype=np.uint8)
                hab_risk[hab_probability > 0.7] = 3  # High risk
                hab_risk[(hab_probability > 0.4) & (hab_probability <= 0.7)] = 2  # Medium risk
                hab_risk[(hab_probability > 0.2) & (hab_probability <= 0.4)] = 1  # Low risk
                # hab_risk = 0 for probability <= 0.2 (no risk)

                results['hab_risk_level'] = hab_risk

                # Additional detection flags
                if 'ndci_bloom' in results:
                    # Cyanobacteria-like bloom detection (based on NDCI only - FLH removed)
                    results['cyanobacteria_bloom'] = results['ndci_bloom'].copy()

                # Biomass alert levels
                high_biomass = (hab_probability > 0.6).astype(np.float32)
                extreme_biomass = (hab_probability > 0.8).astype(np.float32)

                results['high_biomass_alert'] = high_biomass
                results['extreme_biomass_alert'] = extreme_biomass

                # Calculate statistics - single summary line
                total_pixels = np.sum(~np.isnan(hab_probability))
                if total_pixels > 0:
                    high_risk_pixels = np.sum(hab_risk == 3)
                    medium_risk_pixels = np.sum(hab_risk == 2)
                    low_risk_pixels = np.sum(hab_risk == 1)
                    logger.debug(f"HAB: {', '.join(algorithms_applied)} - High:{high_risk_pixels} Med:{medium_risk_pixels} Low:{low_risk_pixels}")
            else:
                logger.warning("No suitable spectral bands found for HAB detection")
                # Return empty arrays to avoid missing data issues
                results['hab_probability'] = np.zeros(shape, dtype=np.float32)
                results['hab_risk_level'] = np.zeros(shape, dtype=np.uint8)

            return results

        except Exception as e:
            logger.error(f"Error in HAB detection: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def calculate_trophic_state(self, chlorophyll: np.ndarray,
                                secchi_depth: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate Trophic State Index (TSI) using Carlson (1977) methodology.

        Reference:
            Carlson, R.E. (1977). A trophic state index for lakes.
            Limnology and Oceanography, 22(2), 361-369.

        TSI Scale Interpretation:
            < 40: Oligotrophic (clear, low productivity)
            40-50: Mesotrophic (moderate productivity)
            50-70: Eutrophic (high productivity, algae-rich)
            > 70: Hypereutrophic (very high productivity)

        Args:
            chlorophyll: Chlorophyll-a concentration in ug/L (mg/m3)
            secchi_depth: Optional Secchi disk depth in meters

        Returns:
            Dictionary with TSI values and trophic classification
        """
        try:
            logger.info("Calculating Trophic State Index (Carlson 1977)")

            results = {}
            shape = chlorophyll.shape

            # TSI from chlorophyll-a: TSI(CHL) = 9.81 x ln(CHL) + 30.6
            # Valid for CHL > 0
            valid_chl = (chlorophyll > 0) & (~np.isnan(chlorophyll))

            tsi_chl = np.full(shape, np.nan, dtype=np.float32)
            if np.any(valid_chl):
                tsi_chl[valid_chl] = 9.81 * np.log(chlorophyll[valid_chl]) + 30.6
                results['tsi_chlorophyll'] = tsi_chl
                logger.info(f"TSI(CHL) calculated for {np.sum(valid_chl)} pixels")

            # TSI from Secchi depth if available: TSI(SD) = 60 - 14.41 x ln(SD)
            if secchi_depth is not None:
                valid_sd = (secchi_depth > 0) & (~np.isnan(secchi_depth))
                tsi_sd = np.full(shape, np.nan, dtype=np.float32)
                if np.any(valid_sd):
                    tsi_sd[valid_sd] = 60.0 - 14.41 * np.log(secchi_depth[valid_sd])
                    results['tsi_secchi'] = tsi_sd
                    logger.info(f"TSI(SD) calculated for {np.sum(valid_sd)} pixels")

            # Use TSI(CHL) as primary TSI
            tsi_primary = tsi_chl.copy()
            results['tsi'] = tsi_primary

            # Trophic state classification
            trophic_class = np.full(shape, 0, dtype=np.uint8)  # 0 = No data
            trophic_class[tsi_primary < 40] = 1   # Oligotrophic
            trophic_class[(tsi_primary >= 40) & (tsi_primary < 50)] = 2  # Mesotrophic
            trophic_class[(tsi_primary >= 50) & (tsi_primary < 70)] = 3  # Eutrophic
            trophic_class[tsi_primary >= 70] = 4  # Hypereutrophic

            results['trophic_class'] = trophic_class

            # Calculate statistics
            valid_tsi = ~np.isnan(tsi_primary)
            if np.any(valid_tsi):
                oligotrophic = np.sum(trophic_class == 1)
                mesotrophic = np.sum(trophic_class == 2)
                eutrophic = np.sum(trophic_class == 3)
                hypereutrophic = np.sum(trophic_class == 4)
                total = np.sum(valid_tsi)

                logger.info(f"Trophic state classification completed:")
                logger.info(f"  Mean TSI: {np.nanmean(tsi_primary):.1f}")
                logger.info(f"  Oligotrophic (<40): {oligotrophic} pixels ({100*oligotrophic/total:.1f}%)")
                logger.info(f"  Mesotrophic (40-50): {mesotrophic} pixels ({100*mesotrophic/total:.1f}%)")
                logger.info(f"  Eutrophic (50-70): {eutrophic} pixels ({100*eutrophic/total:.1f}%)")
                logger.info(f"  Hypereutrophic (>70): {hypereutrophic} pixels ({100*hypereutrophic/total:.1f}%)")

            return results

        except Exception as e:
            logger.error(f"Error calculating trophic state: {e}")
            import traceback
            traceback.print_exc()
            return {}


