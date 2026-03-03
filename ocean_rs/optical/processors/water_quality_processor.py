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
- Lee, Z. et al. (2005). Diffuse attenuation coefficient of downwelling irradiance. JGR 110.
- Kirk, J.T.O. (2011). Light and photosynthesis in aquatic ecosystems.
- Tyler, J.E. (1968). The Secchi disc.
- Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index.
- Gower, J. et al. (2005). Maximum Chlorophyll Index.
- Carlson, R.E. (1977). A trophic state index for lakes.
"""

import logging
from typing import Dict, Optional

import numpy as np

from ..config import WaterQualityConfig

logger = logging.getLogger('ocean_rs')


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
        - Lee, Z. et al. (2005). Diffuse attenuation coefficient of downwelling irradiance.
          J. Geophys. Res., 110, C02017.
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

            # M-3: NaN filtering — only compute on valid (finite) input pixels
            valid_mask = np.isfinite(absorption) & np.isfinite(backscattering)
            logger.info(f"Water clarity: {np.sum(valid_mask)} valid pixels of {valid_mask.size}")

            # WARNING: For Type III/IV water, 'absorption' is pure-water absorption only
            # (aw at 740/865nm). Particulate absorption is not estimated for these water types.
            # Kd, Secchi, and euphotic depth may underestimate turbidity for Type III/IV pixels.
            logger.debug("  Note: Absorption for Types III/IV is pure-water only — clarity products may underestimate turbidity")

            # Initialize all output arrays as NaN
            kd = np.full_like(absorption, np.nan)
            secchi_depth = np.full_like(absorption, np.nan)
            clarity_index = np.full_like(absorption, np.nan)
            euphotic_depth = np.full_like(absorption, np.nan)
            beam_attenuation = np.full_like(absorption, np.nan)
            relative_turbidity_index = np.full_like(absorption, np.nan)

            if np.any(valid_mask):
                a = absorption[valid_mask]
                bb = backscattering[valid_mask]

                # Lee et al. (2005) diffuse attenuation coefficient
                # Kd = (1 + 0.005 * theta_s) * [a + 4.18 * (1 - 0.52 * exp(-10.8 * a)) * bb]
                # NOTE: Kd is computed from water-type-specific a and bb (reference wavelengths vary:
                # 560nm for Type I, 665nm for Type II, 740/865nm for Types III/IV).
                # This is NOT a standardized Kd(490). Secchi and euphotic depth inherit this
                # wavelength dependence. Compare values within the same water type only.
                theta_s = solar_zenith
                kd_valid = (1 + 0.005 * theta_s) * (
                    a + 4.18 * (1 - 0.52 * np.exp(-10.8 * a)) * bb
                )

                # M-2: Clip Kd BEFORE computing derived products
                kd_valid = np.clip(kd_valid, 0, 20)  # m^-1, max for extremely turbid

                # M2-18: Conditional computation for Secchi/euphotic depth
                # Minimum meaningful Kd (0.001 m^-1) avoids meaningless intermediates
                # (e.g., 1.7e8 m) before clipping. Pixels with Kd <= 0.001 get NaN.
                safe_kd = kd_valid > 0.001

                # C=1.7 (Tyler 1968 in-situ disc convention). Satellite-appropriate value ~1.0 (Doron 2011)
                secchi_valid = np.full_like(kd_valid, np.nan)
                secchi_valid[safe_kd] = 1.7 / kd_valid[safe_kd]

                # Custom proxy indices, not peer-reviewed formulas
                clarity_valid = 1 / (1 + kd_valid)

                # Euphotic depth (1% light level)
                euphotic_valid = np.full_like(kd_valid, np.nan)
                euphotic_valid[safe_kd] = 4.605 / kd_valid[safe_kd]  # ln(100) / kd

                # Beam attenuation: c = a + b (total scattering)
                # bb_ratio=0.0183 (Petzold 1972 coastal average, range 0.005-0.03 by water type)
                BACKSCATTER_RATIO = 0.0183
                total_scattering = bb / BACKSCATTER_RATIO
                beam_att_valid = a + total_scattering

                # M2-1: Warn when beam attenuation exceeds clip limit (before clipping)
                n_saturated = int(np.sum(beam_att_valid > 50.0))
                if n_saturated > 0:
                    logger.warning(f"  Beam attenuation: {n_saturated} pixels saturated at clip limit (50 m^-1)")

                # Physical range clamping for derived products
                secchi_valid = np.clip(secchi_valid, 0, 100)       # meters, max for clearest ocean
                euphotic_valid = np.clip(euphotic_valid, 0, 200)   # meters
                beam_att_valid = np.clip(beam_att_valid, 0, 50)    # m^-1

                # Custom proxy indices, not peer-reviewed formulas
                # Note: This is NOT calibrated NTU - use for relative comparison only
                # Saturates at bb=0.05 m^-1 for coastal water dynamic range
                rel_turb_valid = np.clip(bb * 20, 0, 1)

                # Assign valid results back
                kd[valid_mask] = kd_valid
                secchi_depth[valid_mask] = secchi_valid
                clarity_index[valid_mask] = clarity_valid
                euphotic_depth[valid_mask] = euphotic_valid
                beam_attenuation[valid_mask] = beam_att_valid
                relative_turbidity_index[valid_mask] = rel_turb_valid

            # M-15: proxy_ prefix for custom indices that are not peer-reviewed
            results = {
                'diffuse_attenuation': kd,
                'secchi_depth': secchi_depth,
                'proxy_clarity_index': clarity_index,
                'euphotic_depth': euphotic_depth,
                'beam_attenuation': beam_attenuation,
                'proxy_relative_turbidity_index': relative_turbidity_index
            }

            # Calculate statistics
            valid_pixels = np.sum(~np.isnan(kd))
            if valid_pixels > 0:
                logger.debug(f"Water clarity: {valid_pixels} pixels, Secchi={np.nanmean(secchi_depth):.1f}m")

            return results

        except Exception as e:
            logger.warning(f"Water clarity calculation skipped due to error: {e}")
            return {}

    def detect_harmful_algal_blooms(self, chlorophyll: Optional[np.ndarray],
                                phycocyanin: Optional[np.ndarray],
                                rrs_bands: Dict[int, np.ndarray],
                                water_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        HAB detection using Sentinel-2 spectral bands (NDCI, MCI).

        Note: FLH removed - S2 lacks 681nm band for true fluorescence detection.

        References:
        - Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index.
        - Gower, J. et al. (2005). Maximum Chlorophyll Index.

        Args:
            chlorophyll: Optional chlorophyll array
            phycocyanin: Optional phycocyanin array
            rrs_bands: Dict mapping wavelength (nm) to Rrs arrays
            water_mask: Optional boolean mask (True=water). Used for statistics.

        Returns:
            Dictionary with HAB detection products.
            hab_score is a heuristic bloom likelihood score, not a calibrated probability.
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

            # H-4: Renamed from hab_probability to hab_score
            # Heuristic bloom likelihood score, not a calibrated probability
            hab_score = np.full(shape, np.nan, dtype=np.float32)
            # H2-3: Per-pixel algorithm counting to avoid NaN propagation
            # When one algorithm returns NaN for a pixel, only the other
            # algorithm's score should be used (not dropped to NaN)
            hab_score_accum = np.zeros(shape, dtype=np.float32)
            n_valid = np.zeros(shape, dtype=np.float32)
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
                    (band_705 > -0.001) &   # Allow slight negatives from atm. correction
                    (band_665 > -0.001)
                )

                denominator = band_705 + band_665
                valid_mask = valid_mask & (denominator > 1e-8)

                if np.any(valid_mask):
                    ndci_values[valid_mask] = ((band_705[valid_mask] - band_665[valid_mask]) /
                                            denominator[valid_mask])

                    # NDCI bloom threshold (Mishra & Mishra, 2012)
                    ndci_bloom = np.where(np.isfinite(ndci_values),
                                          (ndci_values > 0.05).astype(np.float32),
                                          np.nan)
                    results['ndci_bloom'] = ndci_bloom
                    results['ndci_values'] = ndci_values

                    # H-4: Scale each algorithm to full [0, 1] range
                    ndci_contribution = np.clip(
                        (ndci_values - 0.05) / (0.3 - 0.05), 0, 1
                    )
                    # H2-3: Per-pixel accumulation — skip NaN pixels
                    valid_alg = np.isfinite(ndci_contribution)
                    hab_score_accum = np.where(valid_alg, hab_score_accum + ndci_contribution, hab_score_accum)
                    n_valid = np.where(valid_alg, n_valid + 1, n_valid)
                    algorithms_applied.append("NDCI")

                    logger.debug(f"NDCI calculated for {np.sum(valid_mask)} pixels")

            # FLH (Fluorescence Line Height) removed: S2 lacks the 681nm band required for
            # true fluorescence peak detection. The 705nm band captures red-edge reflectance,
            # not the chlorophyll fluorescence emission at ~685nm.

            # Method 2: Maximum Chlorophyll Index (MCI) - Gower et al. (2005)
            # L-30: MCI only uses 665, 705, 740nm — 865nm is not needed
            if all(band in rrs_bands for band in [665, 705, 740]):
                logger.debug("Calculating MCI")

                band_665 = rrs_bands[665]
                band_705 = rrs_bands[705]
                band_740 = rrs_bands[740]

                valid_mask = (
                    (~np.isnan(band_665)) &
                    (~np.isnan(band_705)) &
                    (~np.isnan(band_740))
                )

                if np.any(valid_mask):
                    # MCI: height of 705nm peak above 665-740nm baseline
                    # Adapted for Sentinel-2 from Binding et al. (2013)
                    slope = (705 - 665) / (740 - 665)  # = 0.533
                    mci_values[valid_mask] = (
                        band_705[valid_mask] - band_665[valid_mask] -
                        slope * (band_740[valid_mask] - band_665[valid_mask])
                    )

                    # MCI bloom threshold
                    mci_bloom = np.where(np.isfinite(mci_values),
                                         (mci_values > 0.004).astype(np.float32),
                                         np.nan)
                    results['mci_bloom'] = mci_bloom
                    results['mci_values'] = mci_values

                    # H-4: Scale each algorithm to full [0, 1] range
                    mci_contribution = np.clip(
                        (mci_values - 0.004) / (0.02 - 0.004), 0, 1
                    )
                    # H2-3: Per-pixel accumulation — skip NaN pixels
                    valid_alg = np.isfinite(mci_contribution)
                    hab_score_accum = np.where(valid_alg, hab_score_accum + mci_contribution, hab_score_accum)
                    n_valid = np.where(valid_alg, n_valid + 1, n_valid)
                    algorithms_applied.append("MCI")

                    logger.debug(f"MCI calculated for {np.sum(valid_mask)} pixels")

            # Calculate combined HAB score and risk levels
            if algorithms_applied:
                # H2-3: Per-pixel average — only pixels with at least one valid
                # algorithm get a score; others remain NaN
                hab_score = np.where(n_valid > 0, hab_score_accum / n_valid, np.nan)
                hab_score = np.clip(hab_score, 0, 1)
                results['hab_score'] = hab_score

                # Create risk level classification (255 = nodata)
                hab_risk = np.full(shape, 255, dtype=np.uint8)  # 255 = nodata
                valid_score = np.isfinite(hab_score)
                hab_risk[valid_score & (hab_score > 0.7)] = 3   # High risk
                hab_risk[valid_score & (hab_score > 0.4) & (hab_score <= 0.7)] = 2  # Moderate risk
                hab_risk[valid_score & (hab_score > 0.2) & (hab_score <= 0.4)] = 1  # Low risk
                hab_risk[valid_score & (hab_score <= 0.2)] = 0   # No risk

                results['hab_risk_level'] = hab_risk

                # Additional detection flags
                if 'ndci_bloom' in results:
                    # General bloom detection (NDCI-based, not cyanobacteria-specific)
                    results['potential_bloom'] = results['ndci_bloom'].copy()

                # Biomass alert levels
                high_biomass = np.where(np.isfinite(hab_score),
                                        (hab_score > 0.6).astype(np.float32), np.nan)
                extreme_biomass = np.where(np.isfinite(hab_score),
                                           (hab_score > 0.8).astype(np.float32), np.nan)

                results['high_biomass_alert'] = high_biomass
                results['extreme_biomass_alert'] = extreme_biomass

                # L-8: Use water/valid mask to exclude land pixels from statistics
                if water_mask is not None:
                    stats_mask = water_mask & (~np.isnan(hab_score))
                else:
                    stats_mask = ~np.isnan(hab_score)

                total_pixels = np.sum(stats_mask)
                if total_pixels > 0:
                    high_risk_pixels = np.sum(hab_risk[stats_mask] == 3)
                    medium_risk_pixels = np.sum(hab_risk[stats_mask] == 2)
                    low_risk_pixels = np.sum(hab_risk[stats_mask] == 1)
                    logger.debug(f"HAB: {', '.join(algorithms_applied)} - High:{high_risk_pixels} Med:{medium_risk_pixels} Low:{low_risk_pixels}")
            else:
                logger.warning("No suitable spectral bands found for HAB detection")
                # Return NaN arrays (not zeros) to distinguish no-data from no-bloom
                results['hab_score'] = np.full(shape, np.nan, dtype=np.float32)
                results['hab_risk_level'] = np.zeros(shape, dtype=np.uint8)

            return results

        except Exception as e:
            logger.warning(f"HAB detection skipped due to error: {e}")
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
                tsi_chl[valid_chl] = np.clip(
                    9.81 * np.log(chlorophyll[valid_chl]) + 30.6, 0, 100
                )
                results['tsi_chlorophyll'] = tsi_chl
                logger.info(f"TSI(CHL) calculated for {np.sum(valid_chl)} pixels")

            # TSI from Secchi depth if available: TSI(SD) = 60 - 14.41 x ln(SD)
            if secchi_depth is not None:
                valid_sd = (secchi_depth > 0) & (~np.isnan(secchi_depth))
                tsi_sd = np.full(shape, np.nan, dtype=np.float32)
                if np.any(valid_sd):
                    tsi_sd[valid_sd] = np.clip(
                        60.0 - 14.41 * np.log(secchi_depth[valid_sd]), 0, 100
                    )
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
                valid_tc = trophic_class[valid_tsi]
                oligotrophic = int(np.sum(valid_tc == 1))
                mesotrophic = int(np.sum(valid_tc == 2))
                eutrophic = int(np.sum(valid_tc == 3))
                hypereutrophic = int(np.sum(valid_tc == 4))
                total = len(valid_tc)

                logger.info(f"Trophic state classification completed:")
                logger.info(f"  Mean TSI: {np.nanmean(tsi_primary):.1f}")
                logger.info(f"  Oligotrophic (<40): {oligotrophic} pixels ({100*oligotrophic/total:.1f}%)")
                logger.info(f"  Mesotrophic (40-50): {mesotrophic} pixels ({100*mesotrophic/total:.1f}%)")
                logger.info(f"  Eutrophic (50-70): {eutrophic} pixels ({100*eutrophic/total:.1f}%)")
                logger.info(f"  Hypereutrophic (>70): {hypereutrophic} pixels ({100*hypereutrophic/total:.1f}%)")

            return results

        except Exception as e:
            logger.warning(f"Trophic state calculation skipped due to error: {e}")
            return {}


