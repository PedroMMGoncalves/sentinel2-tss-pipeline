"""
SNAP TSM/CHL Calculator.

Calculate TSM and CHL from SNAP C2RCC IOP outputs using official SNAP formulas.

Reference:
    C2RCC (Case 2 Regional Coast Colour) processor in SNAP
    TSM formula: TSM = 1.06 * (bpart + bwit)^0.942
    CHL formula: CHL = apig^1.04 * 21.0
"""

import os
import logging
import traceback
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from ..utils.raster_io import RasterIO

logger = logging.getLogger('sentinel2_tss_pipeline')


@dataclass
class ProcessingResult:
    """Result of a processing operation"""
    success: bool
    output_path: str
    stats: dict
    error_message: Optional[str] = None
    intermediate_paths: Optional[dict] = field(default_factory=dict)


class TSMCHLCalculator:
    """Calculate TSM and CHL from SNAP C2RCC IOP outputs using official SNAP formulas."""

    def __init__(self, tsm_fac: float = 1.06, tsm_exp: float = 0.942,
                 chl_fac: float = 21.0, chl_exp: float = 1.04):
        self.tsm_fac = tsm_fac
        self.tsm_exp = tsm_exp
        self.chl_fac = chl_fac
        self.chl_exp = chl_exp

        logger.debug(f"SNAP TSM/CHL Calculator: TSM={tsm_fac}*(bpart+bwit)^{tsm_exp}, CHL=apig^{chl_exp}*{chl_fac}")

    def calculate_snap_tsm_chl(self, c2rcc_path: str) -> Dict[str, ProcessingResult]:
        """Calculate TSM and CHL from SNAP IOPs using official formulas"""
        try:
            logger.debug("Calculating SNAP TSM/CHL from IOP products")

            # Determine data folder
            if c2rcc_path.endswith('.dim'):
                data_folder = c2rcc_path.replace('.dim', '.data')
            else:
                data_folder = f"{c2rcc_path}.data"

            if not os.path.exists(data_folder):
                return {'error': ProcessingResult(False, "", None, f"Data folder not found: {data_folder}")}

            # Load required IOPs with robust error handling
            iop_files = {
                'apig': os.path.join(data_folder, 'iop_apig.img'),     # For CHL
                'bpart': os.path.join(data_folder, 'iop_bpart.img'),   # For TSM
                'bwit': os.path.join(data_folder, 'iop_bwit.img')      # For TSM
            }

            # Check and load available IOPs
            available_iops = {}

            for iop_name, iop_path in iop_files.items():
                if os.path.exists(iop_path) and os.path.getsize(iop_path) > 1024:
                    try:
                        data, metadata = RasterIO.read_raster(iop_path)
                        available_iops[iop_name] = {'data': data, 'metadata': metadata}
                        logger.debug(f"Loaded {iop_name}: {data.shape}, mean={np.nanmean(data):.4f}")
                    except Exception as e:
                        logger.error(f"Error loading {iop_name}: {e}")
                else:
                    logger.debug(f"Missing or empty: {iop_name}")

            results = {}

            # Calculate CHL from apig using SNAP formula
            if 'apig' in available_iops:
                logger.debug("Calculating CHL from iop_apig")

                apig_data = available_iops['apig']['data']
                metadata = available_iops['apig']['metadata']

                # Handle edge cases properly
                valid_mask = (apig_data > 0) & (~np.isnan(apig_data)) & (~np.isinf(apig_data))

                # Initialize result array
                chl_concentration = np.full_like(apig_data, np.nan, dtype=np.float32)

                if np.any(valid_mask):
                    # Apply SNAP CHL formula: CHL = apig^CHLexp * CHLfac
                    try:
                        valid_apig = apig_data[valid_mask]
                        chl_values = np.power(valid_apig, self.chl_exp) * self.chl_fac

                        # Check for invalid results
                        valid_chl_mask = (~np.isnan(chl_values)) & (~np.isinf(chl_values)) & (chl_values >= 0)

                        # Only assign valid CHL values
                        if np.any(valid_chl_mask):
                            # Create a temporary mask for the original array
                            temp_mask = valid_mask.copy()
                            temp_mask[valid_mask] = valid_chl_mask
                            chl_concentration[temp_mask] = chl_values[valid_chl_mask]

                            logger.debug(f"CHL: {np.sum(temp_mask)} valid pixels")
                        else:
                            logger.warning("No valid CHL values after calculation")

                    except Exception as calc_error:
                        logger.error(f"Error in CHL calculation: {calc_error}")
                        chl_concentration = np.full_like(apig_data, np.nan, dtype=np.float32)

                # Save CHL to .data folder (standard SNAP BEAM-DIMAP practice)
                output_path = os.path.join(data_folder, 'conc_chl.img')
                success = RasterIO.write_raster(
                    chl_concentration, output_path, metadata,
                    f"SNAP Chlorophyll concentration (mg/m3) - CHL = apig^{self.chl_exp} * {self.chl_fac}",
                    nodata=-9999
                )

                if success:
                    stats = RasterIO.calculate_statistics(chl_concentration)
                    logger.debug(f"CHL saved: {stats['coverage_percent']:.1f}% coverage, mean={stats['mean']:.3f} mg/m3")
                    results['snap_chl'] = ProcessingResult(True, output_path, stats, None)
                else:
                    results['snap_chl'] = ProcessingResult(False, output_path, None, "Failed to save CHL")
            else:
                logger.error("Cannot calculate CHL: iop_apig.img not available")
                results['snap_chl'] = ProcessingResult(False, "", None, "Missing iop_apig for CHL calculation")

            # Calculate TSM from bpart + bwit (btot approximation)
            if 'bpart' in available_iops and 'bwit' in available_iops:
                logger.debug("Calculating TSM from bpart + bwit")

                bpart_data = available_iops['bpart']['data']
                bwit_data = available_iops['bwit']['data']
                metadata = available_iops['bpart']['metadata']

                # Approximate btot as bpart + bwit (since iop_btot.img is missing)
                btot_approx = bpart_data + bwit_data

                # Handle edge cases properly
                valid_mask = (btot_approx > 0) & (~np.isnan(btot_approx)) & (~np.isinf(btot_approx))

                # Initialize result array
                tsm_concentration = np.full_like(btot_approx, np.nan, dtype=np.float32)

                if np.any(valid_mask):
                    # Apply SNAP TSM formula: TSM = TSMfac * btot^TSMexp
                    try:
                        valid_btot = btot_approx[valid_mask]
                        tsm_values = self.tsm_fac * np.power(valid_btot, self.tsm_exp)

                        # Check for invalid results
                        valid_tsm_mask = (~np.isnan(tsm_values)) & (~np.isinf(tsm_values)) & (tsm_values >= 0)

                        # Only assign valid TSM values
                        if np.any(valid_tsm_mask):
                            temp_mask = valid_mask.copy()
                            temp_mask[valid_mask] = valid_tsm_mask
                            tsm_concentration[temp_mask] = tsm_values[valid_tsm_mask]

                            logger.debug(f"TSM: {np.sum(temp_mask)} valid pixels")
                        else:
                            logger.warning("No valid TSM values after calculation")

                    except Exception as calc_error:
                        logger.error(f"Error in TSM calculation: {calc_error}")
                        tsm_concentration = np.full_like(btot_approx, np.nan, dtype=np.float32)

                # Save TSM concentration
                output_path = os.path.join(data_folder, 'conc_tsm.img')
                success = RasterIO.write_raster(
                    tsm_concentration, output_path, metadata,
                    f"SNAP TSM concentration (g/m3) - TSM = {self.tsm_fac} * (bpart + bwit)^{self.tsm_exp}",
                    nodata=-9999
                )

                if success:
                    stats = RasterIO.calculate_statistics(tsm_concentration)
                    logger.debug(f"TSM saved: {stats['coverage_percent']:.1f}% coverage, mean={stats['mean']:.3f} g/m3")
                    results['snap_tsm'] = ProcessingResult(True, output_path, stats, None)
                else:
                    results['snap_tsm'] = ProcessingResult(False, output_path, None, "Failed to save TSM")
            else:
                logger.error("Cannot calculate TSM: missing bpart or bwit")
                results['snap_tsm'] = ProcessingResult(False, "", None, "Missing bpart/bwit for TSM calculation")

            return results

        except Exception as e:
            error_msg = f"Error calculating SNAP TSM/CHL: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def calculate_uncertainties(self, c2rcc_path: str) -> Dict[str, ProcessingResult]:
        """Calculate uncertainties for TSM and CHL products"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')

            # Check if uncertainty files exist
            unc_tsm_path = os.path.join(data_folder, 'unc_tsm.img')
            unc_chl_path = os.path.join(data_folder, 'unc_chl.img')

            tsm_path = os.path.join(data_folder, 'conc_tsm.img')
            chl_path = os.path.join(data_folder, 'conc_chl.img')

            results = {}

            # Calculate TSM uncertainty (typically 15-30% of TSM value)
            if os.path.exists(tsm_path) and not (os.path.exists(unc_tsm_path) and os.path.getsize(unc_tsm_path) > 1024):
                logger.info("Calculating TSM uncertainties...")

                tsm_data, tsm_meta = RasterIO.read_raster(tsm_path)

                # Simplified uncertainty model (no in-situ data available for full error propagation)
                unc_tsm_data = np.where(
                    tsm_data > 0,
                    tsm_data * 0.20 + 0.1,  # 20% relative + 0.1 g/m3 absolute
                    np.nan
                )

                success = RasterIO.write_raster(unc_tsm_data, unc_tsm_path, tsm_meta)
                if success:
                    logger.info("TSM uncertainties calculated and saved")

                    # Calculate statistics
                    valid_pixels = np.sum(unc_tsm_data > 0)
                    mean_uncertainty = np.mean(unc_tsm_data[unc_tsm_data > 0]) if valid_pixels > 0 else 0

                    stats = {
                        'coverage_percent': (valid_pixels / unc_tsm_data.size) * 100,
                        'mean': mean_uncertainty,
                        'valid_pixels': int(valid_pixels),
                        'file_size_mb': os.path.getsize(unc_tsm_path) / (1024 * 1024)
                    }

                    results['unc_tsm'] = ProcessingResult(True, unc_tsm_path, stats, None)
                else:
                    logger.error("Failed to save TSM uncertainties")
                    results['unc_tsm'] = ProcessingResult(False, "", None, "Failed to save TSM uncertainties")

            # Calculate CHL uncertainty (typically 25-40% of CHL value)
            if os.path.exists(chl_path) and not (os.path.exists(unc_chl_path) and os.path.getsize(unc_chl_path) > 1024):
                logger.info("Calculating CHL uncertainties...")

                chl_data, chl_meta = RasterIO.read_raster(chl_path)

                # Simple uncertainty model: 30% of CHL value + 0.05 mg/m3 base uncertainty
                unc_chl_data = np.where(
                    chl_data > 0,
                    chl_data * 0.30 + 0.05,  # 30% relative + 0.05 mg/m3 absolute
                    np.nan
                )

                success = RasterIO.write_raster(unc_chl_data, unc_chl_path, chl_meta)
                if success:
                    logger.info("CHL uncertainties calculated and saved")

                    # Calculate statistics
                    valid_pixels = np.sum(unc_chl_data > 0)
                    mean_uncertainty = np.mean(unc_chl_data[unc_chl_data > 0]) if valid_pixels > 0 else 0

                    stats = {
                        'coverage_percent': (valid_pixels / unc_chl_data.size) * 100,
                        'mean': mean_uncertainty,
                        'valid_pixels': int(valid_pixels),
                        'file_size_mb': os.path.getsize(unc_chl_path) / (1024 * 1024)
                    }

                    results['unc_chl'] = ProcessingResult(True, unc_chl_path, stats, None)
                else:
                    logger.error("Failed to save CHL uncertainties")
                    results['unc_chl'] = ProcessingResult(False, "", None, "Failed to save CHL uncertainties")

            return results

        except Exception as e:
            logger.error(f"Error calculating uncertainties: {e}")
            return {
                'unc_error': ProcessingResult(False, "", None, f"Error calculating uncertainties: {e}")
            }
