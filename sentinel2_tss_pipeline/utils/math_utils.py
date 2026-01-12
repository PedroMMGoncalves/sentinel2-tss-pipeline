"""
Safe mathematical operations for NumPy arrays.

Provides overflow-protected operations for division, square root,
logarithm, and power calculations.
"""

import numpy as np
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


class SafeMathNumPy:
    """Safe mathematical operations for NumPy arrays"""

    @staticmethod
    def safe_divide(numerator, denominator, default_value=0.0, min_denominator=1e-10):
        """
        Safely divide arrays with protection against division by zero.

        Args:
            numerator: Numerator array or scalar
            denominator: Denominator array or scalar
            default_value: Value to use when division is not possible
            min_denominator: Minimum absolute value for denominator

        Returns:
            Result array with safe division
        """
        if not isinstance(numerator, np.ndarray):
            numerator = np.array(numerator, dtype=np.float32)
        if not isinstance(denominator, np.ndarray):
            denominator = np.array(denominator, dtype=np.float32)

        result = np.full_like(numerator, default_value, dtype=np.float32)

        if isinstance(denominator, (int, float)):
            if abs(denominator) >= min_denominator:
                result = numerator / denominator
        else:
            valid_mask = np.abs(denominator) >= min_denominator
            result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        return result

    @staticmethod
    def safe_sqrt(value, min_value=0.0, default_value=0.0):
        """
        Safely calculate square root.

        Args:
            value: Input array or scalar
            min_value: Minimum value allowed for sqrt
            default_value: Value to use for invalid inputs

        Returns:
            Square root result array
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)

        result = np.full_like(value, default_value, dtype=np.float32)
        valid_mask = value >= min_value
        result[valid_mask] = np.sqrt(value[valid_mask])

        return result

    @staticmethod
    def safe_log(value, base=10, min_value=1e-10, default_value=-999):
        """
        Safely calculate logarithm.

        Args:
            value: Input array or scalar
            base: Logarithm base (default: 10)
            min_value: Minimum value for valid log
            default_value: Value to use for invalid inputs

        Returns:
            Logarithm result array
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)

        result = np.full_like(value, default_value, dtype=np.float32)

        if isinstance(value, (int, float)):
            if value > min_value:
                result = np.log(value) / np.log(base)
        else:
            valid_mask = value > min_value
            result[valid_mask] = np.log(value[valid_mask]) / np.log(base)

        return result

    @staticmethod
    def safe_power(base, exponent, min_base=1e-10, max_exponent=100, default_value=0.0):
        """
        Safely calculate power operations.

        Args:
            base: Base array or scalar
            exponent: Exponent array or scalar
            min_base: Minimum base value
            max_exponent: Maximum absolute exponent
            default_value: Value to use for invalid inputs

        Returns:
            Power result array
        """
        if not isinstance(base, np.ndarray):
            base = np.array(base, dtype=np.float32)
        if not isinstance(exponent, np.ndarray):
            exponent = np.array(exponent, dtype=np.float32)

        result = np.full_like(base, default_value, dtype=np.float32)

        # Avoid extreme values that could cause overflow
        valid_mask = (base >= min_base) & (np.abs(exponent) <= max_exponent)

        if np.any(valid_mask):
            try:
                result[valid_mask] = np.power(base[valid_mask], exponent[valid_mask])
            except (OverflowError, RuntimeWarning):
                logger.warning("Power calculation overflow detected, using default values")

        return result
