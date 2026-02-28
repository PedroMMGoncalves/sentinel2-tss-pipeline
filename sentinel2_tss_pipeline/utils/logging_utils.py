"""
Logging utilities for Sentinel-2 TSS Pipeline

Provides colored console output and file logging capabilities.
"""

import os
import logging
from datetime import datetime


_current_log_file = None  # Track current log file to avoid duplicate messages


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced formatting"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"

        # Enhanced format with more info
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%H:%M:%S'
        )
        return formatter.format(record)


def setup_enhanced_logging(log_level=logging.INFO, output_folder: str = None):
    """
    Setup enhanced logging with proper file placement.

    Args:
        log_level: Logging level (default: INFO)
        output_folder: Directory for log files

    Returns:
        Tuple of (logger, log_file_path)
    """
    logger = logging.getLogger('sentinel2_tss_pipeline')
    logger.setLevel(log_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Determine log file location
    if output_folder:
        log_dir = os.path.join(output_folder, "Logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'unified_s2_tss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = f'unified_s2_tss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    global _current_log_file
    if _current_log_file is None:
        logger.info(f"Logging configured - File: {log_file}")
    else:
        logger.debug(f"Logging redirected to: {log_file}")
    _current_log_file = log_file

    return logger, log_file


def get_default_logger():
    """
    Create logger with smart default location.

    Prevents logs from cluttering the code directory by using a default
    results folder.

    Returns:
        Configured logger instance
    """
    try:
        # Create a default results folder to avoid cluttering code directory
        default_output = os.path.join(os.getcwd(), "S2_TSS_Results")
        os.makedirs(default_output, exist_ok=True)

        # Setup logging in the default location
        logger, log_file = setup_enhanced_logging(log_level=logging.INFO, output_folder=default_output)

        # Print to console so user knows where logs are going
        print(f"Default logging location: {log_file}")
        print(f"Logs will be saved to: {default_output}/Logs/")

        return logger

    except Exception as e:
        # Fallback to current directory if anything fails
        print(f"Warning: Could not create default log folder: {e}")
        print("Using current directory for logs")
        return setup_enhanced_logging()[0]
