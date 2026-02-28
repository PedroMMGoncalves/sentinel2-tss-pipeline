"""
Logging utilities for Sentinel-2 TSS Pipeline

Provides colored console output, file logging, and GRACE-style
structured logging with per-step CPU/RAM monitoring.
"""

import os
import time
import logging
import threading
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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


# ── GRACE-Style Structured Logging ──────────────────────────────────────

BOX_WIDTH = 76


class StepTracker:
    """
    GRACE-style structured logging with per-step CPU/RAM monitoring.

    Usage:
        tracker = StepTracker(logger)
        tracker.banner("SENTINEL-2 TSS PIPELINE v2.0", "Mode: Complete Pipeline")
        tracker.box_start("S2B_MSIL1C_20240315T112119_T29SNC")
        with tracker.step("[1/2] Resampling + C2RCC"):
            ... do work ...
        tracker.box_line("Scene complete: 44 products (7.1 min)")
        tracker.box_end()
        tracker.summary_banner({"Processed": "2 scenes", ...})
    """

    def __init__(self, logger):
        self.logger = logger
        self._batch_cpu_samples = []
        self._batch_ram_samples = []
        self._scene_cpu_samples = []
        self._scene_ram_samples = []

    def banner(self, *lines):
        """Double-line banner for top-level section headers."""
        self.logger.info("")
        self.logger.info("\u2550" * BOX_WIDTH)
        for line in lines:
            self.logger.info(f"  {line}")
        self.logger.info("\u2550" * BOX_WIDTH)

    def summary_banner(self, kv_pairs: dict, title: str = "COMPLETE"):
        """Final summary banner with key-value pairs."""
        self.logger.info("")
        self.logger.info("\u2550" * BOX_WIDTH)
        self.logger.info(f"  {title} \u2014 Sentinel-2 TSS Pipeline")
        self.logger.info("\u2550" * BOX_WIDTH)
        for key, value in kv_pairs.items():
            self.logger.info(f"  {key + ':':<14s} {value}")
        self.logger.info("\u2550" * BOX_WIDTH)

    def log_step(self, message):
        """Top-level step message outside any box."""
        self.logger.info("")
        self.logger.info(message)

    def config_box(self, kv_pairs: dict):
        """Configuration box with key-value pairs."""
        border = "\u2500" * (BOX_WIDTH - 17)
        self.logger.info("")
        self.logger.info(f"\u250C\u2500 Configuration {border}")
        for key, value in kv_pairs.items():
            self.logger.info(f"\u2502  {key + ':':<14s} {value}")
        self.logger.info(f"\u2514" + "\u2500" * BOX_WIDTH)

    def box_start(self, title):
        """Open a scene processing box."""
        padding = BOX_WIDTH - len(title) - 4
        border = "\u2500" * max(padding, 2)
        self.logger.info("")
        self.logger.info(f"\u250C\u2500 {title} {border}")
        self.logger.info("\u2502")
        self._scene_cpu_samples = []
        self._scene_ram_samples = []

    def box_line(self, text=""):
        """Print a line inside the current box."""
        if text:
            self.logger.info(f"\u2502  {text}")
        else:
            self.logger.info("\u2502")

    def box_end(self):
        """Close the current box."""
        self.logger.info("\u2502")
        self.logger.info(f"\u2514" + "\u2500" * BOX_WIDTH)

    def format_resources(self, cpu_samples, ram_samples):
        """Format CPU/RAM stats as a string."""
        if not cpu_samples or not ram_samples:
            return None
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        peak_cpu = max(cpu_samples)
        avg_ram = sum(ram_samples) / len(ram_samples)
        peak_ram = max(ram_samples)
        return (f"CPU: avg {avg_cpu:.0f}% peak {peak_cpu:.0f}%  "
                f"RAM: avg {avg_ram:.1f} GB peak {peak_ram:.1f} GB")

    def step(self, label: str):
        """Context manager that times a step and samples CPU/RAM."""
        return _StepContext(self, label)


class _StepContext:
    """Context manager for a single processing step with resource tracking."""

    SAMPLE_INTERVAL = 2.0

    def __init__(self, tracker: StepTracker, label: str):
        self.tracker = tracker
        self.label = label
        self._cpu_samples = []
        self._ram_samples = []
        self._sampling = False
        self._thread = None
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()
        if _HAS_PSUTIL:
            psutil.cpu_percent(interval=0)  # prime the first measurement
            self._sampling = True
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sampling = False
        elapsed = time.time() - self._start_time

        if self._thread:
            self._thread.join(timeout=3.0)

        # Final sample
        if _HAS_PSUTIL:
            try:
                self._cpu_samples.append(psutil.cpu_percent(interval=0))
                self._ram_samples.append(psutil.virtual_memory().used / (1024 ** 3))
            except Exception:
                pass

        # Format elapsed time
        if elapsed >= 60:
            elapsed_str = f"{elapsed / 60:.1f} min"
        else:
            elapsed_str = f"{elapsed:.1f} sec"

        status = f"FAILED ({elapsed_str})" if exc_type else f"done ({elapsed_str})"

        # Build dotted line
        total_width = 58
        dots_needed = total_width - len(self.label) - len(status) - 2
        dot_str = " " + "." * max(dots_needed, 3) + " "
        self.tracker.box_line(f"{self.label}{dot_str}{status}")

        # Resource stats line
        if self._cpu_samples and self._ram_samples:
            # Indent to align under label text after bracket
            if "]" in self.label:
                indent = " " * (self.label.index("]") + 2)
            else:
                indent = "      "
            res = self.tracker.format_resources(self._cpu_samples, self._ram_samples)
            self.tracker.box_line(f"{indent}{res}")

            # Accumulate for scene and batch summaries
            self.tracker._scene_cpu_samples.extend(self._cpu_samples)
            self.tracker._scene_ram_samples.extend(self._ram_samples)
            self.tracker._batch_cpu_samples.extend(self._cpu_samples)
            self.tracker._batch_ram_samples.extend(self._ram_samples)

        return False  # don't suppress exceptions

    def _sample_loop(self):
        """Background sampling of CPU and RAM."""
        while self._sampling:
            try:
                self._cpu_samples.append(psutil.cpu_percent(interval=0))
                self._ram_samples.append(psutil.virtual_memory().used / (1024 ** 3))
            except Exception:
                pass
            time.sleep(self.SAMPLE_INTERVAL)
