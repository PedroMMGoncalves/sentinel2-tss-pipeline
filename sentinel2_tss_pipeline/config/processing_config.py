"""
Processing Configuration

Part of the sentinel2_tss_pipeline package.
"""

from dataclasses import dataclass

from .enums import ProcessingMode
from .s2_config import ResamplingConfig, SubsetConfig, C2RCCConfig
from .jiang_config import JiangTSSConfig


@dataclass
class ProcessingConfig:
    """Complete processing configuration"""
    processing_mode: ProcessingMode
    input_folder: str
    output_folder: str
    resampling_config: ResamplingConfig
    subset_config: SubsetConfig
    c2rcc_config: C2RCCConfig
    jiang_config: JiangTSSConfig
    skip_existing: bool = True
    test_mode: bool = False
    memory_limit_gb: int = 8
    thread_count: int = 4
