# Sentinel-2 TSS Pipeline

A comprehensive Python pipeline for processing Sentinel-2 imagery and estimating Total Suspended Solids (TSS) in aquatic environments using the Jiang et al. (2021) semi-analytical methodology.

**Version:** 2.0.0
**Author:** Pedro Goncalves
**License:** Research use

## Features

### Complete Processing Pipeline
- **L1C to C2RCC Processing**: Atmospheric correction with ECMWF integration
- **Automatic SNAP Products**: TSM, CHL concentrations with uncertainty maps
- **Advanced TSS Estimation**: Jiang et al. (2021) semi-analytical methodology
- **Water Type Classification**: Adaptive processing for 4 water turbidity classes
- **Quality Assessment**: Comprehensive validation and statistics

### Comprehensive Output Products
- **18 RGB Composite Variants**: Natural color, false color, and water-specific visualizations
- **17+ Spectral Indices**: NDWI, NDTI, NDCI, FLH, MCI, TSI, and more
- **Water Quality Parameters**: TSM, CHL, TSS, absorption, backscattering
- **Trophic State Index (TSI)**: Carlson (1977) methodology

### Professional GUI Interface
- **Tabbed Configuration**: Organized parameter management
- **Real-time Monitoring**: System resources and processing status
- **Progress Tracking**: ETA calculations and detailed statistics
- **Configuration Management**: Save/load processing settings

### Production-Ready Features
- **Batch Processing**: Handle multiple products efficiently
- **Memory Management**: Automatic cleanup and monitoring (optimized for 256GB RAM)
- **Error Recovery**: Graceful handling of processing failures
- **Comprehensive Logging**: Detailed processing logs and statistics

## Scientific References

### Primary TSS Algorithm
```
Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
"Remotely Estimating Total Suspended Solids Concentration in Clear to
Extremely Turbid Waters Using a Novel Semi-Analytical Method."
Remote Sensing of Environment, 258, 112386.
DOI: https://doi.org/10.1016/j.rse.2021.112386
```

### Water Type Classification
| Type | Condition | Reference Band | TSS Range |
|------|-----------|----------------|-----------|
| Type I (Clear) | Rrs(490) > Rrs(560) | 560nm | < 10 g/m³ |
| Type II (Moderate) | Rrs(490) > Rrs(620) | 665nm | 10-50 g/m³ |
| Type III (Turbid) | Default | 740nm | 50-200 g/m³ |
| Type IV (Extreme) | Rrs(740) > Rrs(490) AND Rrs(740) > 0.010 | 865nm | > 200 g/m³ |

### Additional Scientific Basis
- **C2RCC**: Brockmann et al. (2016) - Atmospheric correction
- **QAA v6.0**: Lee et al. - Quasi-Analytical Algorithm
- **TSI**: Carlson (1977) - Trophic State Index
- **NDCI**: Mishra & Mishra (2012) - Harmful Algal Bloom detection
- **MCI**: Gower et al. (2005) - Maximum Chlorophyll Index
- **RDI**: Stumpf et al. (2003) - Relative Depth Index (bathymetric change detection)
- **Pure Water Constants**: Pope & Fry (1997)

## Requirements

### System Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 16GB (256GB recommended for large datasets)
- **Storage**: 50GB+ free space for processing
- **CPU**: Multi-core processor recommended

### Software Dependencies

#### Required
- Python 3.8+
- numpy
- GDAL (osgeo)
- tkinter (GUI)
- psutil (system monitoring)

#### Required for Processing
- SNAP GPT (ESA SNAP Graph Processing Tool) v9.0+
- SNAP C2RCC module

#### Optional
- geopandas (geometry loading)
- fiona (shapefile support)
- shapely (geometry operations)
- tqdm (progress bars)

## Installation

### Step 1: Install SNAP

Download and install SNAP from the official ESA website:
- **Download**: https://step.esa.int/main/download/snap-download/

Important: Install SNAP in the default location and note the installation path.

### Step 2: Create Conda Environment

```bash
# Create new environment
conda create -n snap-c2rcc python=3.9

# Activate the environment
conda activate snap-c2rcc
```

### Step 3: Install Dependencies

```bash
# Install core scientific packages
conda install numpy pandas scipy matplotlib

# Install geospatial packages
conda install -c conda-forge gdal rasterio pyproj

# Install GUI and system packages
conda install -c conda-forge psutil tqdm

# Verify GDAL installation
python -c "from osgeo import gdal; print('GDAL version:', gdal.__version__)"
```

### Step 4: Configure SNAP Environment

**Windows (in Anaconda Prompt):**
```bash
set SNAP_HOME=C:\Program Files\esa-snap
setx SNAP_HOME "C:\Program Files\esa-snap"
```

**Linux/macOS:**
```bash
export SNAP_HOME=/usr/local/snap
```

### Step 5: Clone Repository

```bash
git clone https://github.com/PedroMMGoncalves/sentinel2-tss-pipeline.git
cd sentinel2-tss-pipeline
```

### Step 6: Verify Installation

```bash
# Test package imports
python -c "from sentinel2_tss_pipeline import UnifiedS2TSSProcessor, S2Processor, JiangTSSProcessor; print('All imports OK')"

# Test CLI
python -m sentinel2_tss_pipeline --help
```

## Project Structure

```
sentinel2-tss-pipeline/
├── README.md                        # This file
│
├── sentinel2_tss_pipeline/          # Main modular package (v2.0)
│   ├── __init__.py                  # Package exports
│   ├── __main__.py                  # Entry point for python -m
│   ├── main.py                      # CLI and GUI entry points
│   │
│   ├── config/                      # Configuration dataclasses
│   │   ├── enums.py                 # ProcessingMode, ProductType
│   │   ├── s2_config.py             # ResamplingConfig, SubsetConfig, C2RCCConfig
│   │   ├── jiang_config.py          # JiangTSSConfig
│   │   ├── water_quality_config.py  # WaterQualityConfig
│   │   ├── marine_config.py         # MarineVisualizationConfig
│   │   └── processing_config.py     # ProcessingConfig
│   │
│   ├── utils/                       # Utility modules
│   │   ├── logging_utils.py         # ColoredFormatter, setup_enhanced_logging
│   │   ├── math_utils.py            # SafeMathNumPy
│   │   ├── memory_manager.py        # MemoryManager
│   │   ├── raster_io.py             # RasterIO (GDAL wrapper)
│   │   └── product_detector.py      # ProductDetector, SystemMonitor
│   │
│   ├── processors/                  # Processing modules
│   │   ├── snap_calculator.py       # SNAPTSMCHLCalculator
│   │   ├── jiang_processor.py       # JiangTSSProcessor
│   │   ├── water_quality_processor.py # WaterQualityProcessor
│   │   ├── marine_viz.py            # S2MarineVisualizationProcessor
│   │   └── s2_processor.py          # S2Processor
│   │
│   ├── core/                        # Core processing
│   │   └── unified_processor.py     # UnifiedS2TSSProcessor
│   │
│   └── gui/                         # GUI module
│       └── unified_gui.py           # UnifiedS2TSSGUI
│
└── legacy/                          # Original monolithic implementation
    ├── sentinel2_tss_pipeline.py    # Original single-file version
    └── snap_diagnostics.py          # SNAP diagnostic tool
```

## Usage

### Method 1: GUI (Recommended)

```bash
# Activate environment
conda activate snap-c2rcc

# Launch GUI
python -m sentinel2_tss_pipeline
```

### Method 2: Command Line Interface

```bash
# Basic usage
python -m sentinel2_tss_pipeline -i /path/to/L1C -o /path/to/results

# Complete pipeline with all options
python -m sentinel2_tss_pipeline \
    -i /path/to/L1C_products \
    -o /path/to/results \
    --mode complete_pipeline \
    --enable-jiang

# Show help
python -m sentinel2_tss_pipeline --help
```

### Method 3: Python API

```python
from sentinel2_tss_pipeline import UnifiedS2TSSProcessor
from sentinel2_tss_pipeline.config import ProcessingConfig, ProcessingMode

# Create configuration
config = ProcessingConfig(
    input_folder="/path/to/L1C",
    output_folder="/path/to/results",
    processing_mode=ProcessingMode.COMPLETE_PIPELINE,
    enable_jiang_tss=True
)

# Run processing
processor = UnifiedS2TSSProcessor(config)
processor.process()
```

## Processing Modes

| Mode | Input | Output | Description |
|------|-------|--------|-------------|
| Complete Pipeline | L1C .SAFE/.zip | C2RCC + TSS + Visualizations | Full processing chain |
| S2 Processing Only | L1C .SAFE/.zip | C2RCC products | Atmospheric correction only |
| TSS Processing Only | C2RCC .dim | TSS + Visualizations | TSS estimation only |

## Output Products

### Directory Structure

```
output_folder/
├── Geometric_Products/              # Resampled S2 products
├── C2RCC_Products/                  # Atmospheric correction results
│   ├── *.dim                        # BEAM-DIMAP format
│   └── *.data/                      # Associated data folders
│       ├── rrs_B1-B8A.img           # Remote sensing reflectance
│       ├── rhow_*.img               # Water-leaving reflectance
│       ├── iop_*.img                # Inherent optical properties
│       ├── conc_tsm.img             # TSM concentration
│       ├── conc_chl.img             # CHL concentration
│       ├── unc_tsm.img              # TSM uncertainty
│       └── unc_chl.img              # CHL uncertainty
├── TSS_Products/                    # TSS estimation results
│   ├── *_Jiang_TSS.tif              # Jiang TSS estimation
│   ├── *_Absorption.tif             # Absorption coefficient
│   ├── *_Backscattering.tif         # Backscattering coefficient
│   └── *_WaterTypes.tif             # Water type classification
├── RGB_Composites/                  # Visualization products
│   ├── *_NaturalColor.tif           # True color composite
│   ├── *_FalseColor.tif             # False color composite
│   └── *_Water_*.tif                # 18 water-specific variants
├── Spectral_Indices/                # Water quality indices
│   ├── *_NDWI.tif                   # Normalized Difference Water Index
│   ├── *_NDTI.tif                   # Normalized Difference Turbidity Index
│   ├── *_NDCI.tif                   # Normalized Difference Chlorophyll Index
│   ├── *_FLH.tif                    # Fluorescence Line Height
│   ├── *_MCI.tif                    # Maximum Chlorophyll Index
│   ├── *_TSI.tif                    # Trophic State Index
│   └── *_SecchiDepth.tif            # Secchi Disk Depth
└── Logs/                            # Processing logs
    └── unified_s2_tss_*.log         # Detailed processing logs
```

### Product Descriptions

| Product | Description | Units | Typical Range |
|---------|-------------|-------|---------------|
| conc_tsm | SNAP TSM concentration | g/m³ | 0.1 - 100 |
| conc_chl | SNAP CHL concentration | mg/m³ | 0.1 - 50 |
| Jiang_TSS | Jiang TSS estimation | g/m³ | 0.1 - 1000 |
| WaterTypes | Water type classification | Class 1-4 | 1 - 4 |
| TSI | Trophic State Index | Index | 0 - 100 |
| NDWI | Water index | Index | -1 to 1 |
| SecchiDepth | Water transparency | meters | 0.1 - 30 |

### Trophic State Index (TSI) Classification

| TSI Value | Trophic State | Description |
|-----------|---------------|-------------|
| < 40 | Oligotrophic | Clear water, low productivity |
| 40-50 | Mesotrophic | Moderate productivity |
| 50-70 | Eutrophic | High productivity, potential algal blooms |
| > 70 | Hypereutrophic | Very high productivity, frequent blooms |

## SNAP Diagnostics & Troubleshooting

### Running SNAP Diagnostics

```bash
# Run SNAP diagnostics (from legacy folder)
python legacy/snap_diagnostics.py
```

The diagnostic script will:
- Verify SNAP installation
- Check GPT executable
- Test available operators
- Detect plugin conflicts
- Suggest specific fixes

### Common Issues & Solutions

#### SNAP Not Found
```
ERROR: SNAP_HOME not set
```
**Solution:**
```bash
# Windows
setx SNAP_HOME "C:\Program Files\esa-snap"

# Linux/macOS
export SNAP_HOME=/usr/local/snap
```

#### Plugin Conflicts
```
ERROR: NoClassDefFoundError in GPT output
```
**Solution:**
1. Start SNAP Desktop
2. Go to: Tools > Plugins > Installed
3. Disable/Uninstall: ASTER, EOMTBX, or other problematic plugins
4. Restart SNAP Desktop

#### Memory Issues
```
ERROR: OutOfMemoryError
```
**Solution:**
- Reduce memory limit in processing settings
- Close other applications
- Process fewer products simultaneously

#### Java Configuration
**Solution:**
1. Check Java version: `java -version` (should be Java 8 or 11)
2. Reset SNAP user directory:
   - Windows: `rmdir /s "%USERPROFILE%\.snap"`
   - Linux/macOS: `rm -rf ~/.snap`

## Configuration Examples

### Coastal Water Processing

```python
# Optimized for coastal environments
config = ProcessingConfig(
    input_folder="/path/to/L1C",
    output_folder="/path/to/results",
    processing_mode=ProcessingMode.COMPLETE_PIPELINE,
    salinity=35.0,      # PSU
    temperature=15.0,   # Celsius
    enable_jiang_tss=True,
    target_resolution=10  # meters
)
```

### Inland Water Processing

```python
# Optimized for lakes and rivers
config = ProcessingConfig(
    input_folder="/path/to/L1C",
    output_folder="/path/to/results",
    processing_mode=ProcessingMode.COMPLETE_PIPELINE,
    salinity=0.1,       # PSU (freshwater)
    temperature=20.0,   # Celsius
    enable_jiang_tss=True,
    target_resolution=20  # meters
)
```

## Data Storage Formats

| Format | Extension | Usage |
|--------|-----------|-------|
| BEAM-DIMAP | .dim/.data | SNAP intermediate products |
| ENVI | .img | IOPs, TSM/CHL from SNAP |
| GeoTIFF | .tif | Final products (LZW compressed) |
| PNG | .png | Quick preview images |

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sentinel2-tss-pipeline,
  title={Sentinel-2 TSS Pipeline},
  author={Goncalves, Pedro},
  year={2025},
  version={2.0.0},
  url={https://github.com/PedroMMGoncalves/sentinel2-tss-pipeline}
}
```

And the primary scientific reference:

```bibtex
@article{jiang2021tss,
  title={Remotely estimating total suspended solids concentration in clear
         to extremely turbid waters using a novel semi-analytical method},
  author={Jiang, D. and Matsushita, B. and Pahlevan, N. and others},
  journal={Remote Sensing of Environment},
  volume={258},
  pages={112386},
  year={2021},
  doi={10.1016/j.rse.2021.112386}
}
```

## Version History

### v2.0.0 (Current)
- Complete modular refactoring into package structure
- 18 RGB composite visualizations
- 17+ spectral indices (NDWI, NDTI, NDCI, FLH, MCI, TSI, etc.)
- Water type classification (4 types)
- Trophic State Index (Carlson 1977)
- Marine visualization processor
- Improved memory management
- CLI and Python API support

### v1.0.0 (Legacy)
- Monolithic single-file implementation
- Basic C2RCC processing
- SNAP TSM/CHL generation
- Jiang TSS methodology
- GUI interface

## Support

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/PedroMMGoncalves/sentinel2-tss-pipeline/issues)

### Before Reporting Issues
1. Run diagnostics: `python legacy/snap_diagnostics.py`
2. Check logs: Review processing log files
3. Search existing issues: Check if problem already reported

## Acknowledgments

- **ESA SNAP Team**: For the excellent SNAP software
- **C2RCC Developers**: For the atmospheric correction algorithm
- **Jiang et al.**: For the advanced TSS methodology
- **Python Community**: For the amazing scientific libraries

## License

This project is licensed for research use. See the LICENSE file for details.
