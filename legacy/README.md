Unified S2 Processing & TSS Estimation Pipeline

A comprehensive Python pipeline for processing Sentinel-2 data and estimating Total Suspended Solids (TSS) in aquatic environments. This professional-grade tool combines atmospheric correction using C2RCC with advanced TSS estimation methodologies.
🌟 Features
Complete Processing Pipeline

L1C → C2RCC Processing: Atmospheric correction with ECMWF integration
Automatic SNAP Products: TSM, CHL concentrations with uncertainty maps
Advanced TSS Estimation: Optional Jiang et al. 2023 methodology
Quality Assessment: Comprehensive validation and statistics

Professional GUI Interface

Tabbed Configuration: Organized parameter management
Real-time Monitoring: System resources and processing status
Progress Tracking: ETA calculations and detailed statistics
Configuration Management: Save/load processing settings

Scientific Accuracy

ECMWF Integration: Real-time atmospheric data for superior accuracy
Uncertainty Quantification: Statistical uncertainty maps included
Water Type Classification: Adaptive processing for different water bodies
Validation Tools: Quality control and comparison statistics

Production-Ready Features

Batch Processing: Handle multiple products efficiently
Memory Management: Automatic cleanup and monitoring
Error Recovery: Graceful handling of processing failures
Comprehensive Logging: Detailed processing logs and statistics

📋 Requirements
System Requirements

Operating System: Windows 10/11, Linux, or macOS
RAM: Minimum 8GB (16GB+ recommended)
Storage: 50GB+ free space for processing
CPU: Multi-core processor recommended

Software Dependencies

Python: 3.7 or higher
SNAP: Version 9.0 or higher
Anaconda: For environment management
Spyder: For development and execution

🚀 Installation
Step 1: Install SNAP
Download and install SNAP from the official ESA website:

Download: https://step.esa.int/main/download/snap-download/
Installation Guide: https://step.esa.int/main/download/snap-download/

Important: Make sure to install SNAP in the default location and note the installation path.
Step 2: Create Anaconda Environment
Open Anaconda Prompt (Windows) or Terminal (Linux/macOS) and create a new environment:

# Create new environment named 'snap-c2rcc'
conda create -n snap-c2rcc python=3.9

# Activate the environment
conda activate snap-c2rcc
Step 3: Install Dependencies

Install required packages in the activated environment:

# Install core scientific packages
conda install numpy pandas scipy matplotlib

# Install geospatial packages
conda install -c conda-forge gdal rasterio pyproj

# Install GUI and system packages
conda install -c conda-forge psutil tqdm

# Install additional packages via pip
pip install tkinter-tooltip

# Verify GDAL installation
python -c "from osgeo import gdal; print('GDAL version:', gdal.__version__)"
Step 4: Install Spyder

# Install Spyder IDE
conda install spyder

# Install additional Spyder plugins (optional)
conda install spyder-kernels
Step 5: Configure SNAP Environment
Set the SNAP_HOME environment variable:
Windows (in Anaconda Prompt):

# Set SNAP_HOME (adjust path if different)
set SNAP_HOME=C:\Program Files\esa-snap

# Make it permanent (optional)
setx SNAP_HOME "C:\Program Files\esa-snap"
Linux/macOS (in Terminal):

# Add to your shell profile (.bashrc, .zshrc, etc.)
export SNAP_HOME=/usr/local/snap

# Or set temporarily
export SNAP_HOME=/path/to/your/snap/installation
Step 6: Verify Installation
Test your installation:

# Activate environment
conda activate snap-c2rcc

# Test Python imports
python -c "import numpy, gdal, psutil; print('All imports successful!')"

# Test SNAP GPT
gpt -h
🔧 SNAP Diagnostics & Troubleshooting
Running SNAP Diagnostics
If you encounter SNAP-related issues, use the diagnostic script:

# Activate environment
conda activate snap-c2rcc

# Run SNAP diagnostics
python snap_diagnostics.py
The diagnostic script will:

✅ Verify SNAP installation
✅ Check GPT executable
✅ Test available operators
✅ Detect plugin conflicts
✅ Suggest specific fixes

Common SNAP Issues & Solutions
🚨 Plugin Conflicts (Most Common)
Error: NoClassDefFoundError in GPT output
Solution:

Start SNAP Desktop: "C:\Program Files\esa-snap\bin\snap64.exe"
Go to: Tools → Plugins → Installed
Disable/Uninstall: ASTER, EOMTBX, or other problematic plugins
Restart SNAP Desktop

🚨 SNAP_HOME Issues
Error: SNAP_HOME not set or GPT not found
Solution:
bash# Windows
set SNAP_HOME=C:\Program Files\esa-snap
setx SNAP_HOME "C:\Program Files\esa-snap"

# Linux/macOS
export SNAP_HOME=/usr/local/snap
🚨 Java Configuration
Error: Java-related errors in GPT
Solution:

Check Java version: java -version (should be Java 8 or 11)
Reset SNAP user directory:

Windows: rmdir /s "%USERPROFILE%\.snap"
Linux/macOS: rm -rf ~/.snap



🚨 Memory Issues
Error: OutOfMemoryError
Solution:

Reduce memory limit in processing settings
Close other applications
Process fewer products simultaneously

Advanced Diagnostics
bash# Run comprehensive SNAP test
python snap_diagnostics.py

# Test with sample data
python snap_diagnostics.py --test-processing /path/to/sample.zip /path/to/output
📁 Project Setup
Clone Repository
bashgit clone https://github.com/yourusername/unified-s2-tss-pipeline.git
cd unified-s2-tss-pipeline
Directory Structure
unified-s2-tss-pipeline/
├── unified_s2_tss_pipeline.py    # Main pipeline script
├── snap_diagnostics.py           # SNAP diagnostic tool
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── examples/                     # Example configurations
│   ├── config_complete.json      # Complete pipeline config
│   ├── config_s2_only.json       # S2 processing only
│   └── config_tss_only.json      # TSS processing only
├── docs/                         # Documentation
│   ├── user_guide.md             # Detailed user guide
│   ├── api_reference.md          # API documentation
│   └── troubleshooting.md        # Common issues and solutions
└── tests/                        # Unit tests
    ├── test_processors.py        # Test processing components
    └── test_configurations.py    # Test configuration handling

🎯 Quick Start
Method 1: Using Spyder IDE (Recommended)

Launch Spyder:
bashconda activate snap-c2rcc
spyder

Open the Script:

In Spyder: File → Open → unified_s2_tss_pipeline.py


Run the Script:

Click the Run button (▶️) or press F5
The GUI will launch automatically



Method 2: Command Line
bash# Activate environment
conda activate snap-c2rcc

# Run with GUI
python unified_s2_tss_pipeline.py

# Run with command line arguments
python unified_s2_tss_pipeline.py -i /path/to/input -o /path/to/output --mode complete_pipeline
🔧 Configuration
Processing Modes
ModeInputOutputDescriptionComplete PipelineL1C ProductsC2RCC + TSSFull processing chainS2 Processing OnlyL1C ProductsC2RCCAtmospheric correction onlyTSS Processing OnlyC2RCC ProductsTSSTSS estimation only
Essential Settings
Input Data

L1C Products: .zip or .SAFE format
C2RCC Products: .dim format (for TSS-only mode)

C2RCC Configuration

ECMWF: Enabled by default (recommended)
Water Properties: Salinity, temperature
Output Products: TSM, CHL, uncertainties

TSS Options

SNAP TSM/CHL: Automatically generated
Jiang Methodology: Optional advanced TSS estimation

📊 Usage Examples
Example 1: Complete Pipeline
python# Through GUI
1. Select "Complete Pipeline" mode
2. Set input folder containing L1C products
3. Set output folder
4. Configure C2RCC parameters
5. Click "Start Processing"

# Through command line
python unified_s2_tss_pipeline.py \
    -i /path/to/L1C_products \
    -o /path/to/results \
    --mode complete_pipeline \
    --enable-jiang
Example 2: Coastal Water Processing
python# Optimized for coastal environments
1. Mode: Complete Pipeline
2. Water Properties:
   - Salinity: 35.0 PSU
   - Temperature: 15.0°C
3. ECMWF: Enabled
4. Resolution: 10m
5. Enable Jiang TSS: Yes
Example 3: Inland Water Processing
python# Optimized for lakes and rivers
1. Mode: Complete Pipeline
2. Water Properties:
   - Salinity: 0.1 PSU
   - Temperature: 20.0°C
3. ECMWF: Enabled
4. Resolution: 20m
5. Enable Jiang TSS: Yes
📈 Output Products
Directory Structure
output_folder/
├── Geometric_Products/           # Resampled S2 products
├── C2RCC_Products/              # Atmospheric correction results
│   ├── *.dim                    # BEAM-DIMAP format
│   └── *.data/                  # Associated data folders
│       ├── conc_tsm.img         # TSM concentration
│       ├── conc_chl.img         # CHL concentration
│       ├── unc_tsm.img          # TSM uncertainty
│       ├── unc_chl.img          # CHL uncertainty
│       └── rhow_*.img           # Water-leaving reflectance
├── TSS_Products/                # TSS estimation results
│   └── *_Jiang_TSS.tif         # Jiang methodology TSS
└── Logs/                        # Processing logs
    └── unified_s2_tss_*.log     # Detailed processing logs
Product Descriptions
ProductDescriptionUnitsTypical Rangeconc_tsm.imgSNAP TSM concentrationg/m³0.1 - 100conc_chl.imgSNAP CHL concentrationmg/m³0.1 - 50unc_tsm.imgTSM uncertaintyg/m³0.01 - 10unc_chl.imgCHL uncertaintymg/m³0.01 - 5Jiang_TSS.tifJiang TSS estimationg/m³0.1 - 1000
🛠️ Troubleshooting
Step 1: Run Diagnostics
Always start with the diagnostic script:
bashpython snap_diagnostics.py
This will identify 90% of common issues automatically.
Step 2: Check Common Issues
SNAP Not Found
ERROR: SNAP_HOME not set and SNAP installation not found!
Solution:

Verify SNAP installation
Set SNAP_HOME environment variable
Restart terminal/Spyder

Plugin Conflicts
ERROR: NoClassDefFoundError in GPT output
Solution:

Run diagnostic script: python snap_diagnostics.py
Follow plugin cleanup instructions
Restart SNAP Desktop

Memory Issues
ERROR: High memory usage detected
Solution:

Reduce memory limit in settings
Process fewer products simultaneously
Close other applications

Processing Failures
ERROR: GPT processing failed
Solution:

Run diagnostic script first
Check input product integrity
Verify sufficient disk space
Check SNAP GPT configuration

Step 3: Advanced Troubleshooting
Environment Issues
python# Test imports
python -c "import numpy; print('NumPy OK')"
python -c "from osgeo import gdal; print('GDAL OK')"
python -c "import psutil; print('psutil OK')"
GDAL Configuration
bash# Check GDAL installation
gdalinfo --version

# Test GDAL Python bindings
python -c "from osgeo import gdal; print(gdal.VersionInfo())"
SNAP Reset (Last Resort)
bash# Windows
rmdir /s "%USERPROFILE%\.snap"

# Linux/macOS
rm -rf ~/.snap
📚 Documentation
User Guides

User Guide: Comprehensive usage instructions
API Reference: Technical API documentation
Troubleshooting: Common issues and solutions

Scientific Background

C2RCC Algorithm: Brockmann et al., 2016
Jiang TSS Methodology: Jiang et al., 2023
SNAP Documentation: Official SNAP Docs

🤝 Contributing
Development Setup
bash# Clone repository
git clone https://github.com/yourusername/unified-s2-tss-pipeline.git
cd unified-s2-tss-pipeline

# Create development environment
conda create -n snap-c2rcc-dev python=3.9
conda activate snap-c2rcc-dev

# Install development dependencies
conda install numpy pandas scipy matplotlib gdal rasterio pyproj psutil tqdm spyder
pip install pytest black flake8

# Run tests
pytest tests/
Code Style

PEP 8: Follow Python style guidelines
Type Hints: Use type annotations
Documentation: Comprehensive docstrings
Testing: Unit tests for new features

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

ESA SNAP Team: For the excellent SNAP software
C2RCC Developers: For the atmospheric correction algorithm
Jiang et al.: For the advanced TSS methodology
Python Community: For the amazing scientific libraries

📞 Support
Getting Help

Issues: GitHub Issues
Discussions: GitHub Discussions
Email: your.email@example.com

Before Reporting Issues

Run diagnostics: python snap_diagnostics.py
Check logs: Review processing log files
Search existing issues: Check if problem already reported

Citation
If you use this pipeline in your research, please cite:
@software{sentinel2-tss-pipeline,
  title={S2 Processing & TSS Estimation Pipeline},
  author={Pedro Gonçalves},
  year={2025},
  url={https://github.com/PedroMMGoncalves/sentinel2-tss-pipeline}
}

🚀 Version History
v1.0.0 (Current)

✅ Complete S2 processing pipeline
✅ C2RCC atmospheric correction
✅ Automatic SNAP TSM/CHL generation
✅ Optional Jiang TSS methodology
✅ Professional GUI interface
✅ Comprehensive error handling
✅ Real-time monitoring
✅ SNAP diagnostics tool

Planned Features

🔄 Parallel processing optimization
🔄 Quality assessment tools
🔄 Additional TSS methodologies
🔄 Export to cloud-optimized formats
🔄 Integration with cloud platforms
