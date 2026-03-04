@echo off
REM NOTE: This script is for Windows only.
REM For Linux/macOS, use: conda env create -f environment.yml
REM Then: conda activate ocean_rs
REM Then: python setup_environment.py --verify
REM ============================================================
REM OceanRS Environment Installer
REM
REM Creates a conda environment with all dependencies for the
REM OceanRS ocean remote sensing toolkit.
REM
REM Usage: Open Anaconda Prompt, navigate to this folder, and run:
REM   install_environment.bat
REM
REM Note: This script installs packages step-by-step instead of
REM using "conda env create -f environment.yml" to avoid pip
REM conflicts on machines with ArcGIS Pro or multiple Python
REM installations.
REM ============================================================

echo.
echo ============================================================
echo  OceanRS Environment Installer
echo ============================================================
echo.

REM --- Check if conda is available ---
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda not found.
    echo Please run this script from Anaconda Prompt.
    echo.
    pause
    exit /b 1
)

REM --- Check if ocean_rs already exists ---
conda env list | findstr /C:"ocean_rs" >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: ocean_rs environment already exists.
    echo.
    set /p REPLY="Delete and recreate from scratch? (y/n): "
    if /i "%REPLY%"=="y" (
        echo.
        echo Removing existing environment...
        conda deactivate 2>nul
        conda env remove -n ocean_rs -y
        echo Done.
    ) else (
        echo.
        echo Aborted. Existing environment kept.
        pause
        exit /b 0
    )
)

REM --- Step 1: Create environment with Python 3.12 ---
echo.
echo [1/6] Creating ocean_rs environment with Python 3.12...
echo       (Python 3.12 is required — newer versions break GDAL)
echo.
conda create -n ocean_rs python=3.12.* -c conda-forge -y
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to create environment.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

REM --- Step 2: Install core packages ---
echo.
echo [2/6] Installing core packages (numpy, GDAL, psutil)...
echo.
conda install -n ocean_rs -c conda-forge numpy gdal psutil -y
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install core packages.
    echo If antivirus blocked .pyd files, whitelist the conda envs folder.
    pause
    exit /b 1
)

REM --- Step 3: Install geometry packages ---
echo.
echo [3/6] Installing geometry packages (shapely, fiona, geopandas)...
echo.
conda install -n ocean_rs -c conda-forge shapely fiona geopandas requests -y
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install geometry packages.
    pause
    exit /b 1
)

REM --- Step 4: Install Spyder IDE ---
echo.
echo [4/6] Installing Spyder IDE...
echo.
conda install -n ocean_rs -c conda-forge spyder -y
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Spyder installation failed.
    echo You can install it later with: conda install -n ocean_rs -c conda-forge spyder
)

REM --- Step 5: Upgrade pip inside the environment ---
echo.
echo [5/6] Upgrading pip...
echo.
conda run -n ocean_rs python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: pip upgrade failed. Continuing...
)

REM --- Step 6: Install pip-only packages ---
echo.
echo [6/6] Installing pip packages (sv-ttk, asf_search, python-dotenv, tkintermapview)...
echo.
conda run -n ocean_rs python -m pip install sv-ttk asf_search python-dotenv tkintermapview
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Some pip packages may have failed.
    echo You can install them later with:
    echo   conda activate ocean_rs
    echo   python -m pip install sv-ttk asf_search python-dotenv tkintermapview
)

REM --- Verify ---
echo.
echo ============================================================
echo  Installation complete! Running verification...
echo ============================================================
echo.
conda run -n ocean_rs python "%~dp0setup_environment.py"

echo.
echo ============================================================
echo  How to use OceanRS:
echo.
echo  From Anaconda Prompt:
echo    conda activate ocean_rs
echo    spyder
echo.
echo  Or from Anaconda Navigator:
echo    Select "ocean_rs" environment, then Launch Spyder
echo.
echo  In Spyder:
echo    Open run_gui.py and press F5 (Optical)
echo    Open run_sar_gui.py and press F5 (SAR)
echo.
echo  IMPORTANT: In Spyder, set the Python interpreter to ocean_rs:
echo    Tools ^> Preferences ^> Python interpreter ^>
echo    "Use the following interpreter" ^>
echo    Browse to: [conda envs path]\ocean_rs\python.exe
echo ============================================================
pause
