"""
OceanRS Environment Setup Helper.

Run this script to check your environment and install missing dependencies.
Works with Anaconda/Miniconda.

Usage (from Anaconda Prompt):
    1. Create the environment:
       conda env create -f environment.yml

    2. Activate it:
       conda activate ocean_rs

    3. Verify everything works:
       python setup_environment.py

    4. Open Spyder:
       spyder

    5. In Spyder, open run_gui.py or run_sar_gui.py and press F5.
"""

import sys


def check_dependency(name, import_name=None, required=True):
    """Check if a package is installed and importable."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', getattr(mod, 'version', '?'))
        print(f"  [OK] {name:20s} {version}")
        return True
    except ImportError:
        tag = "MISSING" if required else "optional"
        print(f"  [{tag}] {name:20s} not installed")
        return False


def main():
    print("=" * 60)
    print("OceanRS Environment Check")
    print(f"Python {sys.version}")
    print("=" * 60)

    print("\n--- Core (required) ---")
    ok = True
    ok &= check_dependency("numpy")
    ok &= check_dependency("GDAL", "osgeo.gdal")
    ok &= check_dependency("psutil")

    print("\n--- Geometry (recommended) ---")
    check_dependency("shapely", required=False)
    check_dependency("fiona", required=False)
    check_dependency("geopandas", required=False)

    print("\n--- GUI (optional) ---")
    check_dependency("tkinter", "_tkinter", required=False)
    check_dependency("tkintermapview", required=False)

    print("\n--- SAR Downloads (optional) ---")
    check_dependency("requests", required=False)
    check_dependency("asf_search", required=False)
    check_dependency("python-dotenv", "dotenv", required=False)

    print("\n--- OceanRS Package ---")
    try:
        from ocean_rs.optical import UnifiedS2TSSProcessor, TSSProcessor
        print("  [OK] ocean_rs.optical     imports OK")
    except ImportError as e:
        print(f"  [FAIL] ocean_rs.optical   {e}")
        ok = False

    try:
        from ocean_rs.sar import BathymetryPipeline
        print("  [OK] ocean_rs.sar         imports OK")
    except ImportError as e:
        print(f"  [FAIL] ocean_rs.sar       {e}")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("Environment is ready! Open Spyder and run:")
        print("  - run_gui.py      (Optical / Sentinel-2 Water Quality)")
        print("  - run_sar_gui.py  (SAR Bathymetry Toolkit)")
    else:
        print("MISSING required dependencies. Run:")
        print("  conda env create -f environment.yml")
        print("  conda activate ocean_rs")
    print("=" * 60)


if __name__ == "__main__":
    main()
