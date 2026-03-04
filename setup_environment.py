"""
OceanRS Environment Verification.

Checks that all required and optional dependencies are installed
and importable. Run this after setting up the environment.

Usage (from Anaconda Prompt):
    1. Automated install (recommended):
       install_environment.bat

    2. Or manual setup:
       conda create -n ocean_rs python=3.12.* -c conda-forge
       conda activate ocean_rs
       conda install -c conda-forge numpy gdal psutil shapely fiona geopandas requests spyder
       python -m pip install sv-ttk asf_search python-dotenv tkintermapview

    3. Verify:
       python setup_environment.py

    4. In Spyder, set interpreter to ocean_rs:
       Tools > Preferences > Python interpreter >
       "Use the following interpreter" > ocean_rs/python.exe
"""

import sys


def check_dependency(name, import_name=None, required=True):
    """Check if a package is installed and importable."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', getattr(mod, 'version', None))
        if version:
            print(f"  [OK] {name:20s} {version}")
        else:
            print(f"  [OK] {name:20s} installed")
        return True
    except ImportError:
        tag = "MISSING" if required else "optional"
        print(f"  [{tag}] {name:20s} not installed")
        return False
    except PermissionError:
        print(f"  [BLOCKED] {name:20s} antivirus is blocking this package")
        return False


def main():
    print("=" * 60)
    print("OceanRS Environment Check")
    print(f"Python {sys.version}")
    print("=" * 60)

    # Python version check
    major, minor = sys.version_info[:2]
    if minor != 12:
        print(f"\n  WARNING: Python 3.12 recommended (you have 3.{minor})")
        print(f"  GDAL and fiona may not work with Python 3.{minor}")

    print("\n--- Core (required) ---")
    ok = True
    ok &= check_dependency("numpy")
    ok &= check_dependency("GDAL", "osgeo.gdal")
    ok &= check_dependency("psutil")

    print("\n--- Geometry (recommended) ---")
    check_dependency("shapely", required=False)
    check_dependency("fiona", required=False)
    check_dependency("geopandas", required=False)

    print("\n--- GUI ---")
    check_dependency("tkinter", "_tkinter", required=False)
    check_dependency("sv-ttk (Azure theme)", "sv_ttk", required=False)
    check_dependency("tkintermapview", required=False)

    print("\n--- SAR Downloads (optional) ---")
    check_dependency("requests", required=False)
    check_dependency("asf_search", required=False)
    check_dependency("python-dotenv", "dotenv", required=False)

    print("\n--- IDE ---")
    check_dependency("spyder", required=False)

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

    # SAR toolkit verification
    print("\n--- SAR Toolkit ---")
    print("Verifying SAR toolkit imports...")
    try:
        from ocean_rs.sar.insar import InSARPipeline
        print("  InSARPipeline: OK")
    except ImportError as e:
        print(f"  InSARPipeline: FAILED ({e})")

    try:
        from ocean_rs.sar.displacement import compute_dinsar, compute_sbas
        print("  Displacement: OK")
    except ImportError as e:
        print(f"  Displacement: FAILED ({e})")

    # Optional dependencies
    for pkg, desc in [('h5py', 'NISAR HDF5 support'), ('snaphu', 'Phase unwrapping'), ('scipy', 'Scientific computing')]:
        try:
            __import__(pkg)
            print(f"  {pkg} ({desc}): OK")
        except ImportError:
            print(f"  {pkg} ({desc}): NOT INSTALLED (optional)")

    print("\n" + "=" * 60)
    if ok:
        print("Environment is ready! Open Spyder and run:")
        print("  - run_gui.py      (Optical / Sentinel-2 Water Quality)")
        print("  - run_sar_gui.py  (SAR Bathymetry Toolkit)")
    else:
        print("MISSING required dependencies. Run:")
        print("  install_environment.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
