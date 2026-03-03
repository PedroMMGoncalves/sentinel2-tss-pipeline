"""
Launch OceanRS SAR Bathymetry Toolkit GUI.

Open this file in Spyder and press F5 (Run) to start the application.
Environment: conda activate ocean_rs
"""

import sys
import os

# Ensure project root is on sys.path (works from any Spyder working directory)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ocean_rs.sar.main import main

main()
