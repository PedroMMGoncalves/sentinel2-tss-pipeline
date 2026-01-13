"""
Launcher script for Sentinel-2 TSS Pipeline GUI.

Open this file in your IDE and run it (F5 in Spyder).
"""

import sys
import os

# Add package to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Launch GUI
from sentinel2_tss_pipeline.gui import UnifiedS2TSSGUI

if __name__ == "__main__":
    app = UnifiedS2TSSGUI()
    app.run()
