"""
Launcher script for Sentinel-2 TSS Pipeline GUI v2.

This launches the improved GUI with modern styling,
collapsible sections, and better layout.

Open this file in your IDE and run it (F5 in Spyder).

Note: This is a development version. The production GUI
is available via run_gui.py
"""

import sys
import os

# Add package to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Launch GUI v2
from sentinel2_tss_pipeline.gui_v2 import UnifiedS2TSSGUI

if __name__ == "__main__":
    print("Launching Sentinel-2 TSS Pipeline GUI v2...")
    print("(Development version with improved styling)")
    print()

    app = UnifiedS2TSSGUI()
    app.run()
