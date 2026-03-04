"""
InSAR processing subpackage for OceanRS SAR Toolkit.

Provides interferometric SAR processing capabilities:
    - Orbit-based baseline computation
    - SLC co-registration (orbit coarse + ESD/coherence fine)
    - Interferogram formation and coherence estimation
    - Goldstein adaptive phase filtering
    - Phase unwrapping (snaphu-py default, quality-guided fallback)
    - Topographic phase removal using DEM
    - Geocoding via GDAL with GCPs
    - End-to-end InSAR pipeline orchestration
"""

from .baseline import compute_baseline, compute_incidence_angle
from .coregistration import coregister
from .interferogram import form_interferogram
from .phase_filter import goldstein_filter
from .phase_unwrap import unwrap_phase
from .topo_removal import remove_topographic_phase
from .geocoding import geocode
from .insar_pipeline import InSARPipeline

__all__ = [
    'compute_baseline',
    'compute_incidence_angle',
    'coregister',
    'form_interferogram',
    'goldstein_filter',
    'unwrap_phase',
    'remove_topographic_phase',
    'geocode',
    'InSARPipeline',
]
