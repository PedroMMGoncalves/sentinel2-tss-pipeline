"""
Scene discovery using ASF (Alaska Satellite Facility) search API.

Wraps asf_search to find SAR scenes intersecting an AOI.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger('ocean_rs')


@dataclass
class SceneMetadata:
    """Metadata for a discovered SAR scene."""
    granule_id: str
    platform: str
    beam_mode: str
    polarization: str
    orbit_direction: str
    acquisition_date: str
    frame_number: int = 0
    path_number: int = 0
    size_mb: float = 0.0
    download_url: str = ""
    footprint_wkt: str = ""
    processing_level: str = "SLC"
    _asf_result: object = field(default=None, repr=False)


def search_scenes(aoi_wkt: str,
                  start_date: str,
                  end_date: str,
                  platform: str = "Sentinel-1",
                  beam_mode: str = "IW",
                  polarization: Optional[str] = None,
                  orbit_direction: Optional[str] = None,
                  processing_level: str = "SLC",
                  max_results: int = 250) -> List[SceneMetadata]:
    """Search ASF DAAC for SAR scenes.

    Args:
        aoi_wkt: Area of interest as WKT polygon
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        platform: Satellite platform (default: Sentinel-1)
        beam_mode: Beam mode (default: IW)
        polarization: Polarization filter (optional)
        orbit_direction: ASCENDING or DESCENDING (optional)
        processing_level: Processing level (default: SLC)
        max_results: Maximum results to return

    Returns:
        List of SceneMetadata sorted by acquisition date (newest first)
    """
    try:
        import asf_search as asf
    except ImportError:
        raise ImportError(
            "asf_search is required for scene discovery.\n"
            "Install with: pip install asf_search"
        )

    logger.info(f"Searching ASF: platform={platform}, beam={beam_mode}, "
                f"dates={start_date} to {end_date}")

    search_params = {
        'intersectsWith': aoi_wkt,
        'start': datetime.strptime(start_date, "%Y-%m-%d"),
        'end': datetime.strptime(end_date, "%Y-%m-%d"),
        'maxResults': max_results,
    }

    platform_map = {
        "Sentinel-1": asf.PLATFORM.SENTINEL1,
    }
    if platform in platform_map:
        search_params['platform'] = platform_map[platform]

    if beam_mode:
        search_params['beamMode'] = [beam_mode]

    if processing_level:
        search_params['processingLevel'] = [processing_level]

    if polarization:
        search_params['polarization'] = [polarization]

    if orbit_direction and orbit_direction.upper() in ("ASCENDING", "DESCENDING"):
        search_params['flightDirection'] = orbit_direction.upper()

    try:
        results = asf.search(**search_params)
    except Exception as e:
        raise RuntimeError(f"ASF search failed: {str(e)}")

    logger.info(f"Found {len(results)} scenes")

    scenes = []
    for r in results:
        props = r.properties
        try:
            scene = SceneMetadata(
                granule_id=props.get('sceneName', props.get('fileID', 'unknown')),
                platform=props.get('platform', platform),
                beam_mode=props.get('beamModeType', beam_mode),
                polarization=props.get('polarization', ''),
                orbit_direction=props.get('flightDirection', ''),
                acquisition_date=props.get('startTime', ''),
                frame_number=int(props.get('frameNumber', 0)),
                path_number=int(props.get('pathNumber', 0)),
                size_mb=float(props.get('bytes', 0)) / (1024 * 1024),
                download_url=props.get('url', ''),
                footprint_wkt=props.get('wkt', ''),
                processing_level=processing_level,
                _asf_result=r,
            )
            scenes.append(scene)
        except Exception as e:
            logger.warning(f"Failed to parse scene result: {e}")

    scenes.sort(key=lambda s: s.acquisition_date, reverse=True)
    return scenes
