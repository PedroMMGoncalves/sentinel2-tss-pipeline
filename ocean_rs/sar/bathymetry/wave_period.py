"""
Wave period retrieval from WaveWatch III via NOAA ERDDAP.

ERDDAP is free, no authentication required.
"""

import logging
from functools import lru_cache

logger = logging.getLogger('ocean_rs')


@lru_cache(maxsize=32)
def get_wave_period(lon: float, lat: float, datetime_str: str) -> float:
    """Get dominant wave period from WaveWatch III.

    Uses NOAA ERDDAP to query the nearest WaveWatch III grid point.
    Results are cached to avoid repeated API calls.

    Args:
        lon: Longitude (degrees East)
        lat: Latitude (degrees North)
        datetime_str: UTC datetime (YYYY-MM-DDTHH:MM:SSZ)

    Returns:
        Dominant wave period in seconds

    Raises:
        RuntimeError: If ERDDAP query fails
    """
    import requests

    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    # L-10: ERDDAP dataset ID may change over time — verify at
    # https://coastwatch.pfeg.noaa.gov/erddap/
    dataset = "NWW3_Global_Best"

    url = (
        f"{base_url}/{dataset}.json?"
        f"perpw[({datetime_str}):1:({datetime_str})]"
        f"[({lat}):1:({lat})]"
        f"[({lon}):1:({lon})]"
    )

    logger.info(f"Querying WaveWatch III: lon={lon}, lat={lat}, time={datetime_str}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        rows = data.get('table', {}).get('rows', [])
        if rows and len(rows[0]) >= 4:
            period = float(rows[0][3])
            if period > 0:
                logger.info(f"WaveWatch III peak period: {period:.1f}s")
                return period

        raise ValueError("No valid wave period in response")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"ERDDAP query failed: {e}\n"
            "Use manual wave period entry as fallback."
        )
    except (ValueError, KeyError, IndexError) as e:
        raise RuntimeError(
            f"Failed to parse ERDDAP response: {e}\n"
            "Use manual wave period entry as fallback."
        )
