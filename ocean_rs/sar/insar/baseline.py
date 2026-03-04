"""
Orbit-based baseline computation for InSAR.

Computes perpendicular and parallel baselines from orbit state vectors.
Also provides incidence angle and slant range computation needed for
topographic phase removal and displacement decomposition.

References:
    Hanssen, R. (2001). Radar Interferometry: Data Interpretation and Error
    Analysis. Kluwer Academic Publishers.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from ..core.data_models import OrbitStateVector, GeoTransform

logger = logging.getLogger('ocean_rs')


def compute_baseline(
    primary_orbits: List[OrbitStateVector],
    secondary_orbits: List[OrbitStateVector],
    scene_center_time: str,
    scene_center_lat: float = 0.0,
    scene_center_lon: float = 0.0,
) -> Tuple[float, float]:
    """Compute perpendicular and parallel baselines between two SAR acquisitions.

    Uses orbit state vectors interpolated to the scene center time to compute
    the spatial baseline and decompose it into perpendicular and parallel
    components relative to the line of sight.

    Args:
        primary_orbits: Primary image orbit state vectors.
        secondary_orbits: Secondary image orbit state vectors.
        scene_center_time: Scene center time (ISO 8601).
        scene_center_lat: Approximate scene center latitude (degrees).
        scene_center_lon: Approximate scene center longitude (degrees).

    Returns:
        Tuple of (perpendicular_baseline_m, parallel_baseline_m).
        Perpendicular baseline is positive when secondary is further from Earth.

    Raises:
        ValueError: If orbit state vectors are insufficient for interpolation.
    """
    if abs(scene_center_lat) < 0.001 and abs(scene_center_lon) < 0.001:
        logger.warning(
            "Scene center is (0,0) - likely not set from metadata. "
            "Baseline accuracy may be degraded for non-equatorial scenes."
        )

    if len(primary_orbits) < 2 or len(secondary_orbits) < 2:
        logger.warning(
            "Insufficient orbit state vectors for baseline computation. "
            f"Primary: {len(primary_orbits)}, Secondary: {len(secondary_orbits)}. "
            "Returning zero baseline."
        )
        return 0.0, 0.0

    # Interpolate positions to scene center time
    primary_pos = _interpolate_orbit(primary_orbits, scene_center_time)
    secondary_pos = _interpolate_orbit(secondary_orbits, scene_center_time)

    if primary_pos is None or secondary_pos is None:
        logger.warning("Orbit interpolation failed. Returning zero baseline.")
        return 0.0, 0.0

    # Spatial baseline vector
    primary_pos_xyz = np.array(primary_pos[:3])
    primary_vel_xyz = np.array(primary_pos[3:6])
    baseline_vec = np.array(secondary_pos[:3]) - primary_pos_xyz
    baseline_magnitude = np.linalg.norm(baseline_vec)

    if baseline_magnitude < 1e-6:
        return 0.0, 0.0

    # Approximate look vector (satellite to scene center)
    # Use WGS84 ellipsoid to convert lat/lon to ECEF
    scene_ecef = _geodetic_to_ecef(scene_center_lat, scene_center_lon, 0.0)
    look_vec = scene_ecef - primary_pos_xyz
    look_unit = look_vec / np.linalg.norm(look_vec)

    # Decompose baseline into parallel (along look) and perpendicular components
    b_parallel = np.dot(baseline_vec, look_unit)

    # Compute signed perpendicular baseline using velocity-based cross-track
    # direction. The along-track direction is the satellite velocity vector,
    # and the cross-track direction is derived from the cross product of the
    # along-track and look directions. This properly determines the B_perp sign
    # for all orbit geometries (ascending/descending, left/right looking).
    along_track = primary_vel_xyz / np.linalg.norm(primary_vel_xyz)
    cross_track = np.cross(along_track, look_unit)
    cross_track_norm = np.linalg.norm(cross_track)
    if cross_track_norm > 1e-10:
        cross_track = cross_track / cross_track_norm
        b_perpendicular = float(np.dot(baseline_vec, cross_track))
    else:
        # Degenerate geometry — fall back to unsigned magnitude
        b_perp_vec = baseline_vec - b_parallel * look_unit
        b_perpendicular = float(np.linalg.norm(b_perp_vec))
        logger.warning(
            "Cross-track direction degenerate (along-track parallel to look). "
            "B_perp sign may be unreliable."
        )

    logger.info(
        f"Baseline: B_perp={b_perpendicular:.1f}m, "
        f"B_par={b_parallel:.1f}m, |B|={baseline_magnitude:.1f}m"
    )

    return b_perpendicular, b_parallel


def compute_incidence_angle(
    orbit_vectors: List[OrbitStateVector],
    scene_center_time: str,
    geo: GeoTransform,
    dem: Optional[np.ndarray] = None,
    near_angle_deg: float = 29.0,
    far_angle_deg: float = 46.0,
) -> np.ndarray:
    """Compute incidence angle grid for the scene.

    The incidence angle is the angle between the radar line of sight
    and the local surface normal (vertical for flat terrain).

    For flat terrain without DEM:
        theta_inc = arccos(h_sat / R)
    where h_sat is satellite altitude and R is slant range.

    Args:
        orbit_vectors: Satellite orbit state vectors.
        scene_center_time: Scene center time (ISO 8601).
        geo: Scene geotransform (for grid dimensions).
        dem: Optional DEM array (same grid as scene). None = flat terrain.
        near_angle_deg: Near-range incidence angle in degrees.
            Defaults to 29.0 (Sentinel-1 IW near range).
        far_angle_deg: Far-range incidence angle in degrees.
            Defaults to 46.0 (Sentinel-1 IW far range).

    Returns:
        Incidence angle array in radians, shape (rows, cols).
    """
    rows = geo.rows if geo.rows > 0 else 1
    cols = geo.cols if geo.cols > 0 else 1

    if not orbit_vectors:
        default_angle = (near_angle_deg + far_angle_deg) / 2.0
        logger.warning(
            f"No orbit vectors available. Using default incidence angle of "
            f"{default_angle:.1f} degrees."
        )
        return np.full((rows, cols), np.radians(default_angle), dtype=np.float32)

    # Interpolate orbit to scene center
    pos = _interpolate_orbit(orbit_vectors, scene_center_time)
    if pos is None:
        default_angle = (near_angle_deg + far_angle_deg) / 2.0
        return np.full((rows, cols), np.radians(default_angle), dtype=np.float32)

    # Satellite altitude above WGS84 ellipsoid
    sat_pos = np.array(pos[:3])
    # Compute geocentric latitude from ECEF position
    sat_lat_gc = np.arctan2(sat_pos[2], np.sqrt(sat_pos[0]**2 + sat_pos[1]**2))
    # WGS84 ellipsoidal radius at this latitude
    a = 6378137.0  # WGS84 semi-major
    b = 6356752.314245  # WGS84 semi-minor
    cos_lat = np.cos(sat_lat_gc)
    sin_lat = np.sin(sat_lat_gc)
    R_local = np.sqrt(
        ((a**2 * cos_lat)**2 + (b**2 * sin_lat)**2) /
        ((a * cos_lat)**2 + (b * sin_lat)**2)
    )
    sat_altitude = np.linalg.norm(sat_pos) - R_local

    # For each range pixel, compute incidence angle
    # Simplified: linear variation across range (near to far)
    near_angle = np.radians(near_angle_deg)
    far_angle = np.radians(far_angle_deg)

    range_angles = np.linspace(near_angle, far_angle, cols, dtype=np.float32)
    incidence = np.tile(range_angles, (rows, 1))

    return incidence


def _interpolate_orbit(
    orbit_vectors: List[OrbitStateVector],
    target_time: str,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Interpolate orbit state vectors to a target time.

    Uses linear interpolation between the two nearest state vectors.
    Lagrange interpolation would be more accurate but linear is
    sufficient for baseline computation.

    Note: Uses linear interpolation between nearest state vectors. For
    sub-meter accuracy, Lagrange polynomial (degree 5+) interpolation
    is recommended.

    Returns:
        Tuple (x, y, z, vx, vy, vz) or None if interpolation fails.
    """
    if not orbit_vectors:
        return None

    # Parse target time
    target_dt = _parse_time(target_time)
    if target_dt is None:
        # Use midpoint of orbit vectors
        n = len(orbit_vectors)
        mid = orbit_vectors[n // 2]
        return (mid.x, mid.y, mid.z, mid.vx, mid.vy, mid.vz)

    # Convert orbit times to seconds relative to first vector
    orbit_times = []
    for osv in orbit_vectors:
        dt = _parse_time(osv.time_utc)
        if dt is not None:
            orbit_times.append((dt - target_dt).total_seconds())
        else:
            orbit_times.append(None)

    # Find bracketing vectors
    valid_indices = [i for i, t in enumerate(orbit_times) if t is not None]
    if not valid_indices:
        mid = orbit_vectors[len(orbit_vectors) // 2]
        return (mid.x, mid.y, mid.z, mid.vx, mid.vy, mid.vz)

    # Sort by absolute time difference
    valid_indices.sort(key=lambda i: abs(orbit_times[i]))

    if len(valid_indices) < 2:
        osv = orbit_vectors[valid_indices[0]]
        return (osv.x, osv.y, osv.z, osv.vx, osv.vy, osv.vz)

    # Find nearest before and after target
    before_idx = None
    after_idx = None
    for i in valid_indices:
        t = orbit_times[i]
        if t <= 0 and (before_idx is None or t > orbit_times[before_idx]):
            before_idx = i
        if t >= 0 and (after_idx is None or t < orbit_times[after_idx]):
            after_idx = i

    if before_idx is None:
        before_idx = valid_indices[0]
    if after_idx is None:
        after_idx = valid_indices[-1]
    if before_idx == after_idx:
        osv = orbit_vectors[before_idx]
        return (osv.x, osv.y, osv.z, osv.vx, osv.vy, osv.vz)

    # Linear interpolation
    t0 = orbit_times[before_idx]
    t1 = orbit_times[after_idx]
    dt = t1 - t0
    if abs(dt) < 1e-10:
        osv = orbit_vectors[before_idx]
        return (osv.x, osv.y, osv.z, osv.vx, osv.vy, osv.vz)

    w = -t0 / dt  # Weight for after_idx (0 at t0, 1 at t1)
    w = max(0.0, min(1.0, w))

    osv0 = orbit_vectors[before_idx]
    osv1 = orbit_vectors[after_idx]

    return (
        osv0.x + w * (osv1.x - osv0.x),
        osv0.y + w * (osv1.y - osv0.y),
        osv0.z + w * (osv1.z - osv0.z),
        osv0.vx + w * (osv1.vx - osv0.vx),
        osv0.vy + w * (osv1.vy - osv0.vy),
        osv0.vz + w * (osv1.vz - osv0.vz),
    )


def _geodetic_to_ecef(lat_deg: float, lon_deg: float,
                       alt_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ECEF (WGS84).

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        alt_m: Altitude above ellipsoid in meters.

    Returns:
        ECEF position as numpy array [x, y, z].
    """
    # WGS84 parameters
    a = 6378137.0          # Semi-major axis (m)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f * f     # First eccentricity squared

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1 - e2 * sin_lat ** 2)  # Radius of curvature

    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * sin_lat

    return np.array([x, y, z])


def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse various time string formats to datetime."""
    if not time_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None
