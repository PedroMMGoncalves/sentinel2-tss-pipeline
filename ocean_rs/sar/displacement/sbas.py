"""
Small Baseline Subset (SBAS) time-series InSAR.

Inverts a network of interferograms to estimate cumulative displacement
over time using SVD least-squares inversion.

Algorithm (Berardino et al., 2002):
    1. Build interferogram network with baseline thresholds
    2. Form design matrix A (rows=pairs, cols=acquisitions-1)
    3. Unwrap all interferograms
    4. SVD solve: A · displacement_rate = Δφ → cumulative displacement
    5. Temporal coherence filter
    6. Optional atmospheric phase screen estimation

References:
    Berardino, P. et al. (2002). A new algorithm for surface deformation
    monitoring based on small baseline differential SAR interferograms.
    IEEE TGRS, 40(11), 2375-2383.
"""

import logging
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from ..core.data_models import Interferogram, DisplacementField
from ocean_rs.shared.raster_io import check_memory_for_array

logger = logging.getLogger('ocean_rs')


def build_network(
    dates: List[str],
    perpendicular_baselines: Optional[List[float]] = None,
    max_temporal_days: int = 180,
    max_perpendicular_m: float = 200.0,
) -> List[Tuple[int, int]]:
    """Build interferometric pair network for SBAS.

    Selects pairs where both temporal and perpendicular baselines
    are below the specified thresholds.

    Args:
        dates: List of acquisition dates (ISO 8601 or YYYY-MM-DD).
        perpendicular_baselines: Optional perpendicular baselines (m)
            for each acquisition relative to a reference.
        max_temporal_days: Maximum temporal baseline (days).
        max_perpendicular_m: Maximum perpendicular baseline (m).

    Returns:
        List of (primary_idx, secondary_idx) pairs.
    """
    n_dates = len(dates)
    if n_dates < 2:
        raise ValueError("At least 2 acquisition dates required for SBAS")

    # Parse dates
    parsed_dates = []
    for d in dates:
        dt = _parse_date(d)
        if dt is None:
            raise ValueError(f"Cannot parse date: {d}")
        parsed_dates.append(dt)

    # Sort by date
    sorted_indices = np.argsort([dt.timestamp() for dt in parsed_dates])
    sorted_dates = [parsed_dates[i] for i in sorted_indices]

    pairs = []
    for i in range(n_dates):
        for j in range(i + 1, n_dates):
            idx_i = sorted_indices[i]
            idx_j = sorted_indices[j]

            # Temporal baseline
            temporal = abs((sorted_dates[j] - sorted_dates[i]).days)
            if temporal > max_temporal_days:
                continue

            # Perpendicular baseline (if available)
            if perpendicular_baselines is not None:
                b_perp = abs(
                    perpendicular_baselines[idx_j] -
                    perpendicular_baselines[idx_i]
                )
                if b_perp > max_perpendicular_m:
                    continue

            pairs.append((int(idx_i), int(idx_j)))

    logger.info(
        f"SBAS network: {len(pairs)} pairs from {n_dates} dates "
        f"(max_temporal={max_temporal_days}d, max_perp={max_perpendicular_m}m)"
    )

    if not pairs:
        raise ValueError(
            f"No valid interferometric pairs found with thresholds: "
            f"temporal={max_temporal_days}d, perpendicular={max_perpendicular_m}m. "
            f"Try increasing thresholds."
        )

    # Check network connectivity
    sorted_edges = [
        (
            int(np.where(sorted_indices == i)[0][0]),
            int(np.where(sorted_indices == j)[0][0]),
        )
        for i, j in pairs
    ]
    if not _check_connectivity(n_dates, sorted_edges):
        logger.warning(
            "SBAS network is DISCONNECTED. Displacement time-series will have "
            "arbitrary offsets between disconnected temporal subsets. "
            "Consider increasing max_temporal_baseline or max_perp_baseline."
        )

    return pairs


def compute_sbas(
    interferograms: List[Interferogram],
    pair_indices: List[Tuple[int, int]],
    dates: List[str],
    temporal_coherence_threshold: float = 0.7,
    reference_pixel: Optional[Tuple[int, int]] = None,
    atmospheric_filter: bool = False,
) -> List[DisplacementField]:
    """Compute SBAS time-series displacement.

    Steps:
        1. Build design matrix A
        2. Extract unwrapped phase from all interferograms
        3. SVD inversion per pixel: A · v = Δφ/(4π/λ)
        4. Integrate displacement rate to cumulative displacement
        5. Filter by temporal coherence
        6. Reference to stable pixel

    Args:
        interferograms: List of unwrapped interferograms.
        pair_indices: List of (primary_idx, secondary_idx) pairs.
        dates: Acquisition dates (ISO 8601), same order as indexed.
        temporal_coherence_threshold: Min temporal coherence for valid pixels.
        reference_pixel: Optional (row, col) for reference point.
        atmospheric_filter: If True, apply atmospheric phase screen removal.

    Returns:
        List of DisplacementField, one per acquisition date (relative to first).

    Raises:
        NotImplementedError: If atmospheric_filter is True (not yet implemented).
    """
    if atmospheric_filter:
        raise NotImplementedError(
            "Atmospheric phase screen removal is not yet implemented. "
            "Set atmospheric_filter=False in DisplacementConfig."
        )
    n_ifg = len(interferograms)
    n_dates = len(dates)

    if n_ifg == 0:
        raise ValueError("No interferograms provided")
    if n_ifg != len(pair_indices):
        raise ValueError(
            f"Mismatch: {n_ifg} interferograms but {len(pair_indices)} pair indices"
        )

    # Check all interferograms have unwrapped phase
    for i, ifg in enumerate(interferograms):
        if ifg.unwrapped_phase is None:
            raise ValueError(f"Interferogram {i} has no unwrapped phase")

    rows, cols = interferograms[0].unwrapped_phase.shape
    wavelength = interferograms[0].wavelength_m

    logger.info(
        f"SBAS inversion: {n_ifg} interferograms, {n_dates} dates, "
        f"{rows}×{cols} pixels"
    )

    # Parse and sort dates
    parsed_dates = [_parse_date(d) for d in dates]
    if any(d is None for d in parsed_dates):
        raise ValueError("Could not parse all acquisition dates")

    # Time intervals between consecutive dates (days)
    sorted_indices = np.argsort([dt.timestamp() for dt in parsed_dates])
    sorted_dates = [parsed_dates[i] for i in sorted_indices]

    # Build design matrix A (n_ifg × n_dates-1)
    # Each row corresponds to an interferogram
    # Each column corresponds to a time interval between consecutive dates
    A = np.zeros((n_ifg, n_dates - 1), dtype=np.float64)

    for k, (i, j) in enumerate(pair_indices):
        # Find positions in sorted order
        pos_i = np.where(sorted_indices == i)[0][0]
        pos_j = np.where(sorted_indices == j)[0][0]

        # Mark time intervals between date_i and date_j
        for m in range(min(pos_i, pos_j), max(pos_i, pos_j)):
            dt_interval = (sorted_dates[m + 1] - sorted_dates[m]).days
            sign = 1.0 if pos_j > pos_i else -1.0
            A[k, m] = sign * dt_interval

    # Memory check before allocating large phase stack
    check_memory_for_array(n_ifg, rows * cols, bytes_per_pixel=8,
                           description="SBAS phase stack")

    # Stack all unwrapped phases: (n_ifg, n_pixels)
    phase_stack = np.zeros((n_ifg, rows * cols), dtype=np.float64)
    for k, ifg in enumerate(interferograms):
        phase_stack[k, :] = ifg.unwrapped_phase.ravel()

    # In-place conversion: phase -> displacement (saves one full-stack copy)
    phase_stack *= -(wavelength / (4 * np.pi))
    disp_stack = phase_stack  # rename, same memory
    del phase_stack

    # SVD inversion per pixel
    logger.info("Running SVD inversion...")

    # SVD of design matrix (done once — same for all pixels)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Regularize: truncate small singular values
    s_threshold = S.max() * 1e-6
    S_inv = np.where(S > s_threshold, 1.0 / S, 0.0)

    # Pseudo-inverse: A+ = V · S^-1 · U^T
    A_pinv = (Vt.T * S_inv) @ U.T  # (n_dates-1, n_ifg)

    # Solve for displacement rate at each pixel (m/day)
    displacement_rate = A_pinv @ disp_stack  # (n_dates-1, n_pixels)

    # Integrate to cumulative displacement (multiply rate by time interval)
    # Accumulate in sorted order first to avoid read-before-write indexing bugs,
    # then map back to original date ordering.
    sorted_cumulative = np.zeros((n_dates, rows * cols), dtype=np.float64)
    for m in range(n_dates - 1):
        dt = (sorted_dates[m + 1] - sorted_dates[m]).days
        sorted_cumulative[m + 1, :] = sorted_cumulative[m, :] + displacement_rate[m, :] * dt

    # Map back to original date ordering
    cumulative = np.zeros_like(sorted_cumulative)
    for m in range(n_dates):
        cumulative[sorted_indices[m], :] = sorted_cumulative[m, :]

    # Temporal coherence: circular mean of residual phase
    # Convert displacement residuals back to phase for coherence calculation
    model_disp = A @ displacement_rate
    # In-place residual: avoids allocating a separate residual array
    residual_disp = disp_stack.copy()
    residual_disp -= model_disp
    del model_disp
    residual_phase = -(4 * np.pi / wavelength) * residual_disp
    temporal_coherence = np.abs(
        np.mean(np.exp(1j * residual_phase), axis=0)
    ).reshape(rows, cols)

    # Apply temporal coherence mask
    mask = temporal_coherence < temporal_coherence_threshold
    n_masked = np.sum(mask)
    if n_masked > 0:
        pct = 100 * n_masked / mask.size
        logger.info(
            f"Temporal coherence filter: masked {n_masked} pixels "
            f"({pct:.1f}%) below threshold {temporal_coherence_threshold}"
        )

    # Reference pixel subtraction
    if reference_pixel is not None:
        ref_r, ref_c = reference_pixel
        if 0 <= ref_r < rows and 0 <= ref_c < cols:
            ref_idx = ref_r * cols + ref_c
            for t in range(n_dates):
                cumulative[t, :] -= cumulative[t, ref_idx]
    else:
        # Auto-select: highest temporal coherence pixel
        ref_idx = np.argmax(temporal_coherence.ravel())
        ref_r, ref_c = divmod(ref_idx, cols)
        logger.info(f"Auto-selected reference pixel: ({ref_r}, {ref_c})")
        for t in range(n_dates):
            cumulative[t, :] -= cumulative[t, ref_idx]

    # Build output displacement fields
    results = []
    geo = interferograms[0].geo

    # Per-epoch uncertainty: scale base residual uncertainty by cumulative time span
    base_uncertainty = np.std(residual_disp, axis=0)
    total_span = (sorted_dates[-1] - sorted_dates[0]).days

    for t in range(n_dates):
        disp_map = cumulative[t, :].reshape(rows, cols).astype(np.float32)

        # Apply temporal coherence mask
        disp_map[mask] = np.nan

        # Scale uncertainty by normalized cumulative time from reference
        # Epochs further from reference accumulate more error proportionally
        sorted_pos = int(np.where(sorted_indices == t)[0][0])
        days_from_ref = abs((sorted_dates[sorted_pos] - sorted_dates[0]).days)
        # Linear scaling: uncertainty grows with sqrt of number of integration steps
        # from the reference epoch (error propagation through cumulative sum)
        n_steps = max(sorted_pos, 1)  # number of intervals integrated
        time_scale = np.sqrt(n_steps)
        uncertainty = (base_uncertainty * time_scale).reshape(rows, cols).astype(np.float32)
        uncertainty[mask] = np.nan

        results.append(DisplacementField(
            displacement_m=disp_map,
            uncertainty_m=uncertainty,
            component="LOS",
            reference_date=dates[sorted_indices[0]],
            measurement_date=dates[t],
            geo=geo,
            metadata={
                'method': 'SBAS',
                'wavelength_m': wavelength,
                'n_interferograms': n_ifg,
                'n_dates': n_dates,
                'temporal_coherence_threshold': temporal_coherence_threshold,
                'reference_pixel': (int(ref_r), int(ref_c)),
                'sign_convention': 'positive = uplift / toward sensor; negative = subsidence / away',
            },
        ))

    # Report max displacement from the last chronological epoch
    last_sorted_idx = sorted_indices[-1]
    logger.info(
        f"SBAS complete: {n_dates} displacement maps. "
        f"Max cumulative displacement (last epoch): "
        f"{np.nanmax(np.abs(cumulative[last_sorted_idx, :]))*1000:.1f} mm"
    )

    return results


def _check_connectivity(n_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    """Check if the interferogram network graph is fully connected.

    Uses BFS to verify all nodes are reachable from node 0.

    Args:
        n_nodes: Number of nodes (acquisition dates).
        edges: List of (i, j) edges (interferometric pairs in sorted order).

    Returns:
        True if the network is fully connected, False otherwise.
    """
    if n_nodes <= 1:
        return True

    adj = {i: set() for i in range(n_nodes)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    visited = set()
    queue = deque([0])
    visited.add(0)
    while queue:
        node = queue.popleft()  # O(1) instead of list.pop(0) O(N)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == n_nodes


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
