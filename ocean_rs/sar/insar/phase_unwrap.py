"""
Phase unwrapping for InSAR interferograms.

Default: snaphu-py (pip-installable SNAPHU wrapper) for production use.
Fallback: quality-guided unwrapping (pure Python) for small scenes only.

The wrapped phase φ ∈ [-π, π] must be unwrapped to recover the true
continuous phase which is proportional to the range difference.

References:
    Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping
    with use of statistical models for cost functions in nonlinear
    optimization. JOSA A, 18(2), 338-351.

    Ghiglia, D.C. & Pritt, M.D. (1998). Two-Dimensional Phase Unwrapping:
    Theory, Algorithms, and Software. Wiley.
"""

import heapq
import logging
import math

import numpy as np

from ..core.data_models import Interferogram

logger = logging.getLogger('ocean_rs')

# Maximum image size for pure-Python fallback (pixels per dimension)
MAX_QUALITY_GUIDED_SIZE = 5000

# --- Numba optional import ---
try:
    from numba import njit
    HAS_NUMBA = True
    logger.debug("Numba available — JIT-accelerated phase unwrapping enabled")
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity decorator fallback when Numba is not installed."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


def unwrap_phase(
    interferogram: Interferogram,
    method: str = "auto",
    nlooks: float = None,
    cost: str = "smooth",
) -> np.ndarray:
    """Unwrap interferometric phase.

    Args:
        interferogram: Interferogram with wrapped phase and coherence.
        method: Unwrapping method.
            'auto': snaphu if available, quality-guided fallback.
            'snaphu': Use snaphu-py (raises ImportError if not installed).
            'quality_guided': Pure Python quality-guided (slow for large images).
        nlooks: Effective number of looks for SNAPHU. If None, derived from
            interferogram metadata (coherence_window_range * coherence_window_azimuth).
            Falls back to 20 if metadata is not available.
        cost: SNAPHU cost mode. Valid values: 'smooth', 'defo', 'topo'.
            Default 'smooth'.

    Returns:
        Unwrapped phase array (radians), same shape as input phase.

    Raises:
        ImportError: If snaphu method requested but snaphu-py not installed.
        RuntimeError: If unwrapping fails.
    """
    phase = interferogram.phase
    coherence = interferogram.coherence

    if phase is None:
        raise ValueError("Interferogram phase is None — cannot unwrap")
    if coherence is None:
        raise ValueError("Interferogram coherence is None — required for unwrapping")
    if phase.shape != coherence.shape:
        raise ValueError(
            f"Phase shape {phase.shape} != coherence shape {coherence.shape}"
        )

    rows, cols = phase.shape

    # Derive nlooks from coherence estimation window if not provided
    if nlooks is None:
        meta = interferogram.metadata if interferogram.metadata else {}
        win_range = meta.get('coherence_window_range')
        win_azimuth = meta.get('coherence_window_azimuth')
        if win_range is not None and win_azimuth is not None:
            nlooks = float(win_range) * float(win_azimuth)
            logger.info(
                f"Derived nlooks={nlooks:.0f} from coherence window "
                f"({win_range}×{win_azimuth})"
            )
        else:
            nlooks = 20.0
            logger.info(
                "Coherence window metadata not available, using default nlooks=20"
            )

    logger.info(f"Phase unwrapping: {rows}×{cols} pixels, method={method}, nlooks={nlooks:.1f}")

    if method == "auto":
        try:
            import snaphu  # noqa: F401
            method = "snaphu"
            logger.info("Using snaphu-py for phase unwrapping")
        except ImportError:
            if rows > MAX_QUALITY_GUIDED_SIZE or cols > MAX_QUALITY_GUIDED_SIZE:
                raise ImportError(
                    f"Image too large ({rows}×{cols}) for quality-guided unwrapping. "
                    f"Install snaphu-py: pip install snaphu"
                )
            method = "quality_guided"
            logger.warning(
                "snaphu-py not installed. Using quality-guided fallback "
                "(slow, suitable for small scenes only). "
                "Install snaphu: pip install snaphu"
            )

    if method == "quality_guided" and (rows > MAX_QUALITY_GUIDED_SIZE or cols > MAX_QUALITY_GUIDED_SIZE):
        raise ValueError(
            f"Image too large ({rows}×{cols}) for quality-guided unwrapping "
            f"(max {MAX_QUALITY_GUIDED_SIZE}×{MAX_QUALITY_GUIDED_SIZE}). "
            f"Use method='snaphu' or method='auto'."
        )

    if cost not in ("smooth", "defo", "topo"):
        raise ValueError(
            f"Unknown SNAPHU cost mode: {cost!r}. "
            f"Valid values: 'smooth', 'defo', 'topo'."
        )

    if method == "snaphu":
        return _unwrap_snaphu(phase, coherence, nlooks, cost=cost)
    elif method == "quality_guided":
        if HAS_NUMBA:
            logger.info("Using Numba JIT-accelerated quality-guided unwrapper")
            return _unwrap_quality_guided_numba(phase, coherence)
        return _unwrap_quality_guided(phase, coherence)
    else:
        raise ValueError(f"Unknown unwrapping method: {method}")


def _unwrap_snaphu(
    phase: np.ndarray,
    coherence: np.ndarray,
    nlooks: float = 20.0,
    cost: str = "smooth",
) -> np.ndarray:
    """Unwrap phase using snaphu-py.

    snaphu-py is a Python wrapper around the SNAPHU algorithm
    (Statistical-cost, Network-flow Algorithm for Phase Unwrapping).

    Args:
        phase: Wrapped phase array.
        coherence: Coherence array.
        nlooks: Effective number of looks (coherence_window_range *
            coherence_window_azimuth). Affects the statistical cost model.
        cost: SNAPHU cost mode ('smooth', 'defo', or 'topo').
    """
    try:
        import snaphu
    except ImportError:
        raise ImportError(
            "snaphu-py is required for SNAPHU unwrapping.\n"
            "Install with: pip install snaphu"
        )

    rows, cols = phase.shape
    logger.info(
        f"Running SNAPHU unwrapping on {rows}×{cols} image, "
        f"nlooks={nlooks:.1f}, cost={cost}"
    )

    # snaphu-py expects a complex interferogram, not real-valued phase
    igram = np.exp(1j * phase.astype(np.float64))
    coherence_f32 = coherence.astype(np.float32)

    # Run SNAPHU
    result, _ = snaphu.unwrap(
        igram,
        coherence_f32,
        nlooks=nlooks,
        cost=cost,
    )

    # Extract phase from result
    if np.issubdtype(result.dtype, np.complexfloating):
        raise RuntimeError(
            "SNAPHU returned complex output (unexpected). "
            "Please update snaphu-py: pip install --upgrade snaphu"
        )
    unwrapped = result.astype(np.float32)

    logger.info(
        f"SNAPHU unwrapping complete. "
        f"Range: [{np.nanmin(unwrapped):.1f}, {np.nanmax(unwrapped):.1f}] rad"
    )

    return unwrapped


# --- Numba JIT kernel: array-based binary max-heap unwrapper ---

@njit(cache=True)
def _heap_push(keys, vals, size, key, val):
    """Push (key, val) onto a max-heap stored in pre-allocated arrays."""
    pos = size
    keys[pos] = key
    vals[pos] = val
    size += 1
    # Sift up
    while pos > 0:
        parent = (pos - 1) >> 1
        if keys[parent] < keys[pos]:
            keys[parent], keys[pos] = keys[pos], keys[parent]
            vals[parent], vals[pos] = vals[pos], vals[parent]
            pos = parent
        else:
            break
    return size


@njit(cache=True)
def _heap_pop(keys, vals, size):
    """Pop max element from the heap. Returns (key, val, new_size)."""
    key = keys[0]
    val = vals[0]
    size -= 1
    keys[0] = keys[size]
    vals[0] = vals[size]
    # Sift down
    pos = 0
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        largest = pos
        if left < size and keys[left] > keys[largest]:
            largest = left
        if right < size and keys[right] > keys[largest]:
            largest = right
        if largest != pos:
            keys[pos], keys[largest] = keys[largest], keys[pos]
            vals[pos], vals[largest] = vals[largest], vals[pos]
            pos = largest
        else:
            break
    return key, val, size


@njit(cache=True)
def _unwrap_qg_kernel(phase, quality, seed_row, seed_col):
    """Quality-guided phase unwrapping — Numba JIT kernel.

    Uses an array-based binary max-heap (sorted by coherence) instead of
    Python's heapq, enabling full JIT compilation of the flood-fill loop.

    Returns unwrapped phase (float64).
    """
    rows = phase.shape[0]
    cols = phase.shape[1]
    n_pixels = rows * cols
    two_pi = 2.0 * math.pi

    unwrapped = np.empty((rows, cols), dtype=np.float64)
    unwrapped[:, :] = np.nan
    processed = np.zeros((rows, cols), dtype=np.bool_)

    # Pre-allocate heap arrays (quality as key, flat index as value)
    # Worst case: each pixel can appear in the heap multiple times before
    # being popped, but empirically bounded by 2N. For 5000×5000: ~800 MB.
    max_heap = 2 * n_pixels
    heap_keys = np.empty(max_heap, dtype=np.float64)
    heap_vals = np.empty(max_heap, dtype=np.int64)
    heap_size = 0

    # Seed pixel
    unwrapped[seed_row, seed_col] = phase[seed_row, seed_col]
    processed[seed_row, seed_col] = True

    # 4-connected neighbor offsets
    dr = np.array([-1, 1, 0, 0], dtype=np.int64)
    dc = np.array([0, 0, -1, 1], dtype=np.int64)

    # Push neighbors of seed
    for k in range(4):
        nr = seed_row + dr[k]
        nc = seed_col + dc[k]
        if 0 <= nr < rows and 0 <= nc < cols and not processed[nr, nc]:
            flat = nr * cols + nc
            heap_size = _heap_push(heap_keys, heap_vals, heap_size,
                                   quality[nr, nc], flat)

    while heap_size > 0:
        q, flat, heap_size = _heap_pop(heap_keys, heap_vals, heap_size)
        r = flat // cols
        c = flat % cols

        if processed[r, c]:
            continue

        # Find best processed neighbor
        best_q = -1.0
        best_r = -1
        best_c = -1
        for k in range(4):
            nr = r + dr[k]
            nc = c + dc[k]
            if 0 <= nr < rows and 0 <= nc < cols and processed[nr, nc]:
                nq = quality[nr, nc]
                if nq > best_q:
                    best_q = nq
                    best_r = nr
                    best_c = nc

        if best_r < 0:
            continue

        # Unwrap relative to best neighbor
        diff = phase[r, c] - phase[best_r, best_c]
        diff = diff - two_pi * round(diff / two_pi)
        unwrapped[r, c] = unwrapped[best_r, best_c] + diff
        processed[r, c] = True

        # Push unprocessed neighbors
        for k in range(4):
            nr = r + dr[k]
            nc = c + dc[k]
            if 0 <= nr < rows and 0 <= nc < cols and not processed[nr, nc]:
                flat = nr * cols + nc
                heap_size = _heap_push(heap_keys, heap_vals, heap_size,
                                       quality[nr, nc], flat)

    return unwrapped


def _unwrap_quality_guided_numba(
    phase: np.ndarray,
    coherence: np.ndarray,
) -> np.ndarray:
    """Quality-guided unwrapping via Numba JIT (wrapper with logging)."""
    rows, cols = phase.shape
    logger.info(
        f"Quality-guided unwrapping (Numba JIT): {rows}×{cols} pixels "
        f"(first call compiles — may take a few seconds)"
    )

    quality = coherence.astype(np.float64)  # creates a copy, avoids redundant .copy()
    quality[np.isnan(quality)] = 0.0

    if np.max(quality) < 1e-10:
        raise ValueError(
            "All coherence values are zero or NaN. "
            "Phase unwrapping requires valid coherence data."
        )

    seed_idx = np.unravel_index(np.argmax(quality), quality.shape)

    from ocean_rs.shared.raster_io import check_memory_for_array
    # Heap arrays (2*N*8*2) + unwrapped (N*8) + processed (N*1) ≈ 41 bytes/pixel
    check_memory_for_array(rows, cols, bytes_per_pixel=41,
                           description="quality-guided phase unwrapping")

    unwrapped = _unwrap_qg_kernel(
        phase.astype(np.float64),
        quality,  # already float64
        seed_idx[0], seed_idx[1],
    )

    nan_count = np.sum(np.isnan(unwrapped))
    total_pixels = rows * cols
    if nan_count > 0:
        logger.warning(
            f"{nan_count} pixels ({100*nan_count/total_pixels:.1f}%) "
            f"could not be unwrapped (low coherence regions) — marked as NaN"
        )

    logger.info(
        f"Quality-guided unwrapping complete. "
        f"Range: [{np.nanmin(unwrapped):.1f}, {np.nanmax(unwrapped):.1f}] rad"
    )

    return unwrapped.astype(np.float32)


def _unwrap_quality_guided(
    phase: np.ndarray,
    coherence: np.ndarray,
) -> np.ndarray:
    """Quality-guided phase unwrapping (pure Python).

    Processes pixels from highest to lowest quality (coherence).
    Each pixel is unwrapped relative to its best already-unwrapped neighbor:
        unwrapped[i,j] = unwrapped[neighbor] + wrap(phase[i,j] - phase[neighbor])

    where wrap(x) = x - 2π·round(x/2π)

    This is a simple but effective algorithm for moderate-quality interferograms.
    For large or low-coherence images, use SNAPHU instead.
    """
    rows, cols = phase.shape
    logger.info(
        f"Quality-guided unwrapping: {rows}×{cols} pixels "
        f"(pure Python — this may be slow)"
    )

    unwrapped = np.full_like(phase, np.nan, dtype=np.float64)
    processed = np.zeros((rows, cols), dtype=bool)

    # Start from highest coherence pixel
    quality = coherence.copy()
    quality[np.isnan(quality)] = 0.0

    seed_idx = np.unravel_index(np.argmax(quality), quality.shape)
    unwrapped[seed_idx] = phase[seed_idx]
    processed[seed_idx] = True

    # Priority queue: (-quality, row, col) — max-heap via negation
    # Initialize with neighbors of seed
    heap = []
    _push_neighbors(heap, seed_idx[0], seed_idx[1], rows, cols, quality, processed)

    pixels_unwrapped = 1
    total_pixels = rows * cols

    while heap:
        neg_q, r, c = heapq.heappop(heap)

        if processed[r, c]:
            continue

        # Find best processed neighbor
        best_neighbor = _best_neighbor(r, c, rows, cols, processed, quality)
        if best_neighbor is None:
            continue

        nr, nc = best_neighbor

        # Unwrap relative to neighbor
        phase_diff = phase[r, c] - phase[nr, nc]
        unwrapped[r, c] = unwrapped[nr, nc] + _wrap(phase_diff)
        processed[r, c] = True
        pixels_unwrapped += 1

        # Add neighbors to queue
        _push_neighbors(heap, r, c, rows, cols, quality, processed)

        # Progress logging
        if pixels_unwrapped % 1000000 == 0:
            pct = 100.0 * pixels_unwrapped / total_pixels
            logger.info(f"  Unwrapping progress: {pct:.1f}%")

    # Preserve NaN for pixels that could not be unwrapped (no data)
    # Filling with 0 would bias results — NaN correctly indicates no valid data
    nan_count = np.sum(np.isnan(unwrapped))
    if nan_count > 0:
        logger.warning(
            f"{nan_count} pixels ({100*nan_count/total_pixels:.1f}%) "
            f"could not be unwrapped (low coherence regions) — marked as NaN"
        )

    logger.info(
        f"Quality-guided unwrapping complete. "
        f"Range: [{np.nanmin(unwrapped):.1f}, {np.nanmax(unwrapped):.1f}] rad"
    )

    return unwrapped.astype(np.float32)


def _wrap(phase_diff: float) -> float:
    """Wrap phase difference to [-π, π]."""
    return phase_diff - 2 * np.pi * round(phase_diff / (2 * np.pi))


def _push_neighbors(
    heap: list,
    r: int, c: int,
    rows: int, cols: int,
    quality: np.ndarray,
    processed: np.ndarray,
) -> None:
    """Push unprocessed neighbors onto the priority queue."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not processed[nr, nc]:
            heapq.heappush(heap, (-quality[nr, nc], nr, nc))


def _best_neighbor(
    r: int, c: int,
    rows: int, cols: int,
    processed: np.ndarray,
    quality: np.ndarray,
) -> tuple:
    """Find the best (highest quality) processed neighbor."""
    best = None
    best_q = -1.0

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and processed[nr, nc]:
            q = quality[nr, nc]
            if q > best_q:
                best_q = q
                best = (nr, nc)

    return best
