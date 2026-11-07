"""
Shared coverage utilities for exploration nodes.

Both frontier_explorer and rl_agent import check_coverage() and load_expected_cells()
to monitor how much of the expected traversable area has been mapped.
The expected cell count is pre-calculated by generate_random_world() and stored in
WORLD_META_FILE so nodes can read it regardless of how they are launched.
"""
import json

from nav_msgs.msg import OccupancyGrid
import numpy as np

from tb3_slam.utils import load_world_size, WORLD_META_FILE

DEFAULT_COVERAGE_THRESHOLD: float = 0.97

EPISODE_TIMEOUT_BY_WORLD_SIZE: dict[float, float] = {
    8.0: 600.0,
    12.0: 900.0,
}
_TIMEOUT_PER_AREA_S: float = EPISODE_TIMEOUT_BY_WORLD_SIZE[8.0] / (8.0 * 8.0)


def timeout_for_world_size(world_size: float) -> float:
    """Episode timeout (simulation seconds) appropriate for a square world.

    Uses the tuned per-size budget for the study sizes (8, 12) and falls back to
    an area-proportional estimate for any other size.
    """
    if world_size in EPISODE_TIMEOUT_BY_WORLD_SIZE:
        return EPISODE_TIMEOUT_BY_WORLD_SIZE[world_size]
    return _TIMEOUT_PER_AREA_S * float(world_size) * float(world_size)


def load_episode_timeout() -> float:
    """Episode timeout for the world described by the shared metadata file."""
    return timeout_for_world_size(load_world_size(WORLD_META_FILE))


def load_expected_cells() -> int | None:
    """Read the expected number of traversable cells from the world metadata.

    The count is written by generate_random_world(). Returns None if the file
    does not exist or cannot be parsed – callers should fall back to a sensible
    default (e.g. total map size) and log a warning.
    """
    if not WORLD_META_FILE.exists():
        return None
    try:
        return int(json.loads(WORLD_META_FILE.read_text())['expected_traversable_cells'])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def check_coverage(msg: OccupancyGrid, expected_cells: int) -> tuple[float, int]:
    """Compute exploration coverage against the pre-calculated cell count.

    A cell counts as "discovered" when its occupancy value is >= 0 (i.e. not unknown).
    The ratio is capped at 1.0 because SLAM may classify slightly more cells as known
    than the geometry-based estimate predicts.

    Args:
        msg:            Latest OccupancyGrid message from /map.
        expected_cells: Pre-calculated traversable cell count (from load_expected_cells).

    Returns
    -------
        (coverage_ratio, known_cell_count)

    """
    if expected_cells <= 0:
        return 0.0, 0

    data = np.asarray(msg.data, dtype=np.int8)
    # Only count free cells (0–25) – occupied cells (walls) must not inflate the
    # ratio, because the denominator expected_cells counts only traversable area.
    known_free = int(np.sum((data >= 0) & (data <= 25)))
    coverage = min(1.0, known_free / expected_cells)
    return coverage, known_free
