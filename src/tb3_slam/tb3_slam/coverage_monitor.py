"""
Shared coverage utilities for exploration nodes.

Both frontier_explorer and rl_agent import check_coverage() and load_expected_cells()
to monitor how much of the expected traversable area has been mapped.
The expected cell count is pre-calculated by generate_random_world() and stored in
WORLD_META_FILE so nodes can read it regardless of how they are launched.
"""
import json
from pathlib import Path

import numpy as np
from nav_msgs.msg import OccupancyGrid

from tb3_slam.utils import WORLD_META_FILE

DEFAULT_COVERAGE_THRESHOLD: float = 0.99


def load_expected_cells() -> int | None:
    """
    Read the expected number of traversable cells from the world metadata file
    written by generate_random_world().

    Returns None if the file does not exist or cannot be parsed – callers should
    fall back to a sensible default (e.g. total map size) and log a warning.
    """
    if not WORLD_META_FILE.exists():
        return None
    try:
        return int(json.loads(WORLD_META_FILE.read_text())['expected_traversable_cells'])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def check_coverage(msg: OccupancyGrid, expected_cells: int) -> tuple[float, int]:
    """
    Compute exploration coverage against the pre-calculated traversable cell count.

    A cell counts as "discovered" when its occupancy value is >= 0 (i.e. not unknown).
    The ratio is capped at 1.0 because SLAM may classify slightly more cells as known
    than the geometry-based estimate predicts.

    Args:
        msg:            Latest OccupancyGrid message from /map.
        expected_cells: Pre-calculated traversable cell count (from load_expected_cells).

    Returns:
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
