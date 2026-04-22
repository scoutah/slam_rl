"""Tests for centroids_from_map_debug stats dict."""
import numpy as np
from nav_msgs.msg import OccupancyGrid

from tb3_slam.frontier_utils import centroids_from_map_debug


def _make_grid(data: list[list[int]]) -> OccupancyGrid:
    arr = np.array(data, dtype=np.int8)
    msg = OccupancyGrid()
    msg.info.resolution = 0.05
    msg.info.width = arr.shape[1]
    msg.info.height = arr.shape[0]
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.data = arr.flatten().tolist()
    return msg


def test_stats_includes_raw_frontiers():
    # Column 0 is UNCERTAIN (50), rest is FREE. Frontiers are the column-1
    # FREE cells that touch the UNCERTAIN column.
    grid = [
        [50, 0, 0, 0],
        [50, 0, 0, 0],
        [50, 0, 0, 0],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 2))

    assert 'frontiers' in stats, 'stats dict must include frontiers key'
    assert set(stats['frontiers']) == {(0, 1), (1, 1), (2, 1)}


def test_stats_frontiers_empty_when_no_uncertain():
    grid = [[0, 0], [0, 0]]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(0, 0))

    assert stats['frontiers'] == []


def test_free_cell_with_wall_neighbour_is_not_a_frontier():
    # Col 1 FREE cells touch both UNCERTAIN (col 0) AND a WALL (col 2).
    # The wall neighbour disqualifies them, so no frontiers are expected.
    grid = [
        [50, 0, 95],
        [50, 0, 95],
        [50, 0, 95],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 1))

    assert stats['frontiers'] == []


def test_phantom_unknown_beyond_wall_is_not_a_frontier():
    # FREE room on the left; WALL in the middle; UNCERTAIN+UNKNOWN to the
    # right (simulating phantom territory behind a wall). The old rule would
    # flag the UNKNOWN cells; the new rule must not.
    grid = [
        [0, 0, 95, 50, -1],
        [0, 0, 95, 50, -1],
        [0, 0, 95, 50, -1],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 0))

    # No FREE cell on the left touches UNCERTAIN (50 is only next to the
    # wall, not to any FREE cell in col 1), so no frontiers expected.
    assert stats['frontiers'] == []

