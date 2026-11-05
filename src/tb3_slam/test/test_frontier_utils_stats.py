"""Tests for centroids_from_map_debug stats dict."""
from nav_msgs.msg import OccupancyGrid
import numpy as np

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
    # Five rows: a frontier line shorter than MIN_GROUP_CELLS is filtered.
    grid = [[50, 0, 0, 0] for _ in range(5)]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(2, 2))

    assert 'frontiers' in stats, 'stats dict must include frontiers key'
    assert set(stats['frontiers']) == {(r, 1) for r in range(5)}


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


def test_free_cell_with_second_degree_wall_is_not_a_frontier():
    # Col 1 FREE cells touch UNCERTAIN at col 0; walls sit at col 3,
    # exactly 2 cells away. The 5×5 wall exclusion must reject them.
    grid = [
        [50, 0, 0, 95],
        [50, 0, 0, 95],
        [50, 0, 0, 95],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 1))

    # Col 1 rejected (wall 2 cells away at col 3). Col 2 has no UNCERTAIN
    # neighbour. So nothing qualifies.
    assert stats['frontiers'] == []


def test_free_cell_with_wall_three_cells_away_is_a_frontier():
    # Walls now 3 cells away at col 4 — outside the 5×5 window. Col 1
    # FREE cells should qualify as frontiers again (UNKNOWN behind col 0
    # keeps the group from being a phantom).
    grid = [[-1, 50, 0, 0, 0, 95] for _ in range(5)]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(2, 2))

    assert set(stats['frontiers']) == {(r, 2) for r in range(5)}


def test_wall_sandwiched_uncertain_band_is_phantom():
    # UNCERTAIN band between FREE and WALL, no UNKNOWN anywhere near.
    grid = [[0, 0, 0, 0, 50, 0, 95, 95] for _ in range(8)]
    _, stats = centroids_from_map_debug(_make_grid(grid), robot_grid_pos=(4, 0))

    assert stats['frontiers'] == []
    assert stats['n_phantom_frontiers'] > 0


def test_tiny_frontier_group_is_phantom():
    grid = [[-1, 50, 0, 0, 0, 0, 0, 0] for _ in range(3)]
    _, stats = centroids_from_map_debug(_make_grid(grid), robot_grid_pos=(1, 5))

    assert stats['frontiers'] == []
    assert stats['n_phantom_frontiers'] == 3


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
