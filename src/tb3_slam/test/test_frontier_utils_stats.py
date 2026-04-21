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
    grid = [
        [-1, 0, 0, 0],
        [-1, 0, 0, 0],
        [-1, 0, 0, 0],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 2))

    assert 'frontiers' in stats, 'stats dict must include frontiers key'
    assert set(stats['frontiers']) == {(0, 0), (1, 0), (2, 0)}


def test_stats_frontiers_empty_when_no_unknown():
    grid = [[0, 0], [0, 0]]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(0, 0))

    assert stats['frontiers'] == []
