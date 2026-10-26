"""Edge-case tests for frontier_utils helpers that were previously untested.

Covers the grid<->world coordinate conversions (including the last row/column
case that the 0.4999 offset exists to fix), world_to_grid out-of-bounds, and the
no-robot global-scan fallback branch of find_frontiers.
"""
from nav_msgs.msg import OccupancyGrid
import numpy as np

from tb3_slam.frontier_utils import (
    find_frontiers,
    grid_to_world,
    world_to_grid,
)


def _msg(width: int, height: int, ox: float = 0.0, oy: float = 0.0,
         res: float = 0.05) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.resolution = res
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = ox
    msg.info.origin.position.y = oy
    return msg


# ── grid_to_world ───────────────────────────────────────────────────

def test_grid_to_world_origin_cell():
    msg = _msg(10, 10)
    x, y = grid_to_world((0, 0), msg)
    assert x == (0.4999 * 0.05)
    assert y == (0.4999 * 0.05)


def test_grid_to_world_respects_origin_offset():
    msg = _msg(10, 10, ox=-1.0, oy=2.0)
    x, y = grid_to_world((0, 0), msg)
    assert x == -1.0 + 0.4999 * 0.05
    assert y == 2.0 + 0.4999 * 0.05


def test_grid_to_world_last_cell_stays_inside():
    # The 0.4999 offset exists so the last row/col maps strictly inside the grid
    # (round() and int() both return the same cell -> no Nav2 worldToMap failure).
    w, h, res = 10, 8, 0.05
    msg = _msg(w, h, res=res)
    x, y = grid_to_world((h - 1, w - 1), msg)
    assert x < w * res
    assert y < h * res
    # And it round-trips back to the same last cell.
    assert world_to_grid(x, y, msg) == (h - 1, w - 1)


def test_grid_to_world_round_trips_for_all_cells():
    msg = _msg(12, 9, ox=-0.3, oy=0.7)
    for r in range(msg.info.height):
        for c in range(msg.info.width):
            x, y = grid_to_world((r, c), msg)
            assert world_to_grid(x, y, msg) == (r, c)


# ── world_to_grid ───────────────────────────────────────────────────

def test_world_to_grid_inside():
    msg = _msg(10, 10)
    assert world_to_grid(0.0, 0.0, msg) == (0, 0)
    assert world_to_grid(0.11, 0.06, msg) == (1, 2)


def test_world_to_grid_out_of_bounds_returns_none():
    msg = _msg(10, 10)
    assert world_to_grid(-1.0, 0.0, msg) is None   # left of origin
    assert world_to_grid(0.0, -1.0, msg) is None   # below origin
    assert world_to_grid(100.0, 0.0, msg) is None  # past right edge
    assert world_to_grid(0.0, 100.0, msg) is None  # past top edge


def test_world_to_grid_honours_origin():
    msg = _msg(10, 10, ox=5.0, oy=5.0)
    assert world_to_grid(5.0, 5.0, msg) == (0, 0)
    assert world_to_grid(4.9, 5.0, msg) is None


# ── find_frontiers no-robot global-scan fallback ────────────────────

def test_find_frontiers_global_scan_without_robot():
    # Column 0 uncertain, columns 1..3 free. With no robot position the function
    # falls back to a full grid scan and returns every qualifying FREE cell.
    grid = np.array([
        [50, 0, 0, 0],
        [50, 0, 0, 0],
        [50, 0, 0, 0],
    ], dtype=np.int8)
    frontiers = find_frontiers(grid, robot_grid_pos=None)
    assert set(frontiers) == {(0, 1), (1, 1), (2, 1)}


def test_find_frontiers_global_scan_excludes_wall_neighbours():
    # A wall within the 5x5 window disqualifies the frontier even in global scan.
    grid = np.array([
        [50, 0, 100],
        [50, 0, 100],
        [50, 0, 100],
    ], dtype=np.int8)
    assert find_frontiers(grid, robot_grid_pos=None) == []


def test_find_frontiers_empty_when_no_uncertain():
    grid = np.zeros((4, 4), dtype=np.int8)
    assert find_frontiers(grid, robot_grid_pos=None) == []
