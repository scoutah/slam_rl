"""Regression tests for goals across blocked or narrow passages (issue #21)."""
from collections import deque

from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

from tb3_slam.frontier_utils import centroids_from_map, find_centroid


def _rooms(corridor_width):
    grid = np.full((80, 100), 95, dtype=np.int8)
    grid[10:70, 5:40] = 0
    grid[10:70, 60:90] = 0
    start = 40 - corridor_width // 2
    grid[start:start + corridor_width, 40:60] = 0
    grid[10:70, 90] = 50
    grid[:, 91:] = -1
    return grid


def _reachable(grid, start):
    """Independent cardinal FREE-cell traversal for checking emitted goals."""
    seen = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for cell in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            r, c = cell
            if (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                    and 0 <= grid[r, c] <= 25 and cell not in seen):
                seen.add(cell)
                queue.append(cell)
    return seen


@pytest.mark.parametrize('barrier', [-1, 50, 95])
def test_centroid_never_crosses_nonfree_barrier(barrier):
    """A target cut off from the robot by non-FREE cells yields no goal at all."""
    grid = _rooms(20)
    grid[:, 45:55] = barrier
    robot = (40, 15)
    assert find_centroid([(row, 88) for row in range(25, 56)], grid, robot) is None


def test_narrow_corridor_gets_safe_approach_goal_instead_of_far_room():
    grid = _rooms(5)  # 0.25 m: not traversable with Nav2's 0.22 m robot radius.
    goal = find_centroid([(row, 88) for row in range(25, 56)], grid, (40, 15))
    assert goal is not None
    assert goal[1] < 40
    r, c = goal
    assert np.all(grid[r - 5:r + 6, c - 5:c + 6] == 0)


def test_wide_corridor_retains_goal_in_far_room():
    grid = _rooms(20)  # 1.0 m: sufficient space for both sides of the footprint.
    goal = find_centroid([(row, 88) for row in range(25, 56)], grid, (40, 15))
    assert goal is not None
    assert goal[1] > 60
    assert goal in _reachable(grid, (40, 15))


@pytest.mark.parametrize('robot', [None, (-1, 15), (40, 110), (0, 0)])
def test_no_goal_without_a_valid_free_robot_position(robot):
    assert find_centroid([(40, 88)], _rooms(20), robot) is None


def test_pipeline_moves_goal_before_narrow_corridor():
    grid = _rooms(5)
    msg = OccupancyGrid()
    msg.info.height, msg.info.width = grid.shape
    msg.info.resolution = 0.05
    msg.data = grid.ravel().tolist()
    goals = centroids_from_map(msg, robot_grid_pos=(40, 15))
    assert goals
    assert all(col < 40 for _, col in goals)


def test_goal_keeps_minimum_motion_distance():
    grid = np.zeros((20, 20), dtype=np.int8)
    assert find_centroid([(10, 11)], grid, (10, 10)) is None


def test_free_robot_can_leave_incompletely_mapped_current_footprint():
    grid = _rooms(20)
    robot = (40, 7)  # FREE, but its clearance neighbourhood touches the boundary.
    goal = find_centroid([(40, 88)], grid, robot)
    assert goal is not None
    assert goal in _reachable(grid, robot)
    r, c = goal
    assert np.all(grid[r - 5:r + 6, c - 5:c + 6] == 0)


def test_start_in_long_narrow_corridor_does_not_jump_to_another_safe_region():
    assert find_centroid([(40, 88)], _rooms(5), (40, 50)) is None


def test_robot_on_nonfree_cell_with_free_ahead_still_gets_goal():
    """A robot whose own footprint reads wall (spawn bleeding) still explores.

    r2 spawns against the south wall; under the lowered WALL_MIN its cell can
    read >=75 while free space lies just ahead. Seeding from the free
    neighbours (not giving up) must still yield a reachable goal.
    """
    grid = _rooms(20)
    grid[40, 15] = 80  # robot cell now reads as wall
    goal = find_centroid([(40, 88)], grid, (40, 15))
    assert goal is not None
    assert goal in _reachable(grid, (41, 15))


def _rooms_with_far_narrow_door():
    """Room A (robot) and room B joined only by a narrow door at the top of the wall."""
    grid = np.full((80, 100), 95, dtype=np.int8)
    grid[10:70, 5:40] = 0
    grid[10:70, 46:90] = 0
    grid[12:17, 40:46] = 0  # 0.25 m door: not traversable, but a FREE-cell path
    return grid


def test_goal_is_pushed_back_along_the_route_not_across_the_wall():
    """Frontier in B hugs the wall; the nearest reachable cell is in A, behind the wall.

    The goal must instead be the last reachable cell on the route to the
    frontier, i.e. in front of the door, not the A cell next to the frontier.
    """
    grid = _rooms_with_far_narrow_door()
    robot = (40, 10)
    goal = find_centroid([(60, 47)], grid, robot)
    assert goal is not None
    assert goal[1] < 40
    assert goal[0] < 25  # near the door, far from the A cells beside the frontier
    assert goal in _reachable(grid, robot)
