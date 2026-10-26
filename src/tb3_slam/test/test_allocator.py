"""Tests for the shared allocator interface and the dummy allocator (issue #11)."""
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

from tb3_slam.allocator import (
    AllocationContext,
    candidates_for_robot,
    Goal,
    RobotState,
)
from tb3_slam.allocator_registry import ALLOCATORS, make_allocator
from tb3_slam.nearest_frontier_allocator import NearestFrontierAllocator


def _map_with_two_frontiers() -> OccupancyGrid:
    """Build a wide free strip with uncertain (frontier) bands near both ends.

    FREE (0) everywhere in the strip, with a vertical band of UNCERTAIN cells
    (value 50) near the left and right ends so that FREE cells adjacent to them
    become frontiers. Walls (95) frame the strip. Robots placed near each end
    should get the frontier on their own side as "nearest".
    """
    grid = np.full((40, 120), 95, dtype=np.int8)
    grid[5:35, 5:115] = 0            # free strip
    grid[5:35, 10] = 50              # left frontier band
    grid[5:35, 109] = 50             # right frontier band
    msg = OccupancyGrid()
    msg.info.height, msg.info.width = grid.shape
    msg.info.resolution = 0.05
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.data = grid.ravel().tolist()
    return msg


def _world(cell_col: int) -> float:
    return (cell_col + 0.4999) * 0.05


def _context(msg: OccupancyGrid, robots: list[RobotState], cost_fn=None) -> AllocationContext:
    return AllocationContext(merged_map=msg, robots=robots, cost_fn=cost_fn)


def test_registry_knows_the_dummy_allocator():
    assert set(ALLOCATORS) >= {'dummy', 'nearest'}
    assert isinstance(make_allocator('dummy'), NearestFrontierAllocator)


def test_registry_maps_the_user_facing_methods():
    # exploration_mode:=classic|rl selects an allocator by the same name, so
    # the method difference lives entirely behind the shared interface.
    assert set(ALLOCATORS) >= {'classic', 'rl'}
    assert make_allocator('classic') is not None
    assert make_allocator('rl') is not None


def test_unknown_allocator_name_raises():
    with pytest.raises(ValueError):
        make_allocator('does-not-exist')


def test_make_allocator_ignores_kwargs_a_class_does_not_accept():
    # The coordinator may pass data_dir for the RL allocator; the dummy must
    # not choke on it so the shared factory can offer a config superset.
    allocator = make_allocator('dummy', data_dir='/tmp/whatever')
    assert isinstance(allocator, NearestFrontierAllocator)


def test_candidates_are_found_for_a_seeded_robot():
    msg = _map_with_two_frontiers()
    robot = RobotState('r1', x=_world(60), y=_world(20))
    candidates = candidates_for_robot(msg, robot)
    assert candidates
    assert all(c.group_size > 0 for c in candidates)


def test_context_candidates_for_matches_module_function():
    msg = _map_with_two_frontiers()
    robot = RobotState('r1', x=_world(60), y=_world(20))
    ctx = _context(msg, [robot])
    assert ctx.candidates_for(robot) == candidates_for_robot(msg, robot)


def test_context_path_cost_is_none_without_a_cost_fn():
    msg = _map_with_two_frontiers()
    robot = RobotState('r1', x=_world(60), y=_world(20))
    assert _context(msg, [robot]).path_cost(robot, 1.0, 1.0) is None


def test_context_path_cost_delegates_to_cost_fn():
    msg = _map_with_two_frontiers()
    robot = RobotState('r1', x=_world(60), y=_world(20))
    seen = {}

    def cost_fn(ns, x, y):
        seen['args'] = (ns, x, y)
        return 3.5

    ctx = _context(msg, [robot], cost_fn=cost_fn)
    assert ctx.path_cost(robot, 1.0, 2.0) == 3.5
    assert seen['args'] == ('r1', 1.0, 2.0)


def test_robot_state_carries_current_goal():
    goal = Goal(x=1.0, y=2.0, group_size=10)
    robot = RobotState('r1', x=0.0, y=0.0, busy=True, current_goal=goal)
    assert robot.current_goal is goal


def test_dummy_assigns_one_goal_per_idle_robot():
    msg = _map_with_two_frontiers()
    robots = [
        RobotState('r1', x=_world(25), y=_world(20)),   # near left frontier
        RobotState('r2', x=_world(95), y=_world(20)),   # near right frontier
    ]
    goals = make_allocator('dummy').allocate(_context(msg, robots))
    assert set(goals) == {'r1', 'r2'}
    assert isinstance(goals['r1'], Goal)
    assert isinstance(goals['r2'], Goal)
    # Each robot's nearest frontier is on its own side of the strip.
    assert goals['r1'].x < goals['r2'].x


def test_dummy_leaves_busy_robots_alone_by_default():
    msg = _map_with_two_frontiers()
    robots = [RobotState('r1', x=_world(25), y=_world(20), busy=True)]
    goals = NearestFrontierAllocator().allocate(_context(msg, robots))
    assert goals['r1'] is None


def test_dummy_retasks_busy_robots_when_configured():
    msg = _map_with_two_frontiers()
    robots = [RobotState('r1', x=_world(25), y=_world(20), busy=True)]
    goals = NearestFrontierAllocator(retask_busy=True).allocate(_context(msg, robots))
    assert isinstance(goals['r1'], Goal)


def test_dummy_returns_none_when_no_frontier_reachable():
    grid = np.full((40, 40), 95, dtype=np.int8)
    grid[5:35, 5:35] = 0  # fully explored room, no uncertain cells → no frontier
    msg = OccupancyGrid()
    msg.info.height, msg.info.width = grid.shape
    msg.info.resolution = 0.05
    msg.data = grid.ravel().tolist()
    robots = [RobotState('r1', _world(20), _world(20))]
    goals = make_allocator('dummy').allocate(_context(msg, robots))
    assert goals['r1'] is None
