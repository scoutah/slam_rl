"""Robot bodies observed by teammates must not trap their own map seed."""
import numpy as np

from tb3_slam.map_merging import clear_robot_footprints, GridSpec


def test_teammate_return_at_current_robot_pose_is_free_but_real_walls_remain():
    spec = GridSpec(20, 20, -0.5, -0.5, 0.05)
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[9:12, 9:12] = 75
    grid[:, 1] = 100
    clear_robot_footprints(grid, spec, [(0.025, 0.025)])
    assert np.all(grid[9:12, 9:12] == 0)
    assert np.all(grid[:, 1] == 100)


def test_missing_or_out_of_bounds_robot_poses_do_not_clear_anything():
    spec = GridSpec(10, 10, 0.0, 0.0, 0.05)
    grid = np.full((10, 10), 75, dtype=np.int8)
    clear_robot_footprints(grid, spec, [])
    assert np.all(grid == 75)
    clear_robot_footprints(grid, spec, [(-5.0, 2.0), (10.0, -10.0)])
    assert np.all(grid == 75)
