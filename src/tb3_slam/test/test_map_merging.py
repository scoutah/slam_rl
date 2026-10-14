"""Tests for fusing the robots' grids into one global grid."""
import numpy as np
import pytest

from tb3_slam.map_merging import fuse_grids, GridSpec, merged_grid_spec, UNKNOWN

RES = 0.05


def _spec(cells: int = 20) -> GridSpec:
    return GridSpec(cells, cells, 0.0, 0.0, RES)


def test_merged_grid_covers_world_plus_margin():
    spec = merged_grid_spec(8.0)
    assert spec.origin_x == spec.origin_y == -5.0
    assert spec.width * spec.resolution >= 10.0
    assert spec.width == spec.height


def test_unknown_only_where_no_robot_knows():
    a = np.full((4, 4), UNKNOWN, np.int8)
    a[0, 0] = 0
    b = np.full((4, 4), UNKNOWN, np.int8)
    b[1, 1] = 100
    merged = fuse_grids(_spec(), [(a, 0.0, 0.0, RES), (b, 0.0, 0.0, RES)])
    assert merged[0, 0] == 0
    assert merged[1, 1] == 100
    assert merged[2, 2] == UNKNOWN


def test_wall_and_uncertain_beat_free_and_values_pass_through():
    a = np.array([[0, 0, 0]], np.int8)
    b = np.array([[100, 40, UNKNOWN]], np.int8)
    merged = fuse_grids(_spec(), [(a, 0.0, 0.0, RES), (b, 0.0, 0.0, RES)])
    assert merged[0, :3].tolist() == [100, 40, 0]


def test_origin_offset_shifts_by_whole_cells():
    data = np.full((2, 2), 100, np.int8)
    # one metre east and half a metre north = 20 and 10 cells
    merged = fuse_grids(_spec(40), [(data, 1.0, 0.5, RES)])
    assert merged[10, 20] == 100
    assert merged[11, 21] == 100
    assert merged[9, 20] == UNKNOWN
    assert merged[10, 19] == UNKNOWN
    assert int((merged != UNKNOWN).sum()) == 4


def test_parts_outside_the_grid_are_clipped():
    data = np.full((6, 6), 0, np.int8)
    # 3 cells left of the grid and 7 up: columns 0-2 and rows 7-9 stay visible
    merged = fuse_grids(_spec(10), [(data, -0.15, 0.35, RES)])
    assert int((merged == 0).sum()) == 3 * 3
    assert merged[7:, :3].min() == 0
    fully_outside = fuse_grids(_spec(10), [(data, 5.0, 5.0, RES)])
    assert int((fully_outside != UNKNOWN).sum()) == 0


def test_resolution_mismatch_is_rejected():
    with pytest.raises(ValueError):
        fuse_grids(_spec(), [(np.zeros((2, 2), np.int8), 0.0, 0.0, 0.1)])
