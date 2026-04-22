"""Tests for find_groups connectivity rule."""
from tb3_slam.frontier_utils import find_groups


def _normalize(groups):
    return sorted([sorted(g) for g in groups], key=lambda g: (len(g), g))


def test_direct_neighbours_merge():
    # 8-connectivity: diagonal neighbours belong to the same group.
    points = [(0, 0), (1, 1), (2, 2)]
    result = find_groups(points)
    assert _normalize(result) == _normalize([[(0, 0), (1, 1), (2, 2)]])


def test_one_cell_gap_splits():
    # Chebyshev distance 2 is *not* enough to merge anymore: a one-cell
    # gap between cells must produce two separate groups (this is the
    # behaviour change that breaks the big ring into smaller clusters).
    points = [(0, 0), (0, 2)]
    result = find_groups(points)
    assert _normalize(result) == _normalize([[(0, 0)], [(0, 2)]])


def test_chain_with_gap_splits_into_two():
    # Two directly-connected pairs with a 1-cell gap between them.
    points = [(0, 0), (0, 1), (0, 3), (0, 4)]
    result = find_groups(points)
    assert _normalize(result) == _normalize([
        [(0, 0), (0, 1)],
        [(0, 3), (0, 4)],
    ])


def test_isolated_point_alone():
    points = [(0, 0), (0, 1), (5, 5)]
    result = find_groups(points)
    assert _normalize(result) == _normalize([
        [(0, 0), (0, 1)],
        [(5, 5)],
    ])
