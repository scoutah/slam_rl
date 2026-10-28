"""Boundary tests for the map-merge priority.

These pin the exact class thresholds frontier_utils shares (25 free, 26
uncertain, 74 uncertain, 75 wall); the existing test_map_merging.py only uses
interior values (0/40/50/100).
"""
import numpy as np

from tb3_slam.map_merging import (
    _priority,
    FREE_MAX,
    fuse_grids,
    GridSpec,
    UNCERTAIN_MAX,
    UNKNOWN,
)

RES = 0.05


def _spec(cells: int = 4) -> GridSpec:
    return GridSpec(cells, cells, 0.0, 0.0, RES)


def test_priority_class_boundaries():
    vals = np.array(
        [UNKNOWN, 0, FREE_MAX, FREE_MAX + 1, UNCERTAIN_MAX,
         UNCERTAIN_MAX + 1, 100],
        dtype=np.int8,
    )
    prio = _priority(vals)
    # unknown=0, uncertain=1, free=2, wall=3
    assert list(prio) == [0, 2, 2, 1, 1, 3, 3]


def test_priority_free_exactly_25_is_free_not_uncertain():
    assert _priority(np.array([25], np.int8))[0] == 2
    assert _priority(np.array([26], np.int8))[0] == 1


def test_priority_wall_exactly_75():
    assert _priority(np.array([74], np.int8))[0] == 1
    assert _priority(np.array([75], np.int8))[0] == 3


def test_fuse_wall_at_boundary_overrides_free():
    free = np.array([[FREE_MAX]], np.int8)
    wall = np.array([[75]], np.int8)
    merged = fuse_grids(_spec(1), [(free, 0.0, 0.0, RES), (wall, 0.0, 0.0, RES)])
    assert merged[0, 0] == 75


def test_fuse_free_boundary_sticky_over_uncertain():
    # FREE (25) must not be overwritten by a later UNCERTAIN (26).
    free = np.array([[FREE_MAX]], np.int8)
    uncertain = np.array([[26]], np.int8)
    merged = fuse_grids(
        _spec(1), [(free, 0.0, 0.0, RES), (uncertain, 0.0, 0.0, RES)],
    )
    assert merged[0, 0] == FREE_MAX


def test_fuse_equal_class_keeps_first_layer():
    # Two FREE observations of the same cell -> the first value is retained
    # (equal priority does not replace what is already there).
    a = np.array([[10]], np.int8)
    b = np.array([[20]], np.int8)
    merged = fuse_grids(_spec(1), [(a, 0.0, 0.0, RES), (b, 0.0, 0.0, RES)])
    assert merged[0, 0] == 10


def test_fuse_stateful_base_keeps_free_against_later_unknown():
    spec = _spec(1)
    first = fuse_grids(spec, [(np.array([[0]], np.int8), 0.0, 0.0, RES)])
    assert first[0, 0] == 0
    # A later cycle reports UNKNOWN for the same cell; stateful merge keeps FREE.
    second = fuse_grids(
        spec, [(np.array([[UNKNOWN]], np.int8), 0.0, 0.0, RES)], base=first,
    )
    assert second[0, 0] == 0
