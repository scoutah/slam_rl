"""Tests for the explorer's ROS-free path-cost helpers (path_cost.py)."""
import math
from types import SimpleNamespace

from tb3_slam.path_cost import cost_key, path_length


def _pose(x: float, y: float):
    return SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))


def _path(points):
    return SimpleNamespace(poses=[_pose(x, y) for x, y in points])


# ── cost_key ────────────────────────────────────────────────────────

def test_cost_key_rounds_to_two_decimals():
    assert cost_key('r1', 0.004, 1.006, 2.011, 3.0) == ('r1', 0.0, 1.01, 2.01, 3.0)


def test_cost_key_distinguishes_namespace():
    assert cost_key('r1', 0, 0, 1, 1) != cost_key('r2', 0, 0, 1, 1)


def test_cost_key_close_positions_collide():
    # Two starts within half a cm round to the same key (cache hit).
    assert cost_key('r1', 1.001, 2.002, 3, 4) == cost_key('r1', 1.004, 1.999, 3, 4)


# ── path_length ─────────────────────────────────────────────────────

def test_path_length_none_when_empty():
    assert path_length(_path([])) is None


def test_path_length_zero_for_single_pose():
    assert path_length(_path([(1.0, 1.0)])) == 0.0


def test_path_length_straight_line():
    assert path_length(_path([(0, 0), (3, 4)])) == 5.0


def test_path_length_multi_segment():
    length = path_length(_path([(0, 0), (0, 2), (2, 2)]))
    assert length == 4.0


def test_path_length_diagonal_accumulates():
    length = path_length(_path([(0, 0), (1, 1), (2, 2)]))
    assert length == math.hypot(1, 1) * 2


def test_path_length_missing_attr_returns_none():
    assert path_length(SimpleNamespace()) is None
