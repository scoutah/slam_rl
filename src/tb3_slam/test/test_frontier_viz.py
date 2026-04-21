"""Tests for build_frontier_markers."""
import numpy as np
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from tb3_slam.frontier_viz import build_frontier_markers


def _make_map(resolution: float = 0.05, origin=(0.0, 0.0), size=(4, 4)) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.resolution = resolution
    msg.info.width = size[1]
    msg.info.height = size[0]
    msg.info.origin.position.x = origin[0]
    msg.info.origin.position.y = origin[1]
    msg.data = (np.zeros(size[0] * size[1], dtype=np.int8)).tolist()
    return msg


def test_markerarray_has_two_markers_with_ids_0_and_1():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    assert len(arr.markers) == 2
    ids = sorted(m.id for m in arr.markers)
    assert ids == [0, 1]


def test_empty_frontiers_yields_add_action_with_no_points():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert frontier_marker.type == Marker.CUBE_LIST
    assert frontier_marker.action == Marker.ADD
    assert list(frontier_marker.points) == []
    assert frontier_marker.color.r == 1.0
    assert frontier_marker.color.g == 0.0
    assert frontier_marker.color.b == 0.0
    assert frontier_marker.color.a == 0.7
    assert frontier_marker.scale.x == msg.info.resolution
    assert frontier_marker.scale.y == msg.info.resolution


def test_frontiers_become_cube_list_points():
    msg = _make_map(resolution=0.1, origin=(0.0, 0.0))
    arr = build_frontier_markers([(1, 2)], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert len(frontier_marker.points) == 1
    p = frontier_marker.points[0]
    assert abs(p.x - (2 + 0.4999) * 0.1) < 1e-6
    assert abs(p.y - (1 + 0.4999) * 0.1) < 1e-6


def test_goal_none_sets_delete_action():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    goal_marker = next(m for m in arr.markers if m.id == 1)
    assert goal_marker.action == Marker.DELETE


def test_goal_sets_blue_cube_at_world_position():
    msg = _make_map(resolution=0.1, origin=(0.0, 0.0))
    arr = build_frontier_markers([], (2, 3), msg)
    goal_marker = next(m for m in arr.markers if m.id == 1)
    assert goal_marker.type == Marker.CUBE
    assert goal_marker.action == Marker.ADD
    assert goal_marker.color.r == 0.0
    assert goal_marker.color.g == 0.0
    assert goal_marker.color.b == 1.0
    assert goal_marker.color.a == 0.9
    assert abs(goal_marker.scale.x - 1.5 * 0.1) < 1e-9
    assert abs(goal_marker.scale.y - 1.5 * 0.1) < 1e-9
    assert abs(goal_marker.pose.position.x - (3 + 0.4999) * 0.1) < 1e-6
    assert abs(goal_marker.pose.position.y - (2 + 0.4999) * 0.1) < 1e-6
    assert goal_marker.pose.orientation.w == 1.0


def test_frame_id_and_namespace():
    msg = _make_map()
    arr = build_frontier_markers([(0, 0)], (0, 0), msg, frame_id='custom_frame')
    for m in arr.markers:
        assert m.header.frame_id == 'custom_frame'
        assert m.ns == 'frontier_viz'


def test_frontier_marker_has_identity_orientation():
    msg = _make_map()
    arr = build_frontier_markers([(0, 0)], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert frontier_marker.pose.orientation.w == 1.0
