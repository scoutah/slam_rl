"""Build RViz markers for detected frontier cells and the current Nav2 goal.

Kept separate from ``frontier_utils`` so the pure-computation module does not
pull in ``visualization_msgs``/``geometry_msgs``.
"""
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from tb3_slam.frontier_utils import grid_to_world

_NAMESPACE = 'frontier_viz'
_FRONTIER_ID = 0
_GOAL_ID = 1


def _new_marker(
    marker_id: int,
    marker_type: int,
    frame_id: str,
    stamp: Time | None,
) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = _NAMESPACE
    m.id = marker_id
    m.type = marker_type
    m.pose.orientation.w = 1.0
    return m


def build_frontier_markers(
    frontiers: list[tuple[int, int]],
    goal: tuple[int, int] | None,
    map_msg: OccupancyGrid,
    frame_id: str = 'map',
    stamp: Time | None = None,
) -> MarkerArray:
    """Return a MarkerArray with a red CUBE_LIST for frontiers + blue CUBE for the goal.

    The returned array always contains exactly two markers (ids 0 and 1). When
    ``goal`` is ``None`` the goal marker uses ``Marker.DELETE`` so the previous
    blue cube disappears in RViz.
    """
    res = map_msg.info.resolution

    frontier_marker = _new_marker(_FRONTIER_ID, Marker.CUBE_LIST, frame_id, stamp)
    frontier_marker.action = Marker.ADD
    frontier_marker.scale.x = res
    frontier_marker.scale.y = res
    frontier_marker.scale.z = 0.02
    frontier_marker.color.r = 1.0
    frontier_marker.color.g = 0.0
    frontier_marker.color.b = 0.0
    frontier_marker.color.a = 0.7
    for cell in frontiers:
        x, y = grid_to_world(cell, map_msg)
        p = Point()
        p.x = x
        p.y = y
        p.z = 0.0
        frontier_marker.points.append(p)

    goal_marker = _new_marker(_GOAL_ID, Marker.CUBE, frame_id, stamp)
    if goal is None:
        goal_marker.action = Marker.DELETE
    else:
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 1.5 * res
        goal_marker.scale.y = 1.5 * res
        goal_marker.scale.z = 0.05
        goal_marker.color.r = 0.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        goal_marker.color.a = 0.9
        gx, gy = grid_to_world(goal, map_msg)
        goal_marker.pose.position.x = gx
        goal_marker.pose.position.y = gy
        goal_marker.pose.position.z = 0.025

    arr = MarkerArray()
    arr.markers = [frontier_marker, goal_marker]
    return arr
