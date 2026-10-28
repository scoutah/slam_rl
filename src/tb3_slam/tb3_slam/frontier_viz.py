"""Build RViz markers for detected frontier cells and the current Nav2 goal.

Kept separate from ``frontier_utils`` so the pure-computation module does not
pull in ``visualization_msgs``/``geometry_msgs``.
"""
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from tb3_slam.frontier_utils import grid_to_world
from visualization_msgs.msg import Marker, MarkerArray

_NAMESPACE = 'frontier_viz'
_FRONTIER_ID = 0
_GOAL_ID = 1
_CENTROID_ID = 2


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
    centroids: list[tuple[int, int]] | None = None,
    goal_color: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> MarkerArray:
    """Return a MarkerArray with frontier cells, optional centroid cubes, and the goal.

    - id=0: red CUBE_LIST — frontier cells
    - id=1: CUBE for the current goal (``goal_color`` RGB, default blue). When
      ``goal`` is ``None`` the marker uses ``Marker.DELETE``.
    - id=2: blue CUBE_LIST for all candidate centroids. Only present when
      ``centroids`` is not ``None``; otherwise the id=2 slot is omitted.
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
        goal_marker.color.r = goal_color[0]
        goal_marker.color.g = goal_color[1]
        goal_marker.color.b = goal_color[2]
        goal_marker.color.a = 0.9
        gx, gy = grid_to_world(goal, map_msg)
        goal_marker.pose.position.x = gx
        goal_marker.pose.position.y = gy
        goal_marker.pose.position.z = 0.025

    arr = MarkerArray()
    arr.markers = [frontier_marker, goal_marker]

    if centroids is not None:
        centroid_marker = _new_marker(_CENTROID_ID, Marker.CUBE_LIST, frame_id, stamp)
        centroid_marker.action = Marker.ADD
        centroid_marker.scale.x = 1.2 * res
        centroid_marker.scale.y = 1.2 * res
        centroid_marker.scale.z = 0.03
        centroid_marker.color.r = 0.0
        centroid_marker.color.g = 0.0
        centroid_marker.color.b = 1.0
        centroid_marker.color.a = 0.8
        for cell in centroids:
            cx, cy = grid_to_world(cell, map_msg)
            p = Point()
            p.x = cx
            p.y = cy
            p.z = 0.015
            centroid_marker.points.append(p)
        arr.markers.append(centroid_marker)

    return arr
