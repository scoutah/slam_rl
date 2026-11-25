"""Tests for the robot collision guards: lidar speed cap and shared-goal filter."""
import math

from tb3_slam.cmd_vel_relay import clearance_ahead, limit_speed
from tb3_slam.multi_robot_explorer import Goal, goal_conflicts


def _scan(hits: dict[float, float], n: int = 360):
    """Return (ranges, angle_min, increment) with returns ``{angle_deg: range}``."""
    inc = 2 * math.pi / n
    ranges = [math.inf] * n
    for deg, r in hits.items():
        ranges[round(math.radians(deg) / inc) % n] = r
    return ranges, 0.0, inc


def test_clearance_sees_obstacle_straight_ahead():
    ranges, a0, inc = _scan({0.0: 0.5})
    assert clearance_ahead(ranges, a0, inc, 1.0, 0.17, 0.12, 3.5) == 0.5


def test_clearance_ignores_obstacle_beside_and_behind():
    ranges, a0, inc = _scan({90.0: 0.3, 180.0: 0.3, 40.0: 0.6})
    assert clearance_ahead(ranges, a0, inc, 1.0, 0.17, 0.12, 3.5) is None
    assert clearance_ahead(ranges, a0, inc, -1.0, 0.17, 0.12, 3.5) == 0.3


def test_limit_speed_stops_inside_stop_distance_and_keeps_sign():
    assert limit_speed(0.2, 0.2, 0.2, 0.5) == 0.0
    assert limit_speed(-0.2, 0.1, 0.2, 0.5) == 0.0


def test_limit_speed_slows_before_obstacle_and_passes_clear_corridor():
    slowed = limit_speed(0.22, 0.23, 0.2, 0.5)
    assert 0.0 < slowed < 0.22
    assert limit_speed(0.22, None, 0.2, 0.5) == 0.22
    assert limit_speed(0.22, 3.0, 0.2, 0.5) == 0.22


def test_goal_conflicts_within_separation_only():
    taken = [Goal(0.0, 0.0)]
    assert goal_conflicts(Goal(1.0, 0.0), taken, 1.5)
    assert not goal_conflicts(Goal(2.0, 0.0), taken, 1.5)
    assert not goal_conflicts(Goal(0.0, 0.0), [], 1.5)
