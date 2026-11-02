"""ROS-free helpers for the explorer's Nav2 path-cost cache.

Kept separate from the rclpy node so this pure logic is importable and
unit-testable without ROS: the cache key and the polyline length of a plan.
path_length duck-types a Nav2 Path (``path.poses[i].pose.position.{x,y}``).
"""
from __future__ import annotations

import math


def cost_key(ns: str, sx: float, sy: float, gx: float, gy: float) -> tuple:
    """Cache key: namespace + start/goal rounded to the map resolution (5 cm)."""
    return (ns, round(sx, 2), round(sy, 2), round(gx, 2), round(gy, 2))


def path_length(path) -> float | None:
    """Total length of a Nav2 plan.

    Returns None when there is no usable plan (no poses), 0.0 for a single pose,
    and the summed segment length otherwise.
    """
    poses = getattr(path, 'poses', [])
    if len(poses) < 2:
        return 0.0 if poses else None
    total = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        total += math.hypot(
            b.pose.position.x - a.pose.position.x,
            b.pose.position.y - a.pose.position.y,
        )
    return total
