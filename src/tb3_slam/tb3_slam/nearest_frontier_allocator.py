"""Dummy nearest-frontier allocator (issue #11).

A minimal allocator used to validate the coordinator plumbing: each idle robot
is assigned its own nearest frontier centroid. It is intentionally dumb — no
deconfliction between robots, no Nav2 path costs — so it exercises the shared
``Allocator`` interface end to end without standing in for the real classical
(#12) or RL (#13) allocators.
"""
from __future__ import annotations

from dataclasses import dataclass

from tb3_slam.allocator import AllocationContext, Goal


@dataclass
class NearestFrontierAllocator:
    """Each idle robot gets its own nearest frontier.

    "Nearest" is Euclidean distance on the merged map (no Nav2 plan), which is
    enough to validate the coordinator plumbing per the issue's acceptance
    criterion. Two robots may be assigned the same frontier — deconfliction is
    a concern of the real classical/RL allocators, not of this stub. A robot
    that is still ``busy`` keeps its current goal (returns ``None``), and a
    robot with no reachable frontier also returns ``None``.
    """

    #: Re-task even robots that are already busy. Off by default so the dummy
    #: does not thrash a moving robot every cycle.
    retask_busy: bool = False

    def allocate(self, context: AllocationContext) -> dict[str, Goal | None]:
        """Assign each idle robot the nearest reachable frontier centroid."""
        goals: dict[str, Goal | None] = {}
        for robot in context.robots:
            if robot.busy and not self.retask_busy:
                goals[robot.name] = None
                continue
            candidates = context.candidates_for(robot)
            if not candidates:
                goals[robot.name] = None
                continue
            nearest = min(
                candidates,
                key=lambda c: (c.x - robot.x) ** 2 + (c.y - robot.y) ** 2,
            )
            goals[robot.name] = Goal(
                x=nearest.x,
                y=nearest.y,
                group_size=nearest.group_size,
            )
        return goals
