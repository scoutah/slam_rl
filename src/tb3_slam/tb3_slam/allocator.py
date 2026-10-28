"""Shared allocator interface for multi-robot exploration (issue #11).

The classical and the RL exploration methods must differ **only** in the
allocator behind one shared interface, so both can run on an otherwise
identical stack. This module defines that interface and the data types that
flow through it. Concrete allocators live in their own modules (e.g. the dummy
nearest-frontier allocator in ``nearest_frontier_allocator.py``).

Interface (conceptually ``(merged_map, poses) -> (goal_r1, goal_r2)``):

    allocator.allocate(context) -> dict[str, Goal | None]

``allocate`` receives one :class:`AllocationContext` bundling everything the
decision needs — the merged ``OccupancyGrid`` (frame ``map``), the current
state of every robot, and an optional Nav2 path-cost callback — and returns one
goal per robot (or ``None`` for a robot that should keep its current goal / has
nothing to do). Bundling the inputs in a context object (rather than a growing
positional signature) is deliberate: #12 needs path costs and #13 needs
team-aware state, and further methods can add fields to the context without
touching every allocator's signature.

Allocators are pure decision functions: they never talk to ROS, so they can be
unit tested and swapped freely. The coordinator node (``coordinator.py``) owns
all ROS I/O — map subscription, TF lookups, Nav2 dispatch, the path-cost
callback and failure handling — and calls the allocator each decision cycle.

Frontier detection and grouping are reused from ``frontier_utils`` so every
method sees identical candidates; only the *selection* among them differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from nav_msgs.msg import OccupancyGrid

from tb3_slam.frontier_utils import (
    centroids_from_map_debug,
    grid_to_world,
    world_to_grid,
)


@dataclass(frozen=True)
class RobotState:
    """Current state of one robot as seen by the allocator.

    ``name`` is the robot namespace (e.g. ``r1``). ``x``/``y`` are the robot's
    position in the merged ``map`` frame (metres). ``busy`` is True while the
    robot already has an active Nav2 goal it is still pursuing, which lets an
    allocator leave a moving robot alone instead of re-tasking it every cycle.
    ``current_goal`` is the goal that robot is currently pursuing (``None`` when
    idle), so an allocator can spread robots out by reducing the utility of
    frontiers near another robot's assigned goal (#12) or feed a teammate's
    target into its features (#13).
    """

    name: str
    x: float
    y: float
    busy: bool = False
    current_goal: 'Goal | None' = None


@dataclass(frozen=True)
class Goal:
    """A navigation goal in the merged ``map`` frame (metres).

    ``group_size`` and ``path_len`` carry the frontier features the goal was
    derived from so an allocator (or logging) can inspect them; they are
    optional and default to ``None`` when not computed.
    """

    x: float
    y: float
    group_size: int | None = None
    path_len: float | None = None


@dataclass
class FrontierCandidate:
    """One navigable frontier centroid shared across allocators.

    ``cell`` is the grid (row, col) on the merged map; ``x``/``y`` the same
    point in world metres; ``group_size`` the number of frontier cells behind
    it. Candidates are computed once per decision cycle (per robot seed) by
    :func:`candidates_for_robot` and handed to the allocator.
    """

    cell: tuple[int, int]
    x: float
    y: float
    group_size: int


#: A Nav2 path-cost callback: given a robot namespace and a goal point in the
#: merged ``map`` frame (metres), return the planned path length in metres from
#: that robot's current pose to the goal, or ``None`` if no plan is available
#: (goal unreachable, planner not ready, or not yet computed this cycle).
#: Supplied by the coordinator (backed by ``compute_path_to_pose``); ``None`` on
#: the context means no planner is available and allocators must fall back to a
#: geometric estimate. Keying on the robot name (not an explicit start) lets the
#: coordinator own the start-pose lookup and cache plans per robot.
CostFn = Callable[[str, float, float], float | None]


@dataclass
class AllocationContext:
    """Everything an allocator needs for one decision cycle.

    Bundling the inputs here keeps the :meth:`Allocator.allocate` signature
    stable as methods grow: #12 reads ``cost_fn`` for Nav2 path lengths, #13
    reads teammate state off each :class:`RobotState`, and future methods can
    add fields without breaking existing allocators.

    ``merged_map`` is the fused ``OccupancyGrid`` (frame ``map``). ``robots`` is
    the state of every robot. ``cost_fn`` is an optional Nav2 path-cost callback
    (see :data:`CostFn`); it is ``None`` when no planner is available, in which
    case allocators fall back to Euclidean distance.
    """

    merged_map: OccupancyGrid
    robots: list[RobotState]
    cost_fn: CostFn | None = None

    def candidates_for(self, robot: RobotState) -> list['FrontierCandidate']:
        """Frontier centroids reachable from ``robot`` on the merged map.

        Convenience wrapper over :func:`candidates_for_robot` so an allocator
        does not need to reach past the context for the shared candidate set.
        """
        return candidates_for_robot(self.merged_map, robot)

    def path_cost(self, robot: RobotState, x: float, y: float) -> float | None:
        """Nav2 path length from ``robot`` to ``(x, y)``, or ``None`` if unknown.

        Returns ``None`` when no planner is available (``cost_fn`` unset) or the
        plan is not ready / the goal is unreachable, so an allocator can decide
        to fall back to Euclidean distance. Keeps the ``cost_fn`` null-check in
        one place.
        """
        if self.cost_fn is None:
            return None
        return self.cost_fn(robot.name, x, y)


def candidates_for_robot(
    merged_map: OccupancyGrid,
    robot: RobotState,
) -> list['FrontierCandidate']:
    """Return the navigable frontier centroids reachable from ``robot``.

    Thin wrapper over :func:`frontier_utils.centroids_from_map_debug` seeded
    from the robot's grid cell, so each robot sees the frontiers reachable from
    where it stands on the *merged* map. Shared by all allocators to guarantee
    identical candidate sets.
    """
    robot_grid_pos = world_to_grid(robot.x, robot.y, merged_map)
    centroids, stats = centroids_from_map_debug(merged_map, robot_grid_pos=robot_grid_pos)
    sizes = stats.get('centroid_group_sizes', [])
    candidates: list[FrontierCandidate] = []
    for i, cell in enumerate(centroids):
        wx, wy = grid_to_world(cell, merged_map)
        group_size = sizes[i] if i < len(sizes) else 0
        candidates.append(FrontierCandidate(cell=cell, x=wx, y=wy, group_size=group_size))
    return candidates


class Allocator(Protocol):
    """Decision function mapping one :class:`AllocationContext` to per-robot goals.

    Implementations must be pure (no ROS I/O, no global state beyond their own
    learned parameters) so the classical and RL methods are drop-in behind this
    one interface. ``allocate`` returns a mapping from robot name to a
    :class:`Goal`, or ``None`` for a robot that should keep doing what it is
    doing (e.g. it is still busy, or no frontier is available for it).
    """

    def allocate(self, context: AllocationContext) -> dict[str, Goal | None]:
        """Return a goal (or ``None``) for every robot in ``context.robots``."""
        ...
