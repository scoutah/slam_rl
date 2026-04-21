"""Shared frontier detection utilities used by all exploration strategies."""
import math
from collections import deque

import numpy as np
from nav_msgs.msg import OccupancyGrid

# Occupancy value ranges
FREE_MAX = 25        # cell value 0–25   → free
FRONTIER_MIN = 26    # cell value 26–89  → uncertain / frontier
FRONTIER_MAX = 89
WALL_MIN = 90        # cell value 90–100 → wall
UNKNOWN = -1


def find_frontiers(
    grid: np.ndarray,
    free_seeds: set[tuple[int, int]] | None = None,
    robot_grid_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return all frontier cells (row, col) reachable from the robot.

    When ``robot_grid_pos`` is given, runs a BFS through FREE cells starting at
    the robot. Any UNKNOWN (-1) cell adjacent to a BFS-reachable FREE cell is
    a frontier. Only UNKNOWN cells are returned — UNCERTAIN (26–89) cells are
    NOT frontiers because walls in the SLAM map can have values in that range
    and must never be navigation targets.

    Without ``robot_grid_pos``, falls back to the original global scan.
    """
    rows, cols = grid.shape
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _seeds: set[tuple[int, int]] = free_seeds if free_seeds is not None else set()

    # ── BFS path ────────────────────────────────────────────────────────
    if robot_grid_pos is not None:
        sr, sc = robot_grid_pos
        if 0 <= sr < rows and 0 <= sc < cols:
            def _is_free(r: int, c: int) -> bool:
                return (0 <= grid[r, c] <= FREE_MAX) or ((r, c) in _seeds)

            visited: set[tuple[int, int]] = set()
            frontier_set: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque()

            if _is_free(sr, sc):
                queue.append((sr, sc))
                visited.add((sr, sc))
            else:
                # Robot in uncertain/unknown space; seed from adjacent free cells.
                for dr, dc in cardinal:
                    nr, nc = sr + dr, sc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and _is_free(nr, nc) and (nr, nc) not in visited):
                        queue.append((nr, nc))
                        visited.add((nr, nc))

            while queue:
                r, c = queue.popleft()
                for dr, dc in cardinal:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    nval = grid[nr, nc]
                    if nval == UNKNOWN:
                        frontier_set.add((nr, nc))
                    elif _is_free(nr, nc) and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            return list(frontier_set)

    # ── Fallback: global scan (no robot position available) ─────────────
    frontiers = []
    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            if cell == UNKNOWN:
                has_known = any(
                    0 <= r + dr < rows and 0 <= c + dc < cols
                    and 0 <= grid[r + dr, c + dc] < WALL_MIN
                    for dr, dc in cardinal
                )
                if has_known:
                    frontiers.append((r, c))
            elif FRONTIER_MIN <= cell <= FRONTIER_MAX:
                has_free = any(
                    0 <= r + dr < rows and 0 <= c + dc < cols
                    and (0 <= grid[r + dr, c + dc] <= FREE_MAX
                         or (r + dr, c + dc) in _seeds)
                    for dr, dc in cardinal
                )
                if has_free:
                    frontiers.append((r, c))
    return frontiers


def find_groups(frontiers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group frontier cells into connected clusters (2nd-degree connectivity, DFS).

    Two frontier cells belong to the same group if they are within Chebyshev
    distance 2 of each other (5×5 window).  This merges clusters separated by
    a one-cell gap that 8-connectivity would split.
    """
    frontier_set = set(frontiers)
    visited: set[tuple[int, int]] = set()
    groups = []
    offsets = [
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if (dr, dc) != (0, 0)
    ]

    for start in frontier_set:
        if start in visited:
            continue

        group = []
        stack = [start]
        visited.add(start)

        while stack:
            p = stack.pop()
            group.append(p)
            r, c = p
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if q in frontier_set and q not in visited:
                    visited.add(q)
                    stack.append(q)

        groups.append(group)

    return groups


def find_centroid(
    group: list[tuple[int, int]], grid: np.ndarray
) -> tuple[int, int] | None:
    """Return a navigable (free) cell near the geometric centroid of a frontier group.

    Strategy:
    1. Compute geometric centroid of the group.
    2. If the centroid itself is free, return it.
    3. Otherwise, find the nearest frontier cell to the centroid.
    4. Return the first free cardinal neighbor of that cell.
    5. Return None if no free neighbor exists.
    """
    if not group:
        return None

    rows, cols = grid.shape
    n = len(group)
    cr = sum(r for r, _ in group) // n
    cc = sum(c for _, c in group) // n

    # Small margin to avoid placing goals exactly at the map boundary.
    # The offset 0.4999 in grid_to_world already prevents Nav2's round()
    # rounding a boundary cell to size+1, so MARGIN=2 is sufficient.
    MARGIN = 2

    def _is_valid_free(r: int, c: int) -> bool:
        return (MARGIN <= r < rows - MARGIN
                and MARGIN <= c < cols - MARGIN
                and 0 <= grid[r, c] <= FREE_MAX)

    # Is the centroid itself free?
    if _is_valid_free(cr, cc):
        return (cr, cc)

    # Search ALL frontier cells' cardinal neighbors for the free cell
    # closest to the centroid.  Checking only the single nearest frontier
    # cell misses valid goals when that cell's neighbors are uncertain or
    # in the margin zone.
    best: tuple[int, int] | None = None
    best_dist = float('inf')
    for fr, fc in group:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = fr + dr, fc + dc
            if _is_valid_free(nr, nc):
                dist = math.hypot(nr - cr, nc - cc)
                if dist < best_dist:
                    best_dist = dist
                    best = (nr, nc)

    return best


def centroids_from_map(
    msg: OccupancyGrid,
    robot_grid_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Run the full frontier pipeline and return navigable centroid cells (row, col).

    This is the single entry point used by both FrontierExplorer and RLExplorer.
    Pass ``robot_grid_pos`` (row, col) to seed frontier detection from the robot's
    known position, which fixes the startup delay when no free cells exist yet.
    """
    grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    free_seeds = {robot_grid_pos} if robot_grid_pos is not None else None
    frontiers = find_frontiers(grid, free_seeds=free_seeds, robot_grid_pos=robot_grid_pos)
    groups = find_groups(frontiers)
    result = []
    for group in groups:
        centroid = find_centroid(group, grid)
        if centroid is not None:
            result.append(centroid)
    return result


def grid_to_world(cell: tuple[int, int], msg: OccupancyGrid) -> tuple[float, float]:
    """Convert a grid cell (row, col) to world coordinates (x, y) in meters.

    Uses offset 0.4999 (not 0.5) because Nav2's NavfnPlanner uses round() when
    converting back – round(col + 0.5) = col + 1 (C++ rounds half away from 0),
    which shifts our goal one cell and fails on the last column/row with
    "worldToMap failed: mx=size_x". 0.4999 makes both round() and int() return
    col consistently. Visual shift: 0.005 mm at 0.05 m resolution – negligible.
    """
    row, col = cell
    res = msg.info.resolution
    x = msg.info.origin.position.x + (col + 0.4999) * res
    y = msg.info.origin.position.y + (row + 0.4999) * res
    return x, y


def world_to_grid(x: float, y: float, msg: OccupancyGrid) -> tuple[int, int] | None:
    """Convert world coordinates (x, y) in meters to a grid cell (row, col).

    Returns None if the position falls outside the grid boundaries.
    """
    res = msg.info.resolution
    col = int((x - msg.info.origin.position.x) / res)
    row = int((y - msg.info.origin.position.y) / res)
    if 0 <= row < msg.info.height and 0 <= col < msg.info.width:
        return (row, col)
    return None
