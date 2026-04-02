"""Shared frontier detection utilities used by all exploration strategies."""
import math

import numpy as np
from nav_msgs.msg import OccupancyGrid

# Occupancy value ranges
FREE_MAX = 25        # cell value 0–25  → free
FRONTIER_MIN = 26    # cell value 26–75 → uncertain / frontier
FRONTIER_MAX = 75
UNKNOWN = -1


def find_frontiers(
    grid: np.ndarray,
    free_seeds: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return all frontier cells (row, col) in the occupancy grid.

    A frontier cell is unknown or uncertain and has at least one free neighbor.
    Cells in ``free_seeds`` are treated as free even if their grid value is not
    in [0, FREE_MAX].  This allows bootstrapping frontier detection from a known
    robot position before Cartographer has decayed any cells to free.
    """
    rows, cols = grid.shape
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _seeds: set[tuple[int, int]] = free_seeds if free_seeds is not None else set()
    frontiers = []

    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            is_frontier = cell == UNKNOWN or FRONTIER_MIN <= cell <= FRONTIER_MAX
            if not is_frontier:
                continue
            for dr, dc in cardinal:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (0 <= grid[nr, nc] <= FREE_MAX or (nr, nc) in _seeds)):
                    frontiers.append((r, c))
                    break

    return frontiers


def find_groups(frontiers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group frontier cells into connected clusters (8-connectivity, DFS)."""
    frontier_set = set(frontiers)
    visited: set[tuple[int, int]] = set()
    groups = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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
            for dr, dc in neighbors:
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

    # Is the centroid itself free?
    if 0 <= cr < rows and 0 <= cc < cols and 0 <= grid[cr, cc] <= FREE_MAX:
        return (cr, cc)

    # Find frontier cell nearest to the centroid
    nearest = min(group, key=lambda p: math.hypot(p[0] - cr, p[1] - cc))

    # Return a free cardinal neighbor
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = nearest[0] + dr, nearest[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and 0 <= grid[nr, nc] <= FREE_MAX:
            return (nr, nc)

    return None


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
    frontiers = find_frontiers(grid, free_seeds=free_seeds)
    groups = find_groups(frontiers)
    result = []
    for group in groups:
        centroid = find_centroid(group, grid)
        if centroid is not None:
            result.append(centroid)
    return result


def grid_to_world(cell: tuple[int, int], msg: OccupancyGrid) -> tuple[float, float]:
    """Convert a grid cell (row, col) to world coordinates (x, y) in meters."""
    row, col = cell
    res = msg.info.resolution
    x = msg.info.origin.position.x + (col + 0.5) * res
    y = msg.info.origin.position.y + (row + 0.5) * res
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
