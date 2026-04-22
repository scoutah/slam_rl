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

    A frontier is a FREE cell (0–25) that has at least one UNCERTAIN (26–89)
    cardinal neighbour *and* no WALL (90–100) cardinal neighbour. Ruling out
    wall neighbours sacrifices a few legitimate frontiers along obstacles but
    eliminates the border cells that hug walls — those goals tend to plan
    badly and sit right on Nav2's inflation layer.

    Phantom frontiers (UNKNOWN cells behind walls reached via BFS leaking
    through UNCERTAIN wall pixels) no longer appear: BFS traverses FREE cells
    only and never passes through UNCERTAIN, so it cannot leak outside the
    walled area.
    """
    rows, cols = grid.shape
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _seeds: set[tuple[int, int]] = free_seeds if free_seeds is not None else set()

    def _is_free(r: int, c: int) -> bool:
        if (r, c) in _seeds:
            return True
        v = grid[r, c]
        return 0 <= v <= FREE_MAX

    def _is_frontier(r: int, c: int) -> bool:
        has_uncertain = False
        for dr, dc in cardinal:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            v = grid[nr, nc]
            if v >= WALL_MIN:
                return False
            if FRONTIER_MIN <= v <= FRONTIER_MAX:
                has_uncertain = True
        return has_uncertain

    # ── BFS path ────────────────────────────────────────────────────────
    if robot_grid_pos is not None:
        sr, sc = robot_grid_pos
        if 0 <= sr < rows and 0 <= sc < cols:
            visited: set[tuple[int, int]] = set()
            frontier_set: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque()

            if _is_free(sr, sc):
                queue.append((sr, sc))
                visited.add((sr, sc))
            else:
                # Robot cell isn't FREE yet — seed from free cardinal neighbours.
                for dr, dc in cardinal:
                    nr, nc = sr + dr, sc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and _is_free(nr, nc) and (nr, nc) not in visited):
                        queue.append((nr, nc))
                        visited.add((nr, nc))

            while queue:
                r, c = queue.popleft()
                if _is_frontier(r, c):
                    frontier_set.add((r, c))
                for dr, dc in cardinal:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if (nr, nc) in visited:
                        continue
                    if _is_free(nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            return list(frontier_set)

    # ── Fallback: global scan (no robot position available) ─────────────
    frontiers: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if _is_free(r, c) and _is_frontier(r, c):
                frontiers.append((r, c))
    return frontiers


def find_groups(frontiers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group frontier cells into connected clusters (8-connectivity, DFS).

    Two frontier cells belong to the same group only if they are direct
    neighbours (Chebyshev distance 1 / 3×3 window). A one-cell gap now
    splits clusters instead of merging them, so the big frontier ring along
    a wall breaks into several smaller, separately-navigable groups.
    """
    frontier_set = set(frontiers)
    visited: set[tuple[int, int]] = set()
    groups = []
    offsets = [
        (dr, dc)
        for dr in range(-1, 2)
        for dc in range(-1, 2)
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


# Minimum grid-cell distance between goal and robot. Nav2's default xy_goal_
# tolerance is ~0.25m (5 cells at 0.05m). In practice we see Nav2 declare
# "Goal succeeded" in <50 ms even at ~0.43m — likely a map/TF origin drift
# between Cartographer updates desynchronising our world frame from Nav2's.
# 15 cells (0.75m) is safely beyond any realistic drift + tolerance.
MIN_GOAL_GRID_DIST = 15


def find_centroid(
    group: list[tuple[int, int]],
    grid: np.ndarray,
    robot_grid_pos: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Return a navigable FREE cell near the geometric centroid of a frontier group.

    The frontier cells are UNKNOWN; the goal must be a FREE cell Nav2 can plan
    to. Walls in this SLAM setup can live in the UNCERTAIN range, so a naive
    cardinal-neighbour search often fails when a frontier group is separated
    from FREE space by a thin band of UNCERTAIN cells. We BFS outward from the
    frontier cells through any non-wall cell (UNKNOWN, UNCERTAIN) until we hit
    a FREE cell, picking the one closest to the group centroid.

    When ``robot_grid_pos`` is given, FREE cells within ``MIN_GOAL_GRID_DIST``
    of the robot are rejected so the commanded goal actually produces motion
    (otherwise Nav2 declares instant success inside its xy_goal_tolerance and
    the robot stays put, re-triggering the same frontier indefinitely).
    """
    if not group:
        return None

    rows, cols = grid.shape
    n = len(group)
    cr = sum(r for r, _ in group) // n
    cc = sum(c for _, c in group) // n

    MARGIN = 2
    min_dist_sq = MIN_GOAL_GRID_DIST * MIN_GOAL_GRID_DIST

    def _too_close_to_robot(r: int, c: int) -> bool:
        if robot_grid_pos is None:
            return False
        dr_ = r - robot_grid_pos[0]
        dc_ = c - robot_grid_pos[1]
        return dr_ * dr_ + dc_ * dc_ < min_dist_sq

    def _is_valid_free(r: int, c: int) -> bool:
        return (MARGIN <= r < rows - MARGIN
                and MARGIN <= c < cols - MARGIN
                and 0 <= grid[r, c] <= FREE_MAX
                and not _too_close_to_robot(r, c))

    if _is_valid_free(cr, cc):
        return (cr, cc)

    visited: set[tuple[int, int]] = set(group)
    queue: deque[tuple[int, int]] = deque(group)
    best: tuple[int, int] | None = None
    best_dist = float('inf')
    expansions = 0
    MAX_EXPANSIONS = rows * cols

    while queue and expansions < MAX_EXPANSIONS:
        r, c = queue.popleft()
        expansions += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            if _is_valid_free(nr, nc):
                dist = math.hypot(nr - cr, nc - cc)
                if dist < best_dist:
                    best_dist = dist
                    best = (nr, nc)
                continue
            v = grid[nr, nc]
            if v >= WALL_MIN:
                continue
            queue.append((nr, nc))

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
    centroids, _ = centroids_from_map_debug(msg, robot_grid_pos=robot_grid_pos)
    return centroids


def centroids_from_map_debug(
    msg: OccupancyGrid,
    robot_grid_pos: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]], dict]:
    """Same as ``centroids_from_map`` but also returns diagnostic stats.

    Stats keys:
      - grid_shape, robot_grid_pos, robot_cell_value
      - n_free, n_uncertain, n_wall, n_unknown
      - n_frontiers, n_groups, n_centroids, n_dropped_groups
      - group_sizes (list of ints), dropped_group_sample (up to 3 dropped groups
        with centroid coords + cell value + 3 frontier cells with cardinal
        neighbour values — for understanding why find_centroid returned None)
    """
    grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    free_seeds = {robot_grid_pos} if robot_grid_pos is not None else None

    rows, cols = grid.shape
    n_unknown = int(np.sum(grid == UNKNOWN))
    n_free = int(np.sum((grid >= 0) & (grid <= FREE_MAX)))
    n_uncertain = int(np.sum((grid >= FRONTIER_MIN) & (grid <= FRONTIER_MAX)))
    n_wall = int(np.sum(grid >= WALL_MIN))
    robot_cell_value: int | None = None
    if (robot_grid_pos is not None
            and 0 <= robot_grid_pos[0] < rows
            and 0 <= robot_grid_pos[1] < cols):
        robot_cell_value = int(grid[robot_grid_pos[0], robot_grid_pos[1]])

    frontiers = find_frontiers(grid, free_seeds=free_seeds, robot_grid_pos=robot_grid_pos)
    groups = find_groups(frontiers)

    centroids: list[tuple[int, int]] = []
    dropped_samples: list[dict] = []
    for group in groups:
        centroid = find_centroid(group, grid, robot_grid_pos=robot_grid_pos)
        if centroid is not None:
            centroids.append(centroid)
        elif len(dropped_samples) < 3:
            n = len(group)
            cr = sum(r for r, _ in group) // n
            cc = sum(c for _, c in group) // n
            centroid_val = (int(grid[cr, cc])
                            if 0 <= cr < rows and 0 <= cc < cols else None)
            sample_cells = []
            for fr, fc in group[:3]:
                neigh = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neigh.append(int(grid[nr, nc]))
                    else:
                        neigh.append(None)
                sample_cells.append({'cell': (fr, fc), 'neighbours': neigh})
            dropped_samples.append({
                'size': n,
                'centroid_rc': (cr, cc),
                'centroid_value': centroid_val,
                'samples': sample_cells,
            })

    stats = {
        'grid_shape': (rows, cols),
        'robot_grid_pos': robot_grid_pos,
        'robot_cell_value': robot_cell_value,
        'n_free': n_free,
        'n_uncertain': n_uncertain,
        'n_wall': n_wall,
        'n_unknown': n_unknown,
        'n_frontiers': len(frontiers),
        'n_groups': len(groups),
        'n_centroids': len(centroids),
        'n_dropped_groups': len(groups) - len(centroids),
        'group_sizes': [len(g) for g in groups],
        'dropped_group_sample': dropped_samples,
        'frontiers': frontiers,
    }
    return centroids, stats


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
