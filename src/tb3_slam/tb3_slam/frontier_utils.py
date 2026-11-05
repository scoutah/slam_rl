"""Shared frontier detection utilities used by all exploration strategies."""
from collections import deque
import heapq
import math

from nav_msgs.msg import OccupancyGrid
import numpy as np

# Occupancy value ranges
# NOTE: WALL_MIN lowered from 90 → 75 as an experiment. Cartographer appears to
# store some wall pixels in the 75–89 band that were previously treated as
# UNCERTAIN, which let frontier cells keep a "wall" neighbour under the
# stricter rule. Tightening the wall threshold treats those cells as walls.
FREE_MAX = 25        # cell value 0–25   → free
FRONTIER_MIN = 26    # cell value 26–74  → uncertain / frontier
FRONTIER_MAX = 74
WALL_MIN = 75        # cell value 75–100 → wall
UNKNOWN = -1


def find_frontiers(
    grid: np.ndarray,
    free_seeds: set[tuple[int, int]] | None = None,
    robot_grid_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return all frontier cells (row, col) reachable from the robot.

    A frontier is a FREE cell (0–25) that has at least one UNCERTAIN
    cardinal neighbour *and* no WALL cell within Chebyshev distance 2
    (5×5 window). Extending the wall exclusion to 2nd-degree neighbours
    gives goals a safety margin away from Nav2's inflation layer, at the
    cost of dropping a few legitimate frontiers right next to obstacles.

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
            if FRONTIER_MIN <= v <= FRONTIER_MAX:
                has_uncertain = True
        if not has_uncertain:
            return False
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr, nc] >= WALL_MIN:
                    return False
        return True

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
# Nav2's default robot_radius, not its entire (non-lethal) inflation layer.
GOAL_CLEARANCE_M = 0.22
# Nav2's costmap keeps the robot off walls; the fallback path mimics that by
# penalising cells closer than this to a non-free cell (inflation radius).
PATH_INFLATION_M = 0.55
PATH_WALL_PENALTY = 8.0
_SQRT2 = math.sqrt(2.0)
_STEPS_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def reachable_goal_cells(
    grid: np.ndarray,
    robot_grid_pos: tuple[int, int] | None,
    clearance_cells: int = 5,
) -> np.ndarray:
    """Return cells reachable over FREE space with footprint clearance.

    Erode known-free space by a conservative square before the cardinal BFS.
    This removes narrow corridors as well as unsafe goals. Unknown/uncertain
    cells and the map boundary are obstacles. A FREE robot may leave its
    currently occupied footprint over FREE cells while that neighbourhood is
    still being mapped; every returned goal lies in the eroded region.
    """
    rows, cols = grid.shape
    reachable = np.zeros(grid.shape, dtype=bool)
    if robot_grid_pos is None or rows == 0 or cols == 0:
        return reachable
    sr, sc = robot_grid_pos
    if not (0 <= sr < rows and 0 <= sc < cols):
        return reachable
    free = (grid >= 0) & (grid <= FREE_MAX)
    # Seed the BFS from the robot cell if it is FREE, else from its FREE
    # neighbours in a small window. A robot can spawn on a cell that reads as
    # wall/uncertain (e.g. the south wall bleeding into its footprint under the
    # lowered WALL_MIN); returning empty there would permanently deny it goals,
    # so fall back to the nearby free space it can actually drive onto.
    seeds: list[tuple[int, int]] = []
    if free[sr, sc]:
        seeds.append((sr, sc))
    else:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc]:
                    seeds.append((nr, nc))
    if not seeds:
        return reachable
    radius = max(0, clearance_cells)
    padded = np.pad(free, radius, constant_values=False)
    safe = free.copy()
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            safe &= padded[dr:dr + rows, dc:dc + cols]
    traversable = safe.copy()
    for seed_r, seed_c in seeds:
        r0, r1 = max(0, seed_r - radius), min(rows, seed_r + radius + 1)
        c0, c1 = max(0, seed_c - radius), min(cols, seed_c + radius + 1)
        traversable[r0:r1, c0:c1] |= free[r0:r1, c0:c1]
    queue = deque()
    for seed in seeds:
        if not reachable[seed]:
            reachable[seed] = True
            queue.append(seed)
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < rows and 0 <= nc < cols
                    and traversable[nr, nc] and not reachable[nr, nc]):
                reachable[nr, nc] = True
                queue.append((nr, nc))
    return reachable & safe


def _wall_clearance(free: np.ndarray, max_cells: int) -> np.ndarray:
    """Chebyshev distance (cells) to the nearest non-free cell, capped at ``max_cells``."""
    rows, cols = free.shape
    dist = np.full(free.shape, max_cells, dtype=np.int32)
    covered = ~free
    dist[covered] = 0
    for k in range(1, max_cells):
        padded = np.pad(covered, 1, constant_values=True)
        grown = covered.copy()
        for dr in range(3):
            for dc in range(3):
                grown |= padded[dr:dr + rows, dc:dc + cols]
        dist[grown & ~covered] = k
        covered = grown
    return dist


def free_space_parents(
    grid: np.ndarray,
    robot_grid_pos: tuple[int, int] | None,
    include_start: bool = False,
    inflation_cells: int = round(PATH_INFLATION_M / 0.05),
) -> dict[tuple[int, int], tuple[int, int] | None]:
    """Cheapest FREE-cell paths from the robot as a parent map (Dijkstra, 8-connected).

    Roots (the robot cell, or its free cardinal neighbours if the robot cell
    does not read FREE) map to ``None``. ``include_start`` treats the robot cell
    as free even if it is not, like ``find_frontiers`` does. Following parents
    from any visited cell leads back to the robot over a real FREE-cell path,
    so a cell that is missing from the result cannot be reached at all.

    Step cost grows near walls, like Nav2's inflated costmap, so the path
    keeps to the middle of rooms instead of hugging the wall; a plain BFS tree
    runs along the room edge and pulled fallback goals far from their frontier.
    Diagonal steps cost sqrt 2 and never cut corners.
    """
    rows, cols = grid.shape
    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    if robot_grid_pos is None:
        return parents
    sr, sc = robot_grid_pos
    if not (0 <= sr < rows and 0 <= sc < cols):
        return parents
    free = (grid >= 0) & (grid <= FREE_MAX)
    inflation = max(1, inflation_cells)
    clearance = _wall_clearance(free, inflation)
    # Per-cell multiplier: 1 away from walls, up to 1 + PATH_WALL_PENALTY at them.
    factor = 1.0 + PATH_WALL_PENALTY * ((inflation - clearance) / inflation) ** 2

    if include_start or free[sr, sc]:
        roots = [(sr, sc)]
    else:
        roots = [(sr + dr, sc + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                 if 0 <= sr + dr < rows and 0 <= sc + dc < cols and free[sr + dr, sc + dc]]
    dist: dict[tuple[int, int], float] = {}
    heap: list[tuple[float, tuple[int, int]]] = []
    for root in roots:
        parents[root] = None
        dist[root] = 0.0
        heap.append((0.0, root))
    heapq.heapify(heap)
    done: set[tuple[int, int]] = set()
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) in done:
            continue
        done.add((r, c))
        for dr, dc in _STEPS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue
            nd = d + (_SQRT2 if dr and dc else 1.0) * factor[nr, nc]
            if nd < dist.get((nr, nc), math.inf):
                dist[nr, nc] = nd
                parents[nr, nc] = (r, c)
                heapq.heappush(heap, (nd, (nr, nc)))
    return parents


def _pop_outward(
    target: tuple[int, int],
    group: list[tuple[int, int]],
    grid: np.ndarray,
    reachable_cells: np.ndarray,
    push_cells: int,
) -> tuple[int, int] | None:
    """Move ``target`` ``push_cells`` away from the frontier line, if that is reachable.

    The outward direction is the sum of vectors from each frontier cell's
    UNCERTAIN cardinal neighbours to that cell, i.e. the normal of the frontier
    line pointing into known free space. For a line parallel to the robot this
    steps straight off the line. Returns ``None`` when there is no direction or
    the popped cell is not footprint-safe and reachable (tight space).
    """
    rows, cols = grid.shape
    vr = vc = 0.0
    for r, c in group:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and FRONTIER_MIN <= grid[nr, nc] <= FRONTIER_MAX):
                vr -= dr
                vc -= dc
    norm = math.hypot(vr, vc)
    if norm == 0.0 or push_cells <= 0:
        return None
    pr = target[0] + round(vr / norm * push_cells)
    pc = target[1] + round(vc / norm * push_cells)
    if 0 <= pr < rows and 0 <= pc < cols and reachable_cells[pr, pc]:
        return pr, pc
    return None


def find_centroid(
    group: list[tuple[int, int]],
    grid: np.ndarray,
    robot_grid_pos: tuple[int, int] | None = None,
    *,
    reachable_cells: np.ndarray | None = None,
    parents: dict[tuple[int, int], tuple[int, int] | None] | None = None,
    push_cells: int = 5,
) -> tuple[int, int] | None:
    """Return the goal cell for a frontier group, or ``None`` if there is none.

    1. Average the group's cells and take the nearest FREE cell (the target).
    2. Pop the target outward: move it ``push_cells`` (the robot clearance)
       away from the frontier line. If that cell is footprint-safe and
       reachable (``reachable_cells``), it is the goal. For a long line next
       to the robot this lands mid-line, off the line.
    3. In a tight space the pop fails; walk the robot's FREE-cell path to the
       target and take the first cell on it, starting at the target, that is
       reachable. Nearest-by-distance would be wrong here, because it can lie
       on the far side of a wall.

    Returns ``None`` if the target cannot be reached over FREE cells at all, or
    if the goal would be closer than ``MIN_GOAL_GRID_DIST`` to the robot. The
    pipeline passes ``reachable_cells`` and ``parents`` once per map; direct
    calls compute them with five cells of clearance (0.25 m at 0.05 m).
    """
    if not group or robot_grid_pos is None:
        return None
    n = len(group)
    mean = (sum(r for r, _ in group) // n, sum(c for _, c in group) // n)
    if reachable_cells is None:
        reachable_cells = reachable_goal_cells(grid, robot_grid_pos)
    if parents is None:
        parents = free_space_parents(grid, robot_grid_pos)
    free_cells = np.argwhere((grid >= 0) & (grid <= FREE_MAX))
    if len(free_cells) == 0 or not parents:
        return None
    nearest = np.argmin(np.sum((free_cells - mean) ** 2, axis=1))
    cell: tuple[int, int] | None = (int(free_cells[nearest][0]), int(free_cells[nearest][1]))
    if cell not in parents:
        return None
    popped = _pop_outward(cell, group, grid, reachable_cells, push_cells)
    if popped is not None:
        cell = popped
    while cell is not None and not reachable_cells[cell]:
        cell = parents[cell]
    if cell is None:
        return None
    if math.dist(cell, robot_grid_pos) < MIN_GOAL_GRID_DIST:
        return None
    return cell


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
    """Return ``centroids_from_map`` results together with diagnostic stats.

    Stats keys:
      - grid_shape, robot_grid_pos, robot_cell_value
      - n_free, n_uncertain, n_wall, n_unknown
      - n_frontiers, n_groups, n_centroids, n_dropped_groups
      - group_sizes (list of ints), dropped_group_sample (up to 3 dropped groups
        with centroid coords + cell value + 3 frontier cells with cardinal
        neighbour values — for understanding why find_centroid returned None)
      - centroid_frontiers (frontier-cell sets aligned with the approach goals;
        groups sharing one approach point are combined)
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
    clearance = (
        math.ceil(GOAL_CLEARANCE_M / msg.info.resolution) if msg.info.resolution > 0 else 5
    )
    reachable = reachable_goal_cells(grid, robot_grid_pos, clearance)
    parents = free_space_parents(
        grid, robot_grid_pos, include_start=True,
        inflation_cells=(
            math.ceil(PATH_INFLATION_M / msg.info.resolution)
            if msg.info.resolution > 0 else 11),
    )

    centroids: list[tuple[int, int]] = []
    centroid_group_sizes: list[int] = []
    centroid_frontiers: list[set[tuple[int, int]]] = []
    dropped_samples: list[dict] = []
    for group in groups:
        centroid = find_centroid(
            group, grid, robot_grid_pos=robot_grid_pos, reachable_cells=reachable,
            parents=parents, push_cells=clearance,
        )
        if centroid is not None:
            if centroid in centroids:
                centroid_group_sizes[centroids.index(centroid)] += len(group)
                centroid_frontiers[centroids.index(centroid)].update(group)
            else:
                centroids.append(centroid)
                centroid_group_sizes.append(len(group))
                centroid_frontiers.append(set(group))
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
        'centroid_group_sizes': centroid_group_sizes,
        'centroid_frontiers': centroid_frontiers,
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
