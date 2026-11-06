"""The vectorised/flat-index frontier code must match the original pure-Python code exactly.

The ``reference_*`` functions are the pre-optimisation implementations, kept as an oracle.
"""
from collections import deque
import heapq
import math

import numpy as np
import pytest
from tb3_slam import frontier_utils as fu


def reference_find_frontiers(
    grid: np.ndarray,
    free_seeds: set[tuple[int, int]] | None = None,
    robot_grid_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _seeds: set[tuple[int, int]] = free_seeds if free_seeds is not None else set()

    def _is_free(r: int, c: int) -> bool:
        if (r, c) in _seeds:
            return True
        v = grid[r, c]
        return 0 <= v <= fu.FREE_MAX

    def _is_frontier(r: int, c: int) -> bool:
        has_uncertain = False
        for dr, dc in cardinal:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            v = grid[nr, nc]
            if fu.FRONTIER_MIN <= v <= fu.FRONTIER_MAX:
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
                if grid[nr, nc] >= fu.WALL_MIN:
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


def reference_free_space_parents(
    grid: np.ndarray,
    robot_grid_pos: tuple[int, int] | None,
    include_start: bool = False,
    inflation_cells: int = round(fu.PATH_INFLATION_M / 0.05),
) -> dict[tuple[int, int], tuple[int, int] | None]:
    rows, cols = grid.shape
    parents: dict[tuple[int, int], tuple[int, int] | None] = {}
    if robot_grid_pos is None:
        return parents
    sr, sc = robot_grid_pos
    if not (0 <= sr < rows and 0 <= sc < cols):
        return parents
    free = (grid >= 0) & (grid <= fu.FREE_MAX)
    inflation = max(1, inflation_cells)
    clearance = fu._wall_clearance(free, inflation)
    # Per-cell multiplier: 1 away from walls, up to 1 + fu.PATH_WALL_PENALTY at them.
    factor = 1.0 + fu.PATH_WALL_PENALTY * ((inflation - clearance) / inflation) ** 2

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
        for dr, dc in fu._STEPS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue
            nd = d + (fu._SQRT2 if dr and dc else 1.0) * factor[nr, nc]
            if nd < dist.get((nr, nc), math.inf):
                dist[nr, nc] = nd
                parents[nr, nc] = (r, c)
                heapq.heappush(heap, (nd, (nr, nc)))
    return parents


def _random_grid(rng):
    h, w = rng.integers(8, 40, 2)
    return rng.choice(
        [-1, 0, 0, 0, 0, 30, 50, 80, 100], size=(h, w),
        p=[.1, .35, .2, .1, .05, .07, .05, .04, .04]).astype(np.int8)


@pytest.mark.parametrize('seed', range(40))
def test_find_frontiers_matches_reference(seed):
    rng = np.random.default_rng(seed)
    grid = _random_grid(rng)
    h, w = grid.shape
    pos = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    for p, seeds in ((pos, {pos}), (pos, None), (None, None), ((h + 3, 2), None)):
        assert fu.find_frontiers(grid, seeds, p) == reference_find_frontiers(grid, seeds, p)


@pytest.mark.parametrize('seed', range(40))
def test_free_space_parents_matches_reference(seed):
    rng = np.random.default_rng(1000 + seed)
    grid = _random_grid(rng)
    h, w = grid.shape
    pos = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    for include_start in (True, False):
        assert (fu.free_space_parents(grid, pos, include_start=include_start)
                == reference_free_space_parents(grid, pos, include_start=include_start))
    assert fu.free_space_parents(grid, None) == {}
