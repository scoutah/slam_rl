"""Fuse the robots' occupancy grids into one global grid.

Pure numpy (no ROS) so it can be unit tested. The merged grid has a fixed extent
(the world plus a margin) in the shared ``map`` frame, so its origin never moves
while the robots' own Cartographer grids grow. Each robot's grid is placed by its
origin plus the robot's spawn pose (``<ns>/map`` starts at the spawn, yaw 0).

Cells are fused by a fixed **priority**, not a raw min/max, and the merge is
**stateful**: the result of the previous fusion is passed back in as ``base`` so a
cell that was confirmed FREE once is not lost when Cartographer later reports it as
UNKNOWN or UNCERTAIN again (which it does for cells at the edge of a scan as the
robot turns/moves). Priority, highest first:

* WALL (75-100)      – always wins (a real obstacle must still override free),
* FREE (0-25)        – *sticky*: beats UNCERTAIN and UNKNOWN, only a WALL replaces it,
* UNCERTAIN (26-74)  – beats only UNKNOWN,
* UNKNOWN (-1)       – never overrides a known value.

The 0-25 / 26-74 / 75-100 / -1 cell classes that ``frontier_utils`` relies on
therefore stay intact, and once-free cells stop flickering back to grey.
"""
import math
from typing import Iterable, NamedTuple

import numpy as np

UNKNOWN: int = -1
DEFAULT_RESOLUTION: float = 0.05
# Space kept around the world's outer walls (m) for sensor returns just outside.
MERGED_MARGIN_M: float = 1.0

# Upper bounds of the cell classes (inclusive) used by frontier_utils.
FREE_MAX: int = 25
UNCERTAIN_MAX: int = 74


def _priority(values: np.ndarray) -> np.ndarray:
    """Map cell values to a fusion priority (higher wins). See module docstring."""
    prio = np.zeros_like(values, dtype=np.int8)
    prio[values == UNKNOWN] = 0            # unknown: lowest, never overrides
    known = values != UNKNOWN
    prio[known & (values > UNCERTAIN_MAX)] = 3   # wall: highest
    prio[known & (values <= FREE_MAX)] = 2       # free: sticky
    prio[known & (values > FREE_MAX) & (values <= UNCERTAIN_MAX)] = 1  # uncertain
    return prio


class GridSpec(NamedTuple):
    """Geometry of a grid: cell counts, origin of cell (0, 0) in metres, cell size."""

    width: int
    height: int
    origin_x: float
    origin_y: float
    resolution: float


def merged_grid_spec(world_size: float,
                     resolution: float = DEFAULT_RESOLUTION,
                     margin: float = MERGED_MARGIN_M) -> GridSpec:
    """Return the fixed merged-grid geometry, centred on the world's centre."""
    half = world_size / 2.0 + margin
    cells = math.ceil(2.0 * half / resolution)
    return GridSpec(cells, cells, -half, -half, resolution)


def fuse_grids(spec: GridSpec,
               layers: Iterable[tuple[np.ndarray, float, float, float]],
               base: np.ndarray | None = None) -> np.ndarray:
    """Fuse robot grids into one ``(height, width)`` int8 array of geometry ``spec``.

    Each layer is ``(data, origin_x, origin_y, resolution)`` with ``data`` a 2D
    int8 array (row = y) and the origin of its cell (0, 0) already expressed in
    the merged frame. Grids are shifted by whole cells (nearest, error <= half a
    cell); parts outside the merged grid are clipped.

    ``base`` is the previous merged grid (same ``spec``); when given, the merge is
    stateful and keeps every cell it already knew unless a higher-priority
    observation replaces it. Pass ``None`` for a fresh (stateless) merge.
    """
    if base is None:
        merged = np.full((spec.height, spec.width), UNKNOWN, dtype=np.int8)
    else:
        if base.shape != (spec.height, spec.width):
            raise ValueError(
                f'base shape {base.shape} does not match spec {(spec.height, spec.width)}'
            )
        merged = base.astype(np.int8, copy=True)
    merged_prio = _priority(merged)

    for data, origin_x, origin_y, resolution in layers:
        if not math.isclose(resolution, spec.resolution, rel_tol=1e-3):
            raise ValueError(
                f'grid resolution {resolution} does not match merged {spec.resolution}'
            )
        rows, cols = data.shape
        col0 = round((origin_x - spec.origin_x) / spec.resolution)
        row0 = round((origin_y - spec.origin_y) / spec.resolution)
        r_lo, r_hi = max(row0, 0), min(row0 + rows, spec.height)
        c_lo, c_hi = max(col0, 0), min(col0 + cols, spec.width)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        source = data[r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0]
        src_prio = _priority(source)
        target = merged[r_lo:r_hi, c_lo:c_hi]
        tgt_prio = merged_prio[r_lo:r_hi, c_lo:c_hi]
        # The observation wins only where its priority is strictly higher, so an
        # equal class keeps the existing (older/other-robot) value and ties never
        # flip free<->uncertain back and forth.
        take = src_prio > tgt_prio
        target[take] = source[take]
        tgt_prio[take] = src_prio[take]
    return merged
