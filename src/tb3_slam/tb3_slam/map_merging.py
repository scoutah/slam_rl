"""Fuse the robots' occupancy grids into one global grid.

Pure numpy (no ROS) so it can be unit tested. The merged grid has a fixed extent
(the world plus a margin) in the shared ``map`` frame, so its origin never moves
while the robots' own Cartographer grids grow. Each robot's grid is placed by its
origin plus the robot's spawn pose (``<ns>/map`` starts at the spawn, yaw 0).

Cells are fused with a per-cell max: unknown (-1) loses to any known value, and
wall/uncertain beats free. The 0-25 / 26-74 / 75-100 / -1 cell classes that
``frontier_utils`` relies on therefore stay intact.
"""
import math
from typing import Iterable, NamedTuple

import numpy as np

UNKNOWN: int = -1
DEFAULT_RESOLUTION: float = 0.05
# Space kept around the world's outer walls (m) for sensor returns just outside.
MERGED_MARGIN_M: float = 1.0


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
               layers: Iterable[tuple[np.ndarray, float, float, float]]) -> np.ndarray:
    """Fuse robot grids into one ``(height, width)`` int8 array of geometry ``spec``.

    Each layer is ``(data, origin_x, origin_y, resolution)`` with ``data`` a 2D
    int8 array (row = y) and the origin of its cell (0, 0) already expressed in
    the merged frame. Grids are shifted by whole cells (nearest, error <= half a
    cell); parts outside the merged grid are clipped.
    """
    merged = np.full((spec.height, spec.width), UNKNOWN, dtype=np.int8)
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
        target = merged[r_lo:r_hi, c_lo:c_hi]
        np.maximum(target, source, out=target)
    return merged
