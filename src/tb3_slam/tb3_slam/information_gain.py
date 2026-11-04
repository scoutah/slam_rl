"""Occlusion-aware information-gain estimate for frontier goals (shared by both explorers)."""
import math

import numpy as np
from tb3_slam.frontier_utils import FRONTIER_MAX, FRONTIER_MIN, UNKNOWN, WALL_MIN

SENSOR_RADIUS_M = 3.5   # lidar range assumed when estimating what a goal reveals


def visible_information(
    grid: np.ndarray, cell: tuple[int, int], resolution: float, radius: float,
) -> frozenset[tuple[int, int]]:
    """Estimate unseen cells within a sensor disc, stopping rays at known walls.

    Unknown and uncertain cells carry information; known-free cells do not.
    Half-cell ray steps and one ray per perimeter cell approximate visibility.
    Unobserved obstacles cannot occlude rays: this is an optimistic area proxy.
    """
    if resolution <= 0 or radius <= 0:
        raise ValueError('resolution and sensor radius must be positive')
    row, col = cell
    height, width = grid.shape
    if not (0 <= row < height and 0 <= col < width):
        return frozenset()
    extent = radius / resolution
    angles = np.linspace(0, 2 * math.pi, max(8, math.ceil(2 * math.pi * extent)),
                         endpoint=False)
    dr, dc = np.sin(angles), np.cos(angles)
    active = np.ones(len(angles), dtype=bool)
    visible = np.zeros(grid.shape, dtype=bool)
    for distance in np.arange(0, extent + 0.25, 0.5):
        rr = np.rint(row + dr * distance).astype(int)
        cc = np.rint(col + dc * distance).astype(int)
        active &= (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        values = grid[rr[indices], cc[indices]]
        active[indices[values >= WALL_MIN]] = False
        unseen = (values == UNKNOWN) | ((values >= FRONTIER_MIN) & (values <= FRONTIER_MAX))
        hits = indices[unseen]
        visible[rr[hits], cc[hits]] = True
    return frozenset(map(tuple, np.argwhere(visible).tolist()))


class InformationField:
    """Per-snapshot cache of :func:`visible_information` for one map.

    Build one per decision cycle; the map must not change while it is in use.
    """

    def __init__(self, grid: np.ndarray, resolution: float,
                 radius: float = SENSOR_RADIUS_M) -> None:
        self._grid = grid
        self._resolution = resolution
        self._radius = radius
        self._cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

    def visible(self, cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
        """Return the (cached) unseen cells a goal at ``cell`` would reveal."""
        if cell not in self._cache:
            self._cache[cell] = visible_information(
                self._grid, cell, self._resolution, self._radius)
        return self._cache[cell]

    def fraction(self, cell: tuple[int, int],
                 covered: frozenset[tuple[int, int]] = frozenset()) -> float:
        """Return the revealed area as a fraction of the sensor disc, in [0, 1].

        ``covered`` cells (e.g. what other robots' goals already reveal) are
        subtracted, so overlapping targets are not counted twice. The disc-area
        normalisation is metric, hence independent of the world size.
        """
        gain = len(self.visible(cell) - covered) * self._resolution ** 2
        return min(gain / (math.pi * self._radius ** 2), 1.0)
