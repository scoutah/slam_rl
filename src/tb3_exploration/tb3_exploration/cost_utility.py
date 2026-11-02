"""Occlusion-aware information estimates and greedy cost-utility assignment."""
from dataclasses import dataclass
import math

import numpy as np
from tb3_slam.frontier_utils import FRONTIER_MAX, FRONTIER_MIN, UNKNOWN, WALL_MIN


@dataclass(frozen=True)
class Bid:
    """One confirmed robot–frontier pair in a single map snapshot."""

    robot: str
    cell: tuple[int, int]
    path_length: float
    information: frozenset[tuple[int, int]]
    frontier: frozenset[tuple[int, int]]


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


def greedy_assignment(
    bids: list[Bid], resolution: float, radius: float, *,
    alpha: float = 1.0, beta: float = 0.2,
    reserved_information: frozenset = frozenset(),
    reserved_frontiers: frozenset = frozenset(),
) -> list[Bid]:
    """Choose the best pair, subtract its information, then repeat for the team.

    Score = alpha * marginal area / sensor-disc area - beta * path / radius.
    Select useful targets even at negative scores: otherwise the last distant
    frontier could be abandoned. Zero marginal information is never assigned.
    Greedy permits recomputing coupled gains after each choice; it is not a
    globally optimal Hungarian assignment over fixed, independent utilities.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (resolution, radius, alpha, beta)):
        raise ValueError('resolution, radius and utility weights must be finite and positive')
    covered = set(reserved_information)
    claimed = set(reserved_frontiers)
    robots, targets = set(), set()
    selected = []
    while True:
        eligible = [
            bid for bid in bids
            if bid.robot not in robots and bid.cell not in targets
            and not (bid.frontier & claimed) and bid.information - covered
            and bid.path_length is not None
            and math.isfinite(bid.path_length) and bid.path_length > 0
        ]
        if not eligible:
            return selected

        def rank(bid):
            gain = len(bid.information - covered) * resolution ** 2 / (math.pi * radius ** 2)
            score = alpha * gain - beta * bid.path_length / radius
            return -score, bid.path_length, bid.robot, bid.cell

        chosen = min(eligible, key=rank)
        selected.append(chosen)
        robots.add(chosen.robot)
        targets.add(chosen.cell)
        claimed.update(chosen.frontier)
        covered.update(chosen.information)
