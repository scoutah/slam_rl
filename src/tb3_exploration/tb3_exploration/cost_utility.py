"""Greedy cost-utility assignment over information bids (gain: tb3_slam.information_gain)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Bid:
    """One confirmed robot–frontier pair in a single map snapshot."""

    robot: str
    cell: tuple[int, int]
    path_length: float
    information: frozenset[tuple[int, int]]
    frontier: frozenset[tuple[int, int]]
    # Metres by which the teammate is closer to this target than the bidder
    # (straight line, 0 if the bidder is closer). Stops robots from crossing the
    # map into the teammate's area.
    rival_gap: float = 0.0


def greedy_assignment(
    bids: list[Bid], resolution: float, radius: float, *,
    alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.3,
    reserved_information: frozenset = frozenset(),
    reserved_frontiers: frozenset = frozenset(),
) -> list[Bid]:
    """Choose the best pair, subtract its information, then repeat for the team.

    Score = alpha * marginal area / sensor-disc area - beta * path / radius
    - gamma * rival_gap / radius. The last term makes a frontier the teammate
    is closer to less attractive, so a robot that has finished its own area
    prefers leftovers near itself over a trip across the map.
    Select useful targets even at negative scores: otherwise the last distant
    frontier could be abandoned. Zero marginal information is never assigned.
    Greedy permits recomputing coupled gains after each choice; it is not a
    globally optimal Hungarian assignment over fixed, independent utilities.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (resolution, radius, alpha, beta)):
        raise ValueError('resolution, radius and utility weights must be finite and positive')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('rival weight gamma must be finite and non-negative')
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
            score = (alpha * gain - beta * bid.path_length / radius
                     - gamma * bid.rival_gap / radius)
            return -score, bid.path_length, bid.robot, bid.cell

        chosen = min(eligible, key=rank)
        selected.append(chosen)
        robots.add(chosen.robot)
        targets.add(chosen.cell)
        claimed.update(chosen.frontier)
        covered.update(chosen.information)
