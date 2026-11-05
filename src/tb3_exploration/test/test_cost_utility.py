"""Check coordinated cost-utility assignment on hand-built snapshots."""
import math

import numpy as np
import pytest

from tb3_exploration.cost_utility import Bid, greedy_assignment
from tb3_slam.information_gain import visible_information


def _bid(robot, cell, cost, information, frontier=None):
    return Bid(robot, cell, cost, frozenset(information),
               frozenset(frontier if frontier is not None else [cell]))


def test_more_information_can_beat_the_nearest_frontier():
    near = _bid('r1', (1, 1), 1.0, range(2))
    far = _bid('r1', (2, 2), 2.0, range(200))
    assert greedy_assignment([near, far], 0.1, 1.0) == [far]


def test_best_pair_is_chosen_over_robot_iteration_order():
    first = _bid('r1', (1, 1), 2.0, range(50))
    second = _bid('r2', (1, 1), 1.0, range(50))
    other = _bid('r1', (8, 8), 3.0, range(50, 100))
    assert greedy_assignment([first, second, other], 0.1, 1.0) == [second, other]


def test_marginal_gain_spreads_robots_beyond_distinct_goal_points():
    left = _bid('r1', (1, 1), 1.0, range(100))
    nearby = _bid('r2', (1, 2), 1.1, range(10, 110))
    right = _bid('r2', (9, 9), 2.0, range(200, 280))
    assert greedy_assignment([left, nearby, right], 0.1, 1.0) == [left, right]


def test_one_frontier_with_different_approaches_is_not_double_assigned():
    first = _bid('r1', (1, 1), 1.0, range(100), frontier=[(5, 5), (5, 6)])
    second = _bid('r2', (1, 2), 2.0, range(100, 200), frontier=[(5, 6), (5, 7)])
    assert greedy_assignment([first, second], 0.1, 1.0) == [first]


def test_active_robot_reserves_its_frontier_and_visible_information():
    same = _bid('r2', (1, 1), 1.0, range(100), frontier=[(5, 5)])
    redundant = _bid('r2', (1, 2), 1.0, range(100))
    useful = _bid('r2', (9, 9), 3.0, range(200, 300))
    assert greedy_assignment(
        [same, redundant, useful], 0.1, 1.0,
        reserved_information=frozenset(range(100)),
        reserved_frontiers=frozenset([(5, 5)]),
    ) == [useful]


@pytest.mark.parametrize('cost', [None, math.inf, math.nan, -1.0, 0.0])
def test_unconfirmed_or_invalid_paths_are_excluded(cost):
    bad = _bid('r1', (1, 1), cost, range(100))
    assert greedy_assignment([bad], 0.1, 1.0) == []


def test_empty_and_fully_redundant_frontiers_allow_idle_robots():
    assert greedy_assignment([], 0.1, 1.0) == []
    assert greedy_assignment([_bid('r1', (1, 1), 1.0, [])], 0.1, 1.0) == []


def test_distant_but_useful_frontier_is_not_abandoned_for_a_negative_score():
    far = _bid('r1', (9, 9), 20.0, [1])
    assert greedy_assignment([far], 0.1, 1.0) == [far]


def test_ties_are_deterministic_and_grid_extent_does_not_enter_the_score():
    first = _bid('r1', (1, 1), 1.0, range(100))
    second = _bid('r2', (9, 9), 1.0, range(100, 200))
    assert greedy_assignment([second, first], 0.1, 1.0) == [first, second]
    grid = np.zeros((80, 80), dtype=np.int8)
    grid[35:45, 45:55] = -1
    larger = np.pad(grid, ((0, 40), (0, 40)), constant_values=-1)
    assert (visible_information(grid, (40, 40), 0.1, 1.0)
            == visible_information(larger, (40, 40), 0.1, 1.0))


def test_rival_gap_prefers_target_in_own_area_over_teammates_area():
    # Equal information and path; the second target is closer to the teammate.
    own = _bid('r1', (1, 1), 4.0, range(50))
    rival_area = Bid('r1', (2, 2), 4.0, frozenset(range(50, 100)), frozenset([(2, 2)]),
                     rival_gap=5.0)
    assert greedy_assignment([rival_area, own], 0.1, 3.5) == [own]
    assert greedy_assignment([rival_area, own], 0.1, 3.5, gamma=0.0)[0] in (own, rival_area)


def test_rival_gap_still_assigns_when_nothing_else_is_left():
    only = Bid('r1', (2, 2), 4.0, frozenset(range(50)), frozenset([(2, 2)]), rival_gap=9.0)
    assert greedy_assignment([only], 0.1, 3.5) == [only]
