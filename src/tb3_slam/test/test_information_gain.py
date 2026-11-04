"""Check the shared information-gain estimate on hand-built grids."""
import math

import numpy as np
import pytest
from tb3_slam.information_gain import InformationField, visible_information


def test_known_wall_occludes_information_and_known_free_has_no_gain():
    grid = np.zeros((17, 17), dtype=np.int8)
    grid[8, 9] = 50
    grid[:, 11] = 95
    grid[:, 12:] = -1
    visible = visible_information(grid, (8, 8), 0.5, 3.5)
    assert (8, 9) in visible
    assert all(col < 11 for _, col in visible)
    assert (8, 8) not in visible
    assert (8, 11) not in visible


def test_information_stays_within_sensor_range_and_grid():
    grid = np.full((20, 20), -1, dtype=np.int8)
    visible = visible_information(grid, (0, 0), 0.5, 2.0)
    assert (0, 3) in visible
    assert (0, 8) not in visible
    assert all(0 <= r < 20 and 0 <= c < 20 for r, c in visible)


def test_invalid_parameters_and_outside_cell():
    grid = np.full((5, 5), -1, dtype=np.int8)
    with pytest.raises(ValueError):
        visible_information(grid, (2, 2), 0.0, 1.0)
    assert visible_information(grid, (9, 9), 0.5, 1.0) == frozenset()


def test_fraction_is_one_in_open_unknown_and_zero_when_known():
    unknown = np.full((81, 81), -1, dtype=np.int8)
    field = InformationField(unknown, 0.05, 1.0)
    assert 0.9 <= field.fraction((40, 40)) <= 1.0
    known = np.zeros((81, 81), dtype=np.int8)
    assert InformationField(known, 0.05, 1.0).fraction((40, 40)) == 0.0


def test_fraction_subtracts_covered_cells():
    grid = np.full((81, 81), -1, dtype=np.int8)
    field = InformationField(grid, 0.05, 1.0)
    full = field.fraction((40, 40))
    assert field.fraction((40, 40), field.visible((40, 40))) == 0.0
    half = field.fraction((40, 40), frozenset(
        c for c in field.visible((40, 40)) if c[1] < 40))
    assert 0.0 < half < full


def test_fraction_is_independent_of_world_size():
    # Same sensor geometry in a bigger world -> same fraction (metric normalisation).
    small = InformationField(np.full((61, 61), -1, dtype=np.int8), 0.05, 1.0)
    large = InformationField(np.full((241, 241), -1, dtype=np.int8), 0.05, 1.0)
    assert math.isclose(small.fraction((30, 30)), large.fraction((120, 120)),
                        rel_tol=1e-9)
