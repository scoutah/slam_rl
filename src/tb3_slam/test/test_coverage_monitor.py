"""Tests for the core coverage math and metadata loading in coverage_monitor.

check_coverage drives episode-end, so its free-cell-only counting, 1.0 cap and
guard against a non-positive denominator are verified here. load_expected_cells
is exercised against a temporary metadata file by monkeypatching WORLD_META_FILE.
"""
import json

from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

import tb3_slam.coverage_monitor as cm
from tb3_slam.coverage_monitor import check_coverage


def _grid(values: list[int]) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.resolution = 0.05
    msg.info.width = len(values)
    msg.info.height = 1
    msg.data = list(np.array(values, dtype=np.int8))
    return msg


# ── check_coverage ─────────────────────────────────────────────────

def test_check_coverage_counts_only_free_cells():
    # 0..25 are free; walls (75..100) and unknown (-1) must not count.
    grid = _grid([0, 25, 26, 74, 75, 100, -1])
    coverage, known_free = check_coverage(grid, expected_cells=10)
    assert known_free == 2          # only the 0 and 25
    assert coverage == pytest.approx(0.2)


def test_check_coverage_walls_do_not_inflate():
    all_walls = _grid([100] * 20)
    coverage, known_free = check_coverage(all_walls, expected_cells=10)
    assert known_free == 0
    assert coverage == 0.0


def test_check_coverage_uncertain_not_counted():
    uncertain = _grid([26, 40, 50, 74])
    coverage, known_free = check_coverage(uncertain, expected_cells=4)
    assert known_free == 0
    assert coverage == 0.0


def test_check_coverage_capped_at_one():
    # More free cells than expected -> ratio capped at 1.0.
    grid = _grid([0] * 20)
    coverage, known_free = check_coverage(grid, expected_cells=5)
    assert known_free == 20
    assert coverage == 1.0


def test_check_coverage_boundary_values():
    # Exactly 25 counts as free, exactly 26 does not.
    assert check_coverage(_grid([25]), 1)[1] == 1
    assert check_coverage(_grid([26]), 1)[1] == 0


def test_check_coverage_zero_expected_guard():
    assert check_coverage(_grid([0, 0, 0]), expected_cells=0) == (0.0, 0)
    assert check_coverage(_grid([0, 0, 0]), expected_cells=-5) == (0.0, 0)


# ── load_expected_cells ────────────────────────────────────────────

def test_load_expected_cells_reads_value(tmp_path, monkeypatch):
    meta = tmp_path / 'slam_world_meta.json'
    meta.write_text(json.dumps({'expected_traversable_cells': 1234}))
    monkeypatch.setattr(cm, 'WORLD_META_FILE', meta)
    assert cm.load_expected_cells() == 1234


def test_load_expected_cells_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'WORLD_META_FILE', tmp_path / 'nope.json')
    assert cm.load_expected_cells() is None


def test_load_expected_cells_missing_key(tmp_path, monkeypatch):
    meta = tmp_path / 'slam_world_meta.json'
    meta.write_text(json.dumps({'world_size': 8.0}))
    monkeypatch.setattr(cm, 'WORLD_META_FILE', meta)
    assert cm.load_expected_cells() is None


def test_load_expected_cells_malformed_json(tmp_path, monkeypatch):
    meta = tmp_path / 'slam_world_meta.json'
    meta.write_text('{not valid json')
    monkeypatch.setattr(cm, 'WORLD_META_FILE', meta)
    assert cm.load_expected_cells() is None


def test_load_expected_cells_non_int_value(tmp_path, monkeypatch):
    meta = tmp_path / 'slam_world_meta.json'
    meta.write_text(json.dumps({'expected_traversable_cells': 'lots'}))
    monkeypatch.setattr(cm, 'WORLD_META_FILE', meta)
    assert cm.load_expected_cells() is None
