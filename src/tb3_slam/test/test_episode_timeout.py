"""Tests for the world-size-derived episode timeout (issue #10)."""
import pytest

from tb3_slam import utils
from tb3_slam.coverage_monitor import (
    EPISODE_TIMEOUT_BY_WORLD_SIZE,
    load_episode_timeout,
    timeout_for_world_size,
)
from tb3_slam.utils import write_world_meta


@pytest.mark.parametrize('world_size', utils.WORLD_SIZES)
def test_study_sizes_have_a_tuned_budget(world_size):
    assert world_size in EPISODE_TIMEOUT_BY_WORLD_SIZE
    assert timeout_for_world_size(world_size) == EPISODE_TIMEOUT_BY_WORLD_SIZE[world_size]


def test_larger_world_gets_a_larger_budget():
    assert timeout_for_world_size(12.0) > timeout_for_world_size(8.0)


def test_unknown_size_falls_back_to_area_proportional():
    # 16 x 16 m is 4x the area of 8 x 8 m, so 4x the 8 m budget.
    assert timeout_for_world_size(16.0) == pytest.approx(
        4.0 * EPISODE_TIMEOUT_BY_WORLD_SIZE[8.0]
    )


def test_load_episode_timeout_reads_world_size(tmp_path, monkeypatch):
    meta = tmp_path / 'w.meta.json'
    write_world_meta(meta, expected_traversable_cells=1, spawns=[(0.0, 0.0)],
                     world_size=12.0)
    monkeypatch.setattr('tb3_slam.coverage_monitor.WORLD_META_FILE', meta)
    assert load_episode_timeout() == EPISODE_TIMEOUT_BY_WORLD_SIZE[12.0]


def test_load_episode_timeout_defaults_when_metadata_missing(tmp_path, monkeypatch):
    missing = tmp_path / 'missing.json'
    monkeypatch.setattr('tb3_slam.coverage_monitor.WORLD_META_FILE', missing)
    # load_world_size falls back to DEFAULT_WORLD_SIZE for a missing file.
    assert load_episode_timeout() == timeout_for_world_size(utils.DEFAULT_WORLD_SIZE)
