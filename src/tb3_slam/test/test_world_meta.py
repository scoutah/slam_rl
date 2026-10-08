"""Tests for per-world metadata sidecars (coverage denominator reuse)."""
import json
import random
import re
from pathlib import Path

import pytest

from tb3_slam import utils
from tb3_slam.utils import world_meta_path, write_world_meta


def test_world_meta_path_is_sidecar():
    assert world_meta_path('/tmp/foo/bar.sdf') == Path('/tmp/foo/bar.meta.json')


def test_write_world_meta_roundtrip(tmp_path):
    meta = tmp_path / 'w.meta.json'
    write_world_meta(meta, expected_traversable_cells=1234, spawn_x=0.0, spawn_y=-1.5,
                     world_size=8.0)
    data = json.loads(meta.read_text())
    assert data['expected_traversable_cells'] == 1234
    assert data['spawn_x'] == 0.0
    assert data['spawn_y'] == -1.5
    assert data['world_size'] == 8.0


def test_generate_world_writes_sidecar(tmp_path, monkeypatch):
    # Keep the shared /tmp meta file out of the way so a running sim is unaffected.
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')

    world = tmp_path / 'world.sdf'
    ok, expected = utils.generate_random_world(str(world))
    assert ok
    assert expected > 0

    sidecar = world_meta_path(str(world))
    assert sidecar.exists(), 'metadata sidecar must be written next to the world'

    data = json.loads(sidecar.read_text())
    assert data['expected_traversable_cells'] == expected
    # Sidecar and shared file describe the same (just generated) world.
    shared = json.loads((tmp_path / 'shared_meta.json').read_text())
    assert shared['expected_traversable_cells'] == expected


def test_generate_world_of_given_size(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')

    small, large = tmp_path / 'small.sdf', tmp_path / 'large.sdf'
    ok_small, cells_small = utils.generate_random_world(str(small), 8.0)
    ok_large, cells_large = utils.generate_random_world(str(large), 12.0)
    assert ok_small and ok_large
    assert cells_large > 2 * cells_small

    data = json.loads(world_meta_path(str(large)).read_text())
    assert data['world_size'] == 12.0
    # Spawn stays just inside the south wall of the larger world.
    assert (data['spawn_x'], data['spawn_y']) == utils.spawn_pose(12.0, utils.WALL_THICKNESS)
    assert data['spawn_y'] < -5.0


@pytest.mark.parametrize('world_size', [5.0, 10.0, 0.0])
def test_only_study_world_sizes_are_valid(tmp_path, world_size):
    with pytest.raises(ValueError):
        utils.generate_random_world(str(tmp_path / 'w.sdf'), world_size)
    assert not (tmp_path / 'w.sdf').exists()


def _split_walls(sdf: Path) -> list[tuple[float, float]]:
    """(size_x, size_y) of every interior split wall in a generated SDF."""
    text = sdf.read_text()
    models = re.findall(r'<model name="split_[^"]*">.*?<size>([^<]+)</size>', text, re.S)
    return [tuple(map(float, m.split()[:2])) for m in models]


@pytest.mark.parametrize('world_size, min_rooms, max_rooms', [(8.0, 3, 7), (12.0, 7, 15)])
def test_room_count_follows_area(tmp_path, monkeypatch, world_size, min_rooms, max_rooms):
    # Rooms are split until house-sized (MIN_ROOM_AREA..MAX_ROOM_AREA), so the
    # room count is roughly area / mean room size, not a fixed split count.
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')
    random.seed(0)
    for i in range(20):
        world = tmp_path / f'w{i}.sdf'
        assert utils.generate_random_world(str(world), world_size)[0]
        rooms = len(set(re.findall(r'name="split_(\d+)_', world.read_text()))) + 1
        assert min_rooms <= rooms <= max_rooms


def test_no_short_wall_stubs(tmp_path, monkeypatch):
    # Doors are flush with a corner or leave MIN_DOOR_SIDE_WALL of wall on
    # each side, so no interior wall piece is shorter than that.
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')
    random.seed(1)
    for i in range(20):
        world = tmp_path / f'w{i}.sdf'
        assert utils.generate_random_world(str(world), 12.0)[0]
        for sx, sy in _split_walls(world):
            assert max(sx, sy) >= utils.MIN_DOOR_SIDE_WALL
