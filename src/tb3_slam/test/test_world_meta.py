"""Tests for per-world metadata sidecars (coverage denominator reuse)."""
import json
import math
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
    spawns = [(-0.5, -1.5), (0.5, -1.5)]
    write_world_meta(meta, expected_traversable_cells=1234, spawns=spawns, world_size=8.0)
    data = json.loads(meta.read_text())
    assert data['expected_traversable_cells'] == 1234
    assert data['world_size'] == 8.0
    assert utils.load_spawns(meta) == spawns


def test_load_spawns_fallbacks(tmp_path):
    # Missing file -> default world geometry.
    assert utils.load_spawns(tmp_path / 'missing.json') == utils.spawn_poses(
        utils.DEFAULT_WORLD_SIZE)
    # Legacy single-robot sidecar -> geometry of its recorded world size.
    legacy = tmp_path / 'legacy.meta.json'
    legacy.write_text(json.dumps({'expected_traversable_cells': 1, 'spawn_x': 0.0,
                                  'spawn_y': -5.0, 'world_size': 12.0}))
    assert utils.load_spawns(legacy) == utils.spawn_poses(12.0)


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
    # Spawns stay just inside the south wall of the larger world.
    spawns = utils.load_spawns(world_meta_path(str(large)))
    assert spawns == utils.spawn_poses(12.0)
    assert all(y < -5.0 for _, y in spawns)


@pytest.mark.parametrize('world_size', utils.WORLD_SIZES)
def test_spawn_poses_two_robots_apart(world_size):
    spawns = utils.spawn_poses(world_size)
    assert len(spawns) == utils.NUM_ROBOTS == 2
    (x1, y1), (x2, y2) = spawns
    # Side by side at the same entry, mirrored around x = 0.
    assert y1 == y2 and x1 == -x2
    assert math.dist(spawns[0], spawns[1]) == pytest.approx(utils.SPAWN_SEPARATION)
    # Clearance discs of 0.4 m (largest checked robot clearance) do not overlap.
    assert utils.SPAWN_SEPARATION >= 2 * 0.4
    # Inside the outer walls with the wall margin on every side.
    inner = world_size / 2.0 - utils.WALL_THICKNESS / 2.0 - utils.SPAWN_WALL_MARGIN
    assert all(abs(x) <= inner and abs(y) <= inner + 1e-9 for x, y in spawns)


def _wall_boxes(sdf: Path) -> list[tuple[str, float, float, float, float]]:
    """(name, x, y, size_x, size_y) of every wall model in a generated SDF."""
    pattern = r'<model name="([^"]+)">.*?<pose>([^<]+)</pose>.*?<size>([^<]+)</size>'
    boxes = []
    for name, pose, size in re.findall(pattern, sdf.read_text(), re.S):
        x, y = map(float, pose.split()[:2])
        sx, sy = map(float, size.split()[:2])
        boxes.append((name, x, y, sx, sy))
    return boxes


@pytest.mark.parametrize('world_size', utils.WORLD_SIZES)
def test_split_walls_keep_out_of_both_spawns(tmp_path, monkeypatch, world_size):
    # No interior wall enters the keep-out box spanning both spawns, so both
    # robots are collision-free and no wall separates them.
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')
    spawns = utils.spawn_poses(world_size)
    ko = utils.SPAWN_KEEP_OUT
    ko_xmin = min(x for x, _ in spawns) - ko
    ko_xmax = max(x for x, _ in spawns) + ko
    ko_ymin = min(y for _, y in spawns) - ko
    ko_ymax = max(y for _, y in spawns) + ko
    random.seed(2)
    for i in range(30):
        world = tmp_path / f'w{i}.sdf'
        assert utils.generate_random_world(str(world), world_size)[0]
        assert utils.load_spawns(world_meta_path(str(world))) == spawns
        for name, x, y, sx, sy in _wall_boxes(world):
            if not name.startswith('split_'):
                continue
            assert (x + sx / 2 <= ko_xmin or x - sx / 2 >= ko_xmax
                    or y + sy / 2 <= ko_ymin or y - sy / 2 >= ko_ymax), name


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
