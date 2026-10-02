"""Tests for per-world metadata sidecars (coverage denominator reuse)."""
import json
from pathlib import Path

from tb3_slam import utils
from tb3_slam.utils import world_meta_path, write_world_meta


def test_world_meta_path_is_sidecar():
    assert world_meta_path('/tmp/foo/bar.sdf') == Path('/tmp/foo/bar.meta.json')


def test_write_world_meta_roundtrip(tmp_path):
    meta = tmp_path / 'w.meta.json'
    write_world_meta(meta, expected_traversable_cells=1234, spawn_x=0.0, spawn_y=-1.5)
    data = json.loads(meta.read_text())
    assert data['expected_traversable_cells'] == 1234
    assert data['spawn_x'] == 0.0
    assert data['spawn_y'] == -1.5


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
