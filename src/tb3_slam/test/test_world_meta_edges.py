"""Edge-case tests for world metadata loaders and the generator failure contract."""
import json

from tb3_slam import utils
from tb3_slam.utils import generate_random_world, load_spawns, load_world_size


# ── generate_random_world failure contract ──────────────────────────

def test_generate_world_returns_false_on_unwritable_path(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, 'WORLD_META_FILE', tmp_path / 'shared_meta.json')
    # A directory that does not exist and cannot be created as a *file* path.
    bad = tmp_path / 'no_such_dir' / 'sub' / 'w.sdf'
    # Make the parent un-creatable by putting a file where a dir must go.
    (tmp_path / 'no_such_dir').write_text('i am a file, not a dir')
    ok, cells = generate_random_world(str(bad), 8.0)
    assert ok is False
    assert cells == 0


# ── load_world_size ────────────────────────────────────────────────

def test_load_world_size_reads_value(tmp_path):
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps({'world_size': 12.0}))
    assert load_world_size(meta) == 12.0


def test_load_world_size_missing_file_default(tmp_path):
    assert load_world_size(tmp_path / 'missing.json') == utils.DEFAULT_WORLD_SIZE


def test_load_world_size_missing_key_default(tmp_path):
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps({'expected_traversable_cells': 5}))
    assert load_world_size(meta) == utils.DEFAULT_WORLD_SIZE


def test_load_world_size_malformed_json_default(tmp_path):
    meta = tmp_path / 'm.json'
    meta.write_text('{broken')
    assert load_world_size(meta) == utils.DEFAULT_WORLD_SIZE


# ── load_spawns wrong-length branch ─────────────────────────────────

def test_load_spawns_wrong_length_falls_back_to_world_size(tmp_path):
    # A sidecar with a single spawn (len != NUM_ROBOTS) must fall through to the
    # recorded world_size geometry, not return the malformed list.
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps({'spawns': [[0.0, -5.0]], 'world_size': 12.0}))
    assert load_spawns(meta) == utils.spawn_poses(12.0)


def test_load_spawns_wrong_length_no_size_uses_default(tmp_path):
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps({'spawns': [[0.0, -5.0], [1.0, -5.0], [2.0, -5.0]]}))
    assert load_spawns(meta) == utils.spawn_poses(utils.DEFAULT_WORLD_SIZE)


def test_load_spawns_malformed_spawns_entry(tmp_path):
    # spawns present but not coordinate pairs -> fall back on world_size.
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps({'spawns': 'nonsense', 'world_size': 8.0}))
    assert load_spawns(meta) == utils.spawn_poses(8.0)
