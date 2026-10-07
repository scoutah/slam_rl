#!/usr/bin/env python3
"""
Offline sanity check for the random world generator (no ROS required).

Generates N worlds with tb3_slam.utils.generate_random_world (or loads the
given .sdf files), rasterises the wall boxes, inflates them by a clearance
radius and checks:

  * connectivity – is all free space one connected region? More than one
    region means some room is unreachable for a robot of that clearance.
  * spawn        – is the robot spawn point itself inside (inflated) wall?

Usage:
    python3 tools/check_world_connectivity.py                 # 300 worlds
    python3 tools/check_world_connectivity.py -n 1000 --seed 1
    python3 tools/check_world_connectivity.py --clearance 0.22 0.3 0.4
    python3 tools/check_world_connectivity.py --world-size 8 12
    python3 tools/check_world_connectivity.py --world worlds/abc123.sdf

Only needs numpy (e.g. `uv run --no-project --with numpy python ...`).
"""
import argparse
import json
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'tb3_slam'))

from tb3_slam import utils  # noqa: E402

RASTER_RES = 0.02         # metres per raster cell
MIN_REGION_AREA = 0.05    # m²; smaller free pockets (wall corners) are ignored


def load_walls(sdf_path: Path) -> list[tuple[float, float, float, float]]:
    """Return (x, y, size_x, size_y) for every box model in the SDF."""
    # The generator emits whitespace before <?xml ...>, which ElementTree rejects.
    root = ET.fromstring(sdf_path.read_text().strip())
    walls = []
    for model in root.iter('model'):
        pose = model.find('pose')
        size = model.find('.//collision/geometry/box/size')
        if pose is None or size is None:
            continue
        x, y = map(float, pose.text.split()[:2])
        sx, sy = map(float, size.text.split()[:2])
        walls.append((x, y, sx, sy))
    return walls


def load_spawn(sdf_path: Path) -> tuple[float, float]:
    """Robot spawn (x, y) from the world's metadata sidecar.

    The generator writes the spawn next to every world, so it matches what the
    launch file uses for that world. Worlds without a sidecar fall back to the
    default world geometry.
    """
    try:
        meta = json.loads(utils.world_meta_path(str(sdf_path)).read_text())
        return float(meta['spawn_x']), float(meta['spawn_y'])
    except (OSError, ValueError, KeyError):
        return utils.spawn_pose(utils.DEFAULT_WORLD_SIZE, utils.WALL_THICKNESS)


def analyze(walls, clearance: float,
            spawn_xy: tuple[float, float]) -> tuple[bool, int]:
    """Return (spawn_blocked, number_of_free_regions) for one world."""
    extent = max(max(abs(x) + sx / 2, abs(y) + sy / 2) for x, y, sx, sy in walls) + 0.1

    n = int(2 * extent / RASTER_RES)
    coords = (np.arange(n) + 0.5) * RASTER_RES - extent
    X, Y = np.meshgrid(coords, coords)

    blocked = np.zeros((n, n), dtype=bool)
    outer = 0.0
    for x, y, sx, sy in walls:
        blocked |= (np.abs(X - x) <= sx / 2 + clearance) & (np.abs(Y - y) <= sy / 2 + clearance)
        outer = max(outer, abs(x) - sx / 2, abs(y) - sy / 2)
    free = ~blocked & (np.abs(X) < outer) & (np.abs(Y) < outer)

    sr = int((spawn_xy[1] + extent) / RASTER_RES)
    sc = int((spawn_xy[0] + extent) / RASTER_RES)
    spawn_blocked = not free[sr, sc]

    seen = np.zeros_like(free)
    min_cells = MIN_REGION_AREA / RASTER_RES ** 2
    regions = 0
    for r, c in zip(*np.nonzero(free)):
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, size = deque([(r, c)]), 0
        while queue:
            a, b = queue.popleft()
            size += 1
            for u, v in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= u < n and 0 <= v < n and free[u, v] and not seen[u, v]:
                    seen[u, v] = True
                    queue.append((u, v))
        if size >= min_cells:
            regions += 1
    return spawn_blocked, regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-n', '--num-worlds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clearance', type=float, nargs='+', default=[0.22, 0.30, 0.40],
                        help='Robot clearance radii in metres (default: 0.22 0.30 0.40)')
    parser.add_argument('--world-size', type=float, nargs='+',
                        default=[utils.DEFAULT_WORLD_SIZE],
                        help='World edge lengths in metres to generate '
                             f'(default: {utils.DEFAULT_WORLD_SIZE})')
    parser.add_argument('--world', type=Path, nargs='+',
                        help='Check existing .sdf files instead of generating worlds')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.world:
            report(args.world, args.clearance, list_failures=True)
            return
        # Keep generate_random_world from overwriting the metadata file a
        # running simulation may be reading.
        utils.WORLD_META_FILE = Path(tmp) / 'meta.json'
        for world_size in args.world_size:
            random.seed(args.seed)
            paths = []
            for i in range(args.num_worlds):
                p = Path(tmp) / f'world_{world_size:g}_{i}.sdf'
                ok, _ = utils.generate_random_world(str(p), world_size)
                if not ok:
                    print(f'generation failed for world {i}', file=sys.stderr)
                    continue
                paths.append(p)
            print(f'world size {world_size:g} x {world_size:g} m:')
            report(paths, args.clearance, list_failures=False)


def report(paths: list[Path], clearances: list[float], list_failures: bool) -> None:
    """Print disconnected / spawn-blocked counts per clearance radius."""
    worlds = [(p, load_walls(p), load_spawn(p)) for p in paths]
    total = len(worlds)
    for clearance in clearances:
        disconnected, spawn_blocked = [], []
        for path, walls, spawn_xy in worlds:
            sb, regions = analyze(walls, clearance, spawn_xy)
            if regions > 1:
                disconnected.append(path.name)
            if sb:
                spawn_blocked.append(path.name)
        print(f'  clearance {clearance:.2f} m: '
              f'disconnected {len(disconnected)}/{total}, '
              f'spawn blocked {len(spawn_blocked)}/{total}')
        if list_failures:
            for name in disconnected:
                print(f'    disconnected: {name}')
            for name in spawn_blocked:
                print(f'    spawn blocked: {name}')


if __name__ == '__main__':
    main()
