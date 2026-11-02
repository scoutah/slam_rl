#!/usr/bin/env python3
"""Offline sanity check for the random world generator (no ROS, numpy only).

Generates (or loads) worlds, rasterises and inflates the walls by a clearance
radius, and reports whether all free space is one connected region and whether
each robot's spawn is blocked/stranded or the two clearance discs overlap.
Run with ``--help`` for options; e.g. ``uv run --no-project --with numpy python ...``.
"""
import argparse
import math
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


def load_spawns(sdf_path: Path) -> list[tuple[float, float]]:
    """Robot spawns [(x, y), ...] from the world's metadata sidecar.

    Read via the same helper as the launch file, so the check tests exactly
    the poses the robots are spawned at.
    """
    return utils.load_spawns(utils.world_meta_path(str(sdf_path)))


def analyze(walls, clearance: float,
            spawns: list[tuple[float, float]]) -> tuple[list[str], int]:
    """Return (status per spawn, number_of_free_regions) for one world.

    A spawn's status is 'ok', 'blocked' (inside inflated wall) or 'stranded'
    (free, but not in the largest free region, so it cannot reach the rest of
    the world).
    """
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

    # Label 4-connected free regions (0 = not free).
    labels = np.zeros((n, n), dtype=np.int32)
    sizes = [0]
    min_cells = MIN_REGION_AREA / RASTER_RES ** 2
    for r, c in zip(*np.nonzero(free)):
        if labels[r, c]:
            continue
        label = len(sizes)
        labels[r, c] = label
        queue, size = deque([(r, c)]), 0
        while queue:
            a, b = queue.popleft()
            size += 1
            for u, v in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= u < n and 0 <= v < n and free[u, v] and not labels[u, v]:
                    labels[u, v] = label
                    queue.append((u, v))
        sizes.append(size)
    regions = sum(size >= min_cells for size in sizes[1:])
    main = int(np.argmax(sizes))

    status = []
    for x, y in spawns:
        label = labels[int((y + extent) / RASTER_RES), int((x + extent) / RASTER_RES)]
        status.append('blocked' if not label else 'ok' if label == main else 'stranded')
    return status, regions


def spawns_overlap(spawns: list[tuple[float, float]], clearance: float) -> bool:
    """True if any two robots' clearance discs overlap at their spawns."""
    return any(math.dist(a, b) < 2 * clearance
               for i, a in enumerate(spawns) for b in spawns[i + 1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-n', '--num-worlds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clearance', type=float, nargs='+', default=[0.22, 0.30, 0.40],
                        help='Robot clearance radii in metres (default: 0.22 0.30 0.40)')
    parser.add_argument('--world-size', type=float, nargs='+',
                        choices=utils.WORLD_SIZES, default=list(utils.WORLD_SIZES),
                        help='World edge lengths in metres to generate '
                             '(default: all valid sizes)')
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
    """Print disconnected / per-robot spawn failure counts per clearance radius."""
    worlds = [(p, load_walls(p), load_spawns(p)) for p in paths]
    total = len(worlds)
    for clearance in clearances:
        failures: dict[str, list[str]] = {'disconnected': [], 'spawns overlap': []}
        for i in range(utils.NUM_ROBOTS):
            failures[f'r{i + 1} blocked'] = []
            failures[f'r{i + 1} stranded'] = []
        for path, walls, spawns in worlds:
            status, regions = analyze(walls, clearance, spawns)
            if regions > 1:
                failures['disconnected'].append(path.name)
            if spawns_overlap(spawns, clearance):
                failures['spawns overlap'].append(path.name)
            for i, st in enumerate(status):
                if st != 'ok':
                    failures[f'r{i + 1} {st}'].append(path.name)
        print(f'  clearance {clearance:.2f} m: '
              + ', '.join(f'{kind} {len(names)}/{total}' for kind, names in failures.items()))
        if list_failures:
            for kind, names in failures.items():
                for name in names:
                    print(f'    {kind}: {name}')


if __name__ == '__main__':
    main()
