#!/usr/bin/env python3
"""Time the frontier pipeline on synthetic merged maps rasterised from saved worlds.

Needs the ROS Python (``/usr/bin/python3``, sourced ROS) because ``frontier_utils``
imports ``OccupancyGrid``. Example: ``python3 tools/profile_frontiers.py --size 12``.
The maps are an approximation: walls come from ``worlds/*.sdf``, "explored" is a
geodesic flood from the spawn with a 2-cell uncertain band at its front.
"""
import argparse
from collections import deque
import cProfile
from pathlib import Path
import pstats
import re
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'tb3_slam'))

from tb3_slam import frontier_utils as fu  # noqa: E402
from tb3_slam.utils import generate_random_world, spawn_poses  # noqa: E402

RES = 0.05
_BOX = re.compile(
    r'<model name="[^"]+">\s*<static>1</static>\s*<pose>([^<]+)</pose>.*?<size>([^<]+)</size>',
    re.S)


def rasterise(world: Path, size: float) -> tuple[np.ndarray, float]:
    """Return a boolean wall mask (row=y, col=x) of the world and its origin."""
    extent = size + 1.0
    n = int(round(extent / RES))
    origin = -extent / 2
    walls = np.zeros((n, n), bool)
    for pose, box in _BOX.findall(world.read_text()):
        x, y = (float(v) for v in pose.split()[:2])
        sx, sy = (float(v) for v in box.split()[:2])
        c0, c1 = int((x - sx / 2 - origin) / RES), int((x + sx / 2 - origin) / RES) + 1
        r0, r1 = int((y - sy / 2 - origin) / RES), int((y + sy / 2 - origin) / RES) + 1
        walls[max(r0, 0):r1, max(c0, 0):c1] = True
    return walls, origin


def _north_y(world: Path) -> float | None:
    """y of the north outer wall = half the world size (None if not found)."""
    m = re.search(r'name="outer_north">\s*<static>1</static>\s*<pose>\S+ (\S+) ', world.read_text())
    return float(m.group(1)) if m else None


def explored_map(walls: np.ndarray, origin: float, spawn: tuple[float, float],
                 radius_cells: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Fake a partially explored Cartographer map: flood from the spawn up to a radius."""
    n = walls.shape[0]
    start = (int((spawn[1] - origin) / RES), int((spawn[0] - origin) / RES))
    dist = np.full(walls.shape, -1, np.int32)
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if dist[r, c] >= radius_cells:
            continue
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n and 0 <= nc < n and not walls[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    grid = np.full(walls.shape, -1, np.int8)
    known = dist >= 0
    grid[known] = 0
    grid[known & (dist > radius_cells - 3)] = 50      # uncertain band at the front
    near = np.zeros_like(known)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near |= np.roll(np.roll(known, dr, 0), dc, 1)
    grid[walls & near] = 100
    return grid, start


def timed(fn, repeat: int) -> float:
    """Median wall seconds of ``fn()`` over ``repeat`` calls."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    """Profile each pipeline stage on maps at several exploration stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=float, choices=(8.0, 12.0), default=8.0)
    parser.add_argument('--world', type=Path, default=None, help='SDF (default: freshly generated)')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--cprofile', action='store_true', help='print cProfile of the full call')
    args = parser.parse_args()

    if args.world is None:
        # Note: this also rewrites the shared /tmp/slam_world_meta.json.
        args.world = Path(tempfile.mkdtemp()) / 'profile_world.sdf'
        ok, _ = generate_random_world(str(args.world), args.size)
        if not ok:
            sys.exit('world generation failed')
    world = args.world
    if _north_y(world) != args.size / 2:
        sys.exit(f'{world} is not a {args.size:g} m world')
    walls, origin = rasterise(world, args.size)
    spawn = spawn_poses(args.size)[0][:2]
    print(f'world={world.name} size={args.size:g} grid={walls.shape} spawn={spawn}')
    print(f'{"radius":>7} {"free":>6} {"frontier":>8} {"find_fr":>8} {"parents":>8} '
          f'{"reach":>7} {"total":>7}   (ms, median of {args.repeat})')
    for radius in (40, 80, 160, 320, 640):
        grid, start = explored_map(walls, origin, spawn, radius)
        msg = SimpleNamespace(
            data=grid.flatten().tolist(),
            info=SimpleNamespace(
                height=grid.shape[0], width=grid.shape[1], resolution=RES,
                origin=SimpleNamespace(position=SimpleNamespace(x=origin, y=origin))),
        )
        seeds = {start}
        t_fr = timed(lambda: fu.find_frontiers(grid, seeds, start), args.repeat)
        t_par = timed(lambda: fu.free_space_parents(grid, start, include_start=True), args.repeat)
        t_reach = timed(lambda: fu.reachable_goal_cells(grid, start, 5), args.repeat)
        t_all = timed(lambda: fu.centroids_from_map_debug(msg, robot_grid_pos=start), args.repeat)
        n_fr = len(fu.find_frontiers(grid, seeds, start))
        n_free = int(((grid >= 0) & (grid <= fu.FREE_MAX)).sum())
        print(f'{radius:>7} {n_free:>6} {n_fr:>8} {t_fr * 1e3:>8.1f} {t_par * 1e3:>8.1f} '
              f'{t_reach * 1e3:>7.1f} {t_all * 1e3:>7.1f}')
    if args.cprofile:
        prof = cProfile.Profile()
        prof.runcall(fu.centroids_from_map_debug, msg, robot_grid_pos=start)
        pstats.Stats(prof).sort_stats('cumulative').print_stats(12)


if __name__ == '__main__':
    main()
