#!/usr/bin/env python3
"""Replay a ``dump_map.py`` snapshot through the frontier pipeline and render a PNG.

Offline (numpy + pillow, no ROS), e.g.
``uv run --no-project --with numpy --with pillow python tools/analyze_map_dump.py
map_020.npz out.png``.
Colours: white free, grey uncertain, black wall, teal unknown; red frontier
cells, blue goal cells, green robots. Prints groups and goals per robot.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'tb3_slam'))

from tb3_slam import frontier_utils as fu  # noqa: E402

SCALE = 5


def main() -> None:
    """Load the snapshot, run the pipeline per robot and write the image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, help='map_NNN.npz from dump_map.py')
    parser.add_argument('output', type=Path, help='PNG to write')
    args = parser.parse_args()

    data = np.load(args.snapshot)
    grid = data['grid']
    msg = SimpleNamespace(
        data=grid.flatten().tolist(),
        info=SimpleNamespace(
            height=grid.shape[0], width=grid.shape[1], resolution=float(data['res'])),
    )
    img = np.zeros(grid.shape + (3,), np.uint8)
    img[grid == fu.UNKNOWN] = (60, 80, 80)
    img[(grid >= 0) & (grid <= fu.FREE_MAX)] = 255
    img[(grid >= fu.FRONTIER_MIN) & (grid <= fu.FRONTIER_MAX)] = 150
    img[grid >= fu.WALL_MIN] = 20
    for name, (r, c) in zip(data['names'], data['robots']):
        r, c = int(r), int(c)
        goals, stats = fu.centroids_from_map_debug(msg, robot_grid_pos=(r, c))
        print(f'{name} robot={(r, c)} groups={stats["group_sizes"]} goals={goals}')
        for fr, fc in stats['frontiers']:
            img[fr, fc] = (255, 0, 0)
        for gr, gc in goals:
            img[gr - 2:gr + 3, gc - 2:gc + 3] = (0, 0, 255)
        img[r - 2:r + 3, c - 2:c + 3] = (0, 200, 0)
    known_r, known_c = np.where(grid != fu.UNKNOWN)
    if len(known_r):
        img = img[max(known_r.min() - 5, 0):known_r.max() + 6,
                  max(known_c.min() - 5, 0):known_c.max() + 6]
    out = Image.fromarray(img[::-1])  # flip: row 0 (south) at the bottom
    out.resize((out.width * SCALE, out.height * SCALE), Image.NEAREST).save(args.output)


if __name__ == '__main__':
    main()
