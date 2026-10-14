# Handoff: map merging (#9)

Branch `feature/multi-robot-stack` (PR #25 = #8 + #9 + slice of #10). **Do not merge to main
yet** – the user merges only fully functional features and decides when.

## State

Verified headless in Docker on an Apple-Silicon Mac (arm64, Jazzy):

- Image builds; `colcon test`: 86 tests, 2 failures that are pre-existing repo-wide lint
  (pep257 D213 / flake8 import order in files untouched by #9). New/changed files are clean.
- `map_merger` publishes `/map` (frame `map`, fixed 200x200 grid at 0.05 m for an 8 m world),
  static `map -> r1/map`, `map -> r2/map`; merged map is one consistent room, walls not doubled,
  both idle and while exploring (classic mode, 240 s).
- `frontier_utils.centroids_from_map_debug` runs unchanged on the merged grid (robot cell inside
  the grid and FREE, centroids found).
- `training_runner.py --mode classic --episodes 1 --headless`: episode ends (timeout rule),
  CSV written, no processes left running.

Design changes vs the first attempt (documented in AGENTS.md *Map merging*):

- `multirobot_map_merge` (m-explore-ros2) was tried and dropped: pixel-space transforms,
  centred output origin, ignores per-map origins -> not in the `map` frame, walls double.
  Replaced by `tb3_slam/map_merging.py` + `map_merger.py`.
- Startup spin reduced from 1.5 rad/s x 4 s to 0.5 rad/s x 12.5 s: the fast spin tilted
  Cartographer's map ~10 degrees against the world and broke the yaw-0 spawn offsets.

## Still to do / open

- [ ] Check on real Linux (or longer Docker runs) that the maps stay aligned over a full
      episode, also for 12 m worlds and several random worlds (drift is not corrected).
- [ ] Coverage was observed fluctuating in earlier (pre-fix) runs; confirm it is monotone-ish
      now over a full episode (`/episode/coverage`).
- [ ] View the merged map in RViz (headless Docker could only dump it to PNG). Without
      `--headless`, `sim_slam.launch.py` / `training_runner.py` now open RViz with
      `src/tb3_slam/rviz/sim_slam.rviz` (fixed frame `map`: merged `/map` on, `/r1/map` and
      `/r2/map` available but off, frontier markers). `rviz:=false` / `--no-rviz` disables it.
      The launch condition was verified, the RViz window itself was not (no display in Docker).
- [ ] #10: episode timeout for 8/12 m worlds is still undecided; the `coverage` CSV column
      is empty for `timeout`/`stuck` rows (pre-existing).
- [ ] #11: explorers still run per robot on their own maps; goals on the merged map need the
      static `map -> <ns>/map` transforms.

## Reproduce

```bash
docker compose build
docker compose run --rm train --mode classic --episodes 1 --headless
# or natively:  colcon build && source install/setup.bash && ros2 launch tb3_slam sim_slam.launch.py headless:=true
```

Delete this file when the PR is ready.
