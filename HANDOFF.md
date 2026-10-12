# Handoff: map merging (#9) – verify on Linux

Branch `feature/multi-robot-stack` (PR #25, #8 + #9 + slice of #10). Written on a Mac where the
simulation cannot run, so **everything below is implemented but unverified**. Your job on the
Linux machine: build it, run the acceptance checks, fix what breaks, and report. Read
`AGENTS.md` first for the architecture.

## Goal / merge policy

Only fully functional features go to `main`. PR #25 must **not** be merged until all
acceptance checks below pass. Do not merge it yourself; report results to the user.

## State

Implemented (unit tests pass on Mac, see below):

- `sim_slam.launch.py::map_merge_stack`: `multirobot_map_merge` node (`map_merge`), started 7 s
  after launch, `known_init_poses: true`, initial poses = spawn poses (yaw 0) from
  `multi_robot.map_merge_params`, `world_frame: map`, `discovery_rate: 0.5`. Plus static TF
  `map -> r1/map`, `map -> r2/map` (the package publishes no TF).
- `episode_manager` reads the merged `/map` (the old `/r1/map` remap is gone).
- `training_runner.py` kill list: added `map_merge`, `static_transform_publisher`.
- `tools/install_map_merge.sh`: builds `multirobot_map_merge` (robo-friends/m-explore-ros2) at
  pinned commit `326cf8a0b487c34246bb8f3326afbcd69576dc60` into `~/m-explore-ws`.
  Dockerfile runs it into `/opt/m-explore-ws`; `docker/entrypoint.sh` sources that overlay.
- Docs: AGENTS.md, README step 3b, docker/README.md.

Not done / out of scope here: explorers still use per-robot maps (allocator is #11); episode
timeout for large worlds (#10); Cartographer drift between robots is not corrected.

## Unverified assumptions (check these first)

1. The package builds on ROS 2 Jazzy (README only says "Humble and newer").
2. Executable/package names `multirobot_map_merge` / `map_merge`.
3. Init-pose parameter names are `/<ns>/map_merge/init_pose_{x,y,z,yaw}` (leading slash, derived
   from the source: robot name = parent namespace of the `/<ns>/map` topic). If the log says a
   robot was skipped for missing init pose, fix the key format in `map_merge_params` /
   `map_merge_stack` (e.g. pass via a YAML params file) and update the unit test.
4. Per-cell `max` fusion keeps the 0-25 / 26-74 / 75-100 / -1 semantics (read from the source
   `grid_compositor.cpp`; confirm on a real map).
5. `/r1/map` and `/r2/map` origins really coincide with the spawn poses (otherwise the merged
   map is shifted – look for doubled walls in RViz).

## Acceptance checks (all must pass)

```bash
export TURTLEBOT3_MODEL=burger
tools/install_map_merge.sh                       # or: docker compose build
source ~/m-explore-ws/install/setup.bash         # BEFORE the project overlay
colcon build && source install/setup.bash
colcon test --packages-select tb3_slam && colcon test-result --verbose

ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic headless:=true
# in another shell:
ros2 topic info /map -v                          # one publisher (map_merge), frame_id: map
ros2 topic echo /map --once --field header
ros2 run tf2_ros tf2_echo map r2/map             # == spawn offset of r2
```

- [ ] `multirobot_map_merge` builds; `map_merge` node starts and logs both robots found.
- [ ] `/map` is published, transient-local, frame `map`, contains both robots' explored area
      (view in RViz: add a Map display on `/map`, fixed frame `map`; walls must not be doubled).
- [ ] `frontier_utils` finds frontiers on the merged grid unchanged (e.g. record `/map` with
      `ros2 bag` or dump one message and run `find_frontier_groups` on it offline).
- [ ] `/episode/coverage` rises from the merged map; `episode_manager` ends the episode
      (threshold / stuck / timeout) – one full run:
      `python3 training_runner.py --mode classic --episodes 1 --headless`
- [ ] After the episode, `pgrep -af "gz|cartographer|nav2|map_merge|static_transform"` shows
      nothing left (teardown covers both robots).
- [ ] Same in Docker: `docker compose run --rm train --episodes 1 --headless`.

## Fallback

If the package cannot be made to build or run on Jazzy within reason: replace it with a small
numpy node (`tb3_slam`, pure-Python core + thin ROS wrapper, unit-tested) that resamples both
grids by the spawn offsets and fuses with per-cell `max` (unknown -1 loses to anything). Keep
the launch wiring, static TFs and consumers identical. Document the change in AGENTS.md
(*Map merging*) and drop `tools/install_map_merge.sh` / the Dockerfile step.

## Local test status (Mac)

`pytest src/tb3_slam/test/test_multi_robot.py` passes (26). flake8 clean on the changed source
files; `training_runner.py` has pre-existing warnings (unused `shutil`, 2 long lines).

## When done

Report which checks passed/failed with log excerpts. If everything passes: push fixes to this
branch, update AGENTS.md/README if behaviour changed, delete this file, and tell the user the
PR is ready for them to merge.
