# SLAM RL – agent notes

TurtleBot3 (Burger) autonomous exploration in Gazebo: random room-like worlds are
generated, Cartographer builds the map, Nav2 drives, and an explorer picks frontier
goals – classically (cost-utility) or with a DQN. Two robots, merged map. University
project inherited from a previous team; README is in German.

## Before you change anything

- **Read `README.md`'s "Code map" first** — it is the per-file index and the pipeline
  overview. **Update that code map in the same change whenever you add, remove, move or
  repurpose a file, node, topic or `/tmp` artefact.** The map and the code must never
  drift apart.
- Keep files self-documenting: short module header (one or two lines saying what the file
  is), real explanation in function/class docstrings and in short comments at the point a
  decision matters. No long design essays in file headers or in this file.

## Environment

- Target: **Ubuntu 24.04 + ROS 2 Jazzy + Gazebo (gz-sim) + TurtleBot3/Nav2/Cartographer**.
  Does **not** run on macOS – develop on the Mac, run on Linux. `export TURTLEBOT3_MODEL=burger`.
- RL needs PyTorch in a project-local `.venv` (launch injects its `site-packages` into
  `PYTHONPATH`, matching the running interpreter).
- Pure-Python parts (`tb3_slam/utils.py`, `tools/`) run anywhere with numpy, e.g.
  `uv run --no-project --with numpy python tools/check_world_connectivity.py`.
- Gitignored runtime dirs: `worlds/`, `results/`, `rl_data/`, `build/`, `install/`, `log/`.

## Commands (run from repo root, never from `src/`)

```bash
colcon build && source install/setup.bash          # re-source after every build
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic   # or :=rl/:=none, headless:=true, rviz:=false, world_size:=12
python3 training_runner.py --mode classic --episodes 20 --headless --randomize-worlds --world-size 12
python3 training_runner.py --world worlds/<id>.sdf --episodes 5     # reuse a saved world
colcon test --packages-select tb3_slam && colcon test-result --verbose
pkill -9 -f gz                                       # when Gazebo survives Ctrl+C
python3 tools/check_world_connectivity.py            # offline generator check, both sizes
```

`training_runner.py` (see `--help`): `--mode classic|rl`, `--episodes`, `--threshold` (0.99),
`--headless`, `--no-rviz`, `--world PATH`, `--world-size 8|12` (not allowed with `--world`),
`--randomize-worlds` (else ONE world is reused), `--save-worlds`. The episode timeout is **not**
configurable: it is derived from world size (`coverage_monitor.timeout_for_world_size`, 600 s @8 m,
900 s @12 m, #10). Results CSV has a `world_size` column.

## Architecture

Three ament_python packages in `src/`; the per-file index is the code map in `README.md`.

**Launch order** (`sim_slam.launch.py`): world generated in an `OpaqueFunction`
(so `world_size` is resolved; skipped if env `SLAM_WORLD_FILE` is set) → gz server (+GUI unless
headless) + global `/clock` bridge → **per robot `r1`, `r2`** the pre-Nav2 phase in parallel:
spawn + gz bridge + `cmd_vel_relay` → 2 s RSP → 5 s Cartographer → 6 s startup spin
(12.5 s @0.5 rad/s, ~1 turn). **Nav2 runs near-parallel per robot but with a small configure
offset**: r1's Nav2 at `NAV2_BASE_DELAY` (6 s, right after Cartographer sets up map→odom), r2's at
+`NAV2_CONFIGURE_OFFSET` (4 s → 10 s) so their lifecycle *configure* phases never issue change_state
concurrently — that overlap was the sole cause of r2's intermittent change_state stall, and it is
only the first few seconds of each bringup (the earlier fully-serial design needlessly delayed r2
~25 s). Nav2 comes up via `nav2_slim.launch.py` (not upstream `navigation_launch.py`): only the 5
nodes the explorer uses are lifecycle-managed — `controller_server`, `planner_server`,
`behavior_server`, `bt_navigator`, `velocity_smoother`; `smoother_server`/`route_server`/
`collision_monitor`/`waypoint_follower`/`docking_server` are dropped for faster activation
(`collision_monitor` is bypassed anyway — `cmd_vel_relay` reads `velocity_smoother`'s
`cmd_vel_smoothed`). One `nav2_gate` process per robot
blocks on that robot's `navigate_to_pose`; the explorer launches when the **last** gate exits
(`OnProcessExit` countdown), i.e. the instant both stacks are active — no fixed explorer delay.
`episode_manager` and `map_merger` start immediately. `exploration_mode` maps `classic` →
`tb3_exploration/classic_explorer`, `rl` → `tb3_reinforced_exploration/rl_explorer`, `none` → bare
stack. No shared allocator/coordinator node.

**Namespacing** (`multi_robot.py`, unit-tested): upstream burger `model.sdf`, bridge yaml,
`turtlebot3_lds_2d.lua` and `nav2_params.yaml` are rewritten per robot into `/tmp/slam_rl_robots_*/<ns>/`.
Each robot has its own `/<ns>/{scan,odom,map,cmd_vel,…}` topics and `navigate_to_pose` /
`compute_path_to_pose` actions, its own TF tree on **`/<ns>/tf`,`/<ns>/tf_static`** (nothing on
global `/tf`; every node remaps `/tf`→`tf`), and prefixed frames `<ns>/map → <ns>/odom →
<ns>/base_footprint → …`. `<ns>/map` starts at that robot's spawn pose.
Inspect: `ros2 run tf2_ros tf2_echo r1/map r1/base_link --ros-args -r /tf:=/r1/tf -r /tf_static:=/r1/tf_static`.
The explorer base node feeds its TF buffer from each `/<ns>/tf[_static]` directly (plus global
`/tf_static` for the `map→<ns>/map` offsets), so poses resolve headless without the viz-only
`tf_relay`, and it dispatches every goal in the robot's own `<ns>/map` frame, since each robot's
Nav2 cannot transform the merged `map` frame.

**Map merging** (`map_merging.py` numpy + `map_merger.py` node): each `/<ns>/map` is placed by its
origin plus the robot's spawn pose onto one fixed-extent grid (world +1 m, 0.05 m, frame `map`,
origin never moves) and fused by per-cell priority. Static `map → <ns>/map` transforms carry the
spawn offset. Offsets are fixed, so inter-robot Cartographer drift is not corrected (acceptable for
short 8/12 m episodes — state it in the paper). The detailed rationale (why not `multirobot_map_merge`,
why the gentle startup spin) lives in `map_merging.py` and `sim_slam.launch.py`. Teammates see each
other as walls, so `map_merger` frees each robot's current 0.22 m footprint disc
(`clear_robot_footprints`, poses from `/<ns>/tf` like the explorer — not via `tf_relay`).

**Frontier pipeline** (`frontier_utils.py`, both explorers): cell classes `0–25` free / `26–74`
uncertain / `75–100` wall / `-1` unknown. A **frontier = FREE cell with an UNCERTAIN cardinal
neighbour** and no wall within a 5×5 window (intentionally *not* "next to −1" — Cartographer's
probability map); found by BFS over free cells. Groups are 8-connected; the goal per group is the
nearest footprint-safe, cardinal-connected FREE cell to the centroid, ≥ `MIN_GOAL_GRID_DIST`
(15 cells) from the robot, with `GOAL_CLEARANCE_M` (0.22 m) clearance. `grid_to_world` uses a
**0.4999** offset on purpose (Nav2 rounding) — don't "fix" to 0.5.

**Explorer base + methods**: common ROS I/O lives in `tb3_slam/multi_robot_explorer.py::MultiRobotExplorerNode`;
a method is a subclass overriding one method, `select_goals(merged_map, [RobotState]) -> {ns: Goal|None}`
(a `Goal` of `None` = leave the robot as is). The base subscribes to the merged `/map`, looks up each
robot pose, calls `select_goals` every 2 s, dispatches to `/<ns>/navigate_to_pose`, and supervises
goals (cancel after `goal_timeout` 45 s, or `no_progress_timeout` 8 s without `progress_step` 0.1 m).
On every merged-`/map` update it also publishes the union of per-robot frontiers/centroids (plus each
active goal) as a `MarkerArray` on `/frontiers_viz` for RViz (`frontier_viz.py`), independent of goal
state and of the chosen method.
Helpers for subclasses: `candidates_for(robot)` (each `FrontierCandidate` also carries its
group's `frontier_cells`) and `path_cost(robot, x, y)`. When `episode_manager`'s episode file
appears (stuck/timeout too), the base stops re-tasking and cancels active goals. Both subclasses
(`ClassicExplorer`, `RLExplorer`) currently inherit the empty `select_goals` (baseline wiring);
the strategies are layered on later.

**Coverage / episode end** (`episode_manager.py` on the merged `/map`): coverage = known-free cells
(0–25) / expected free cells, capped at 1.0. Episode ends on coverage ≥ threshold, <0.5 % gained in
90 s (`stuck`), or the size-derived timeout. First 15 s of `/map` ignored (stale TRANSIENT_LOCAL).

## World generator (`utils.py::generate_random_world`)

Square box of edge length `world_size`, 0.15 m walls, house-like recursive BSP. **Only study sizes
`WORLD_SIZES` (8, 12) are valid** (generator raises `ValueError`; launch, `training_runner.py`,
`check_world_connectivity.py` all reject others). Default `DEFAULT_WORLD_SIZE` = 8. Room count follows
area, not a split count: split while a room is > `MAX_ROOM_AREA` (9 m²) or more elongated than
`MAX_ROOM_ASPECT` (2.5), never below `MIN_ROOM_AREA` (5 m²) / `MIN_ROOM_SPAN` (2 m). One door per
split wall (`MIN_PATH_WIDTH` 1.1 m); the room graph is a tree. Split walls stay out of a keep-out box
around both spawns and may not separate them, so both robots start in the same room.

**Spawns** (`spawn_poses`): derived deterministically from `world_size` — side by side,
`SPAWN_SEPARATION` (1.0 m) apart, mirrored around x=0, just inside the south wall — and written to the
metadata; launch and the connectivity check read them via `load_spawns()`. Change `spawn_poses` /
`SPAWN_WALL_MARGIN` / `SPAWN_SEPARATION` only. Nothing in Cartographer or Nav2 defaults depends on
world size.

## Known issues (start of WS 2026/27)

1. Explorer goal selection should run entirely on the merged `/map` (via the base node's
   `candidates_for`); confirm no path still reads a per-robot map for frontier detection (#9/#10/#11).

## Studienbericht 2 – research question

> How well does an RL-based multi-robot exploration strategy generalise to unknown geometries
> compared to a classical exploration strategy?

Two methods (classical cost-utility vs centralised pointwise-DQN), two robots, two map sizes. **The
RL allocator is trained only on 8 × 8 m**; both methods are evaluated on 8 × 8 m (in-distribution)
and 12 × 12 m (out-of-distribution, unseen). World size is the only independent variable besides
method; robot count stays at two. The classical allocator is the generalisation-invariant reference.
The quantity of interest is the **RL/classic ratio per size and its change from 8 → 12**, not absolute
times. For the RL allocator to transfer, its per-frontier features must be **size-invariant**
(normalised, not absolute coordinates). Both methods share identical worlds, frontier detection, Nav2
config, episode-end rules and metrics (time to coverage, stuck/timeout rate, distance, redundancy).
~8000-word paper; keep scope tight, no single-robot baseline.

## Docker (headless)

Headless-only (no Gazebo client, no RViz) for the macOS co-developer and AWS training. Root files:
`Dockerfile`, `compose.yaml`, `.dockerignore`, plus `docker/{entrypoint.sh,train.sh,README.md}`.
Base `ros:jazzy-ros-base` (amd64+arm64, Python 3.12); the sim stack (TurtleBot3, `ros-gz`, Nav2,
Cartographer, `tf-transformations`) is installed via **apt in the Dockerfile, not `package.xml`**.
CPU-only torch in a venv at `<workspace>/.venv`; workspace at `/root/projects/slam_rl`; RL persistence
to `SLAM_RL_DATA_DIR`. Two services: `dev` (interactive, mounts `src/`) and `train` (runs
`training_runner.py` via `docker/train.sh`, `src/` not mounted for reproducibility); both `init: true`.
Usage: `docker compose build`, then `docker compose run --rm train --episodes 1 --headless` or
`docker compose run --rm dev`. `results/`, `worlds/`, `rl_data/` are bind-mounted.

## Conventions

- Python, ROS 2 `rclpy`; type hints `X | None`; tests via `colcon test` (ament flake8/pep257 run too).
- Shared logic goes in `tb3_slam`, not duplicated in the explorer packages.
- Keep `frontier_utils.py` free of ROS message imports beyond `OccupancyGrid` (viz lives in `frontier_viz.py`).
- New apt/pip deps or a changed Python version → `Dockerfile`; new launch args / data paths →
  `docker/entrypoint.sh` / `train.sh` / `compose.yaml`; changed usage → `docker/README.md`. The sim
  stack is apt-installed in the `Dockerfile`, so new ROS/sim deps go there too (not `package.xml`).
