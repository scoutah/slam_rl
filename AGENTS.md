# SLAM RL – agent notes

TurtleBot3 (Burger) autonomous exploration in Gazebo: random room-like worlds are
generated, Cartographer builds the map, Nav2 drives, and an explorer picks frontier
goals – classically (cost-utility) or with QMIX. Two robots, merged map. University
project inherited from a previous team; README is in German.

## Before you change anything

- **Read `README.md`'s "Code map" first** – per-file index and artefact list. **Update it in the
  same change whenever you add, remove, move or repurpose a file, node, topic or `/tmp` artefact.**
- Keep files self-documenting: short module header (one or two lines), real explanation in
  docstrings and in short comments where a decision matters. No design essays in headers or here.

## Environment

- Target: **Ubuntu 24.04 + ROS 2 Jazzy + Gazebo (gz-sim) + TurtleBot3/Nav2/Cartographer**.
  Does **not** run on macOS – develop on the Mac, run on Linux or in Docker.
  `export TURTLEBOT3_MODEL=burger`.
- RL needs PyTorch in a project-local `.venv` (launch appends its `site-packages` to `PYTHONPATH`).
- Pure-Python parts (`tb3_slam/utils.py`, `tools/check_world_connectivity.py`) run anywhere with numpy,
  e.g. `uv run --no-project --with numpy python tools/check_world_connectivity.py`.
- Gitignored runtime dirs: `worlds/`, `results/`, `rl_data/`, `build/`, `install/`, `log/`.

## Commands (repo root, never `src/`)

```bash
colcon build && source install/setup.bash          # re-source after every build
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic   # or rl / none; headless:=true rviz:=false world_size:=12
python3 training_runner.py --mode classic --episodes 20 --headless --randomize-worlds --world-size 12
python3 training_runner.py --world worlds/<id>.sdf --episodes 5
colcon test && colcon test-result --verbose         # test exits 0 even on failure
pkill -9 -f gz                                       # when Gazebo survives Ctrl+C
python3 tools/check_world_connectivity.py
```

`training_runner.py` flags: see `--help` and README. Facts that matter when changing it:
the episode timeout is **not** configurable (`coverage_monitor.timeout_for_world_size`: 600 s @8 m,
900 s @12 m) and all episode times are **sim time**; `--world-size` conflicts with `--world`;
without `--randomize-worlds` one world is reused; `--sim-speed` / `SLAM_SIM_SPEED` patches Gazebo
timing via `/tmp/slam_world_speed.sdf` (`utils.apply_sim_speed`); the CSV's `elapsed_s` is sim seconds
(from `/tmp/slam_episode_done.sim_s`), next to `world_size`, `wall_s`, `sim_speed`.

## Architecture

Three ament_python packages in `src/`; per-file index in the README code map.

**Launch** (`sim_slam.launch.py`): world generated in an `OpaqueFunction` (skipped if env
`SLAM_WORLD_FILE` is set) → gz server (+GUI unless headless) + `/clock` bridge → per robot `r1`, `r2`, in
parallel: spawn, bridge, `cmd_vel_relay`, RSP (2 s), Cartographer (5 s), startup spin (6 s, 12.5 sim-s).
**Nav2 comes up strictly one robot after the other**: r1's at 6 s, r2's when r1's `nav2_gate` exits.
Concurrent lifecycle `change_state` calls lose a service response and hang the later robot's bringup
(r2 stuck in "Configuring planner_server"); a fixed stagger was not enough under load. Nav2 uses
`nav2_slim.launch.py` (only the 5 nodes the explorer needs; `collision_monitor` is bypassed since
`cmd_vel_relay` reads `cmd_vel_smoothed`). The explorer launches when the **last** `nav2_gate` exits –
no fixed delay. `episode_manager` and `map_merger` start immediately. `exploration_mode`:
`classic` → `tb3_exploration/classic_explorer`, `rl` → `tb3_reinforced_exploration/rl_explorer`,
`none` → bare stack.

**Namespacing** (`multi_robot.py`, unit-tested): upstream burger `model.sdf`, bridge yaml, Cartographer lua
and `nav2_params.yaml` are rewritten per robot into `/tmp/slam_rl_robots_*/<ns>/`. Each robot has its own
`/<ns>/{scan,odom,map,cmd_vel,…}` topics and `navigate_to_pose` / `compute_path_to_pose` actions, its own
TF tree on **`/<ns>/tf`, `/<ns>/tf_static`** (nothing on global `/tf`; every node remaps `/tf`→`tf`), and
prefixed frames `<ns>/map → <ns>/odom → <ns>/base_footprint → …`. `<ns>/map` starts at that robot's spawn
pose. Inspect: `ros2 run tf2_ros tf2_echo r1/map r1/base_link --ros-args -r /tf:=/r1/tf -r /tf_static:=/r1/tf_static`.
The explorer base feeds its TF buffer from each `/<ns>/tf[_static]` plus global `/tf_static` (for
`map→<ns>/map`), so poses resolve headless without the viz-only `tf_relay`, and dispatches goals in the
robot's own `<ns>/map` frame. Nav2's global costmap (`static_layer`) and `bt_navigator` use the **merged
`/map`** in frame `map` (a non-rolling static layer does not transform frames; inflation relaxed to
0.30 m so the 1.1 m doors stay passable), so robots can plan into areas only the teammate mapped.
`map → <ns>/map` is therefore also published on each `/<ns>/tf_static`.

**Map merging** (`map_merging.py` + `map_merger.py`): each `/<ns>/map` is placed by its origin plus the
robot's spawn pose onto one fixed-extent grid (world +1 m, 0.05 m, frame `map`) and fused by per-cell
priority. Offsets are fixed, so inter-robot Cartographer drift is not corrected (acceptable for short
8/12 m episodes – state it in the paper). Teammates look like walls, so `clear_robot_footprints` frees each
robot's current 0.22 m disc.

**Frontier pipeline** (`frontier_utils.py`): cell classes `0–25` free / `26–74` uncertain / `75–100` wall /
`-1` unknown. A **frontier = FREE cell with an UNCERTAIN cardinal neighbour** and no wall within a 5×5
window (intentionally *not* "next to −1" – Cartographer's probability map), found by BFS; groups are
8-connected, tiny and "wall-sandwich" groups are dropped as phantoms. Goal per group: FREE cell nearest
the group mean; if not footprint-safe/reachable (`GOAL_CLEARANCE_M` 0.22 m) it is popped outward from the
frontier line; if that fails (tight space) the first reachable cell on the robot's shortest FREE-cell path
(8-connected Dijkstra with a wall-proximity penalty mimicking Nav2 inflation); no path → no goal. A goal
closer than `MIN_GOAL_GRID_DIST` (15 cells) to the robot is replaced by a viewpoint (≥15 cells away, ≤2 m
from the target, wall-free line of sight) or dropped. `grid_to_world` uses a **0.4999** offset on purpose
(Nav2 rounding) – don't "fix" to 0.5.

**Explorer base + methods** (`tb3_slam/multi_robot_explorer.py::MultiRobotExplorerNode`): owns all ROS I/O –
merged `/map`, robot poses, dispatch to `/<ns>/navigate_to_pose`, goal supervision (cancel after
`goal_timeout` 45 s or `no_progress_timeout` 8 s without `progress_step` 0.1 m), escalating `_recover`
(clear costmaps → back up 0.3 m → spin 90°, repeating until a goal succeeds; no new goal meanwhile), Nav2
path-cost cache, frontier markers on `/frontiers_viz` (only with a subscriber). Every 2 s `_decide` calls
`select_goals(merged_map, [RobotState]) -> {ns: Goal|None}` (`None` = leave the robot). Helpers:
`candidates_for(robot)` and `path_cost(robot, x, y)`. When the episode file appears the base stops
re-tasking and cancels goals.
- `ClassicExplorer` overrides `_decide` (serial Nav2 path checks per robot, greedy cost-utility
  assignment, rival discount).
- `RLExplorer` overrides `select_goals`: **QMIX, CTDE**. Idle robots decide in fixed order on a local
  observation (5 features) with one shared agent net; teammates' goals are communicated as intents and
  candidates near them are dropped, so nothing is double-booked. The global state (4 features) feeds only
  the mixer during training; reward is team coverage gain per second. ε decays per episode; the learner
  loads/saves `agent.pt`, `mixer.pt`, `joint_replay.pkl`, `qmix_meta.pkl` in `data_dir` /
  `$SLAM_RL_DATA_DIR`. It runs untrained (fresh net, near-random at high ε) and **saves to disk** – point
  `SLAM_RL_DATA_DIR` at a throwaway dir for tests.
- `frontier_explorer.py` and `rl_agent.py` are legacy single-robot nodes, not launched.

**Coverage / episode end** (`episode_manager.py`, merged `/map`): coverage = known-free cells (0–25) /
expected free cells, capped at 1.0. Ends on coverage ≥ threshold, <0.5 % gained in 90 s (`stuck`), the
size-derived timeout, or **`exhausted`**: the explorer base sees every robot without goal for
`exhausted_idle_s` (15 sim-s) after at least one dispatch, writes `exhausted (coverage=…%)` into the
episode file (episode_manager never overwrites an existing file), and `training_runner.py` copies the world
(+ `.meta.json`) into `worlds/` for replay. The first 15 s of `/map` are ignored (stale TRANSIENT_LOCAL).

## World generator (`utils.py::generate_random_world`)

Square box of edge `world_size`, 0.15 m walls, house-like recursive BSP. **Only `WORLD_SIZES` (8, 12) are
valid** (generator raises `ValueError`; launch, runner and connectivity check reject others); default 8.
Split while a room is > `MAX_ROOM_AREA` (9 m²) or more elongated than `MAX_ROOM_ASPECT` (2.5), never below
`MIN_ROOM_AREA` (5 m²) / `MIN_ROOM_SPAN` (2 m). One door per split wall (`MIN_PATH_WIDTH` 1.1 m); the room
graph is a tree. Split walls stay out of a keep-out box around both spawns and may not separate them.

**Spawns** (`spawn_poses`): derived from `world_size` – side by side, `SPAWN_SEPARATION` 1.0 m apart,
mirrored around x=0, just inside the south wall – written to the metadata and read back by launch and the
connectivity check via `load_spawns()`. Change `spawn_poses` / `SPAWN_WALL_MARGIN` / `SPAWN_SEPARATION` only.

## Studienbericht 2 – research question

> How well does an RL-based multi-robot exploration strategy generalise to unknown geometries
> compared to a classical exploration strategy?

Two methods (classical cost-utility vs QMIX), two robots, two map sizes. **The RL policy is trained only on
8 × 8 m**; both methods are evaluated on 8 × 8 m (in-distribution) and 12 × 12 m (unseen). World size is the
only independent variable besides method; robot count stays at two. The classical method is the
generalisation-invariant reference. The quantity of interest is the **RL/classic ratio per size and its
change from 8 → 12**, not absolute times. RL features must stay **size-invariant** (normalised by the map
diagonal, no absolute coordinates). Both methods share worlds, frontier detection, Nav2 config, episode-end
rules and metrics (time to coverage, stuck/timeout rate, distance, redundancy). ~8000-word paper; keep scope
tight, no single-robot baseline.

## Docker (headless)

Headless only (no Gazebo client, no RViz), for the macOS co-developer and AWS training. Files: `Dockerfile`,
`compose.yaml`, `.dockerignore`, `docker/{entrypoint.sh,train.sh,README.md}`. Base `ros:jazzy-ros-base`; the
sim stack is installed via **apt in the Dockerfile, not `package.xml`**; CPU-only torch in a venv at
`<workspace>/.venv`; workspace `/root/projects/slam_rl`. Services: `dev` (interactive, mounts `src/`,
`training_runner.py`, `tools/`) and `train` (runs `training_runner.py` via `docker/train.sh`, code baked in
for reproducibility); both `init: true`. `results/`, `worlds/`, `rl_data/` are bind-mounted. Usage:
`docker/README.md`.

## Conventions

- Python, ROS 2 `rclpy`; type hints `X | None`; tests via `colcon test` (ament flake8/pep257 run too).
- Shared logic goes in `tb3_slam`, not duplicated in the explorer packages.
- Keep `frontier_utils.py` free of ROS message imports beyond `OccupancyGrid` (viz lives in `frontier_viz.py`).
- New apt/pip deps or a changed Python version → `Dockerfile`; new launch args / data paths →
  `docker/entrypoint.sh` / `train.sh` / `compose.yaml`; changed usage → `docker/README.md`.
