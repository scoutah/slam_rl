# SLAM RL – notes for Claude

TurtleBot3 (Burger) autonomous exploration in Gazebo: random room-like worlds are
generated, Cartographer builds the map, Nav2 drives, and an exploration node picks
frontier goals – either classically (nearest by path length) or with a DQN.
University project, inherited from a previous team; README is in German.

## Environment

- Target: **Ubuntu 24.04 + ROS 2 Jazzy + Gazebo (gz-sim) + TurtleBot3/Nav2/Cartographer**.
  The simulation does **not** run on macOS – develop on the Mac, run on Linux.
- `export TURTLEBOT3_MODEL=burger` is required.
- RL agent needs PyTorch in a project-local `.venv` (the launch file injects
  `.venv/lib/python<major>.<minor>/site-packages` into `PYTHONPATH`, matching the
  running interpreter).
- Pure-Python parts (`tb3_slam/utils.py`, `tools/`) run anywhere with numpy, e.g.
  `uv run --no-project --with numpy python tools/check_world_connectivity.py`.
- Gitignored runtime dirs: `worlds/`, `results/`, `rl_data/`, `build/`, `install/`, `log/`.

## Commands (run from repo root, never from `src/`)

```bash
colcon build && source install/setup.bash          # re-source after every build
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic   # or :=rl, headless:=true, world_size:=12
python3 training_runner.py --mode classic --episodes 20 --headless --randomize-worlds --world-size 12
python3 training_runner.py --world worlds/<id>.sdf --episodes 5     # reuse a saved world
colcon test --packages-select tb3_slam && colcon test-result --verbose
pkill -9 -f gz                                        # when Gazebo survives Ctrl+C
python3 tools/check_world_connectivity.py           # offline generator check, both sizes (no ROS)
```

`training_runner.py` options: `--mode classic|rl`, `--episodes`, `--timeout` (600 s),
`--threshold` (0.99), `--headless`, `--world PATH`, `--world-size 8|12` (edge length of
generated worlds, default 8; not allowed with `--world` alone), `--randomize-worlds` (otherwise ONE world
is reused for all episodes), `--save-worlds` (persist to `worlds/`). The results CSV has a
`world_size` column (read from the world's metadata sidecar).

## Architecture

Three ament_python packages in `src/`:

| Package | Contents |
|---|---|
| `tb3_slam` | launch file, world generator (`utils.py`), `episode_manager`, `cmd_vel_relay`, **shared** `frontier_utils.py`, `frontier_viz.py`, `coverage_monitor.py` |
| `tb3_exploration` | `frontier_explorer` node (classic) |
| `tb3_reinforced_exploration` | `reinforced_exploration` node (`rl_agent.py`, DQN) |

**Launch sequence** (`src/tb3_slam/launch/sim_slam.launch.py`, timer-based):
world generated at launch time in an `OpaqueFunction` (so the `world_size` launch arg is
resolved; skipped if env `SLAM_WORLD_FILE` is set) →
gz server (+ GUI unless headless) + one global `/clock` bridge → **per robot `r1`, `r2`**
(namespaces `ROBOT_NAMESPACES`, spawns from the world metadata, both stacks in parallel):
spawn `burger_<ns>` + gz bridge + `cmd_vel_relay` → 2 s robot_state_publisher → 5 s
Cartographer (`cartographer_node` + `cartographer_occupancy_grid_node`, launched directly, no
RViz) → 6 s startup spin (4 s of TwistStamped on `/<ns>/cmd_vel`) → 10 s Nav2
(`nav2_bringup/navigation_launch.py` under `PushRosNamespace`, default params rewritten per
robot) → 20 s exploration. `episode_manager` starts immediately.
`exploration_mode:=classic` runs one `frontier_explorer` per robot on its own map, `:=rl` runs
the DQN for `r1` only (shared `rl_data/`), `:=none` brings up the bare stack (goals from
outside). All interim until the allocator (#11) works on the merged map (#9).

**Namespacing** (`tb3_slam/multi_robot.py`, pure Python, unit-tested): at launch the upstream
burger `model.sdf`, bridge yaml, `turtlebot3_lds_2d.lua` and `nav2_params.yaml` are rewritten
per robot into a `/tmp/slam_rl_robots_*/<ns>/` dir. Every robot has
- its own topics `/<ns>/{scan,odom,imu,joint_states,cmd_vel,map,…}` and actions
  `/<ns>/navigate_to_pose`, `/<ns>/compute_path_to_pose`;
- its own TF tree on **`/<ns>/tf`, `/<ns>/tf_static`** (Nav2 multi-robot convention; every
  robot node remaps `/tf` → `tf`), nothing on the global `/tf`;
- prefixed frames `<ns>/map → <ns>/odom → <ns>/base_footprint → <ns>/base_link …` (gz
  DiffDrive frame ids, RSP `frame_prefix`, Cartographer lua, Nav2 params), so the trees can be
  joined later without collisions. `<ns>/map` starts at that robot's spawn pose.
To inspect a tree: `ros2 run tf2_ros tf2_echo r1/map r1/base_link --ros-args -r /tf:=/r1/tf -r /tf_static:=/r1/tf_static`.

**Data flow** (per robot unless noted)
- `/<ns>/map` (OccupancyGrid from Cartographer) → that robot's explorer; `episode_manager`
  watches `/r1/map` until the merged map exists (#10).
- Nav2 actions used: `navigate_to_pose`, `compute_path_to_pose` (for ranking candidates).
- `cmd_vel_relay`: Nav2 `/<ns>/cmd_vel_smoothed` (Twist) → `/<ns>/cmd_vel` (TwistStamped, Jazzy gz bridge).
- `/<ns>/frontiers_viz` (MarkerArray) for RViz; `/episode/coverage` (Float32, global).
- Explorers take `map_frame` / `base_frame` parameters (defaults `map` / `base_link`).

**Inter-process files in `/tmp`**
- `/tmp/slam_world_meta.json` – `expected_traversable_cells` (+ `spawns`, `world_size`), written by
  `generate_random_world`, read by `coverage_monitor.load_expected_cells()`. The generator
  also writes a sidecar `<world>.meta.json` next to each SDF; the launch file copies that
  sidecar into this file when a saved world is reused via `SLAM_WORLD_FILE`/`--world`.
- `/tmp/slam_episode_done` – written by `episode_manager` (and by `frontier_explorer` on
  reaching threshold); content is the reason (`coverage=97.3%`, `stuck (coverage=…)`,
  `timeout`). `training_runner.py` polls it, then kills the whole launch process group
  plus a list of known process names.
- Results: `results/<dd-mm-yyyy>-<mode>-<runid>.csv` (`;`-separated).

**Episode end** (`episode_manager.py`): coverage ≥ threshold, or coverage gained
< 0.5 % within 90 s (`stuck`), or 600 s wall clock. First 15 s of `/map` are ignored
(stale TRANSIENT_LOCAL map from a previous Cartographer).

**Coverage** = known-free cells (value 0–25) / expected free cells from wall-area
geometry, capped at 1.0.

### Frontier pipeline (`tb3_slam/frontier_utils.py`, used by both explorers)

- Cell classes: `0–25` free, `26–74` uncertain, `75–100` wall, `-1` unknown.
- **Frontier = FREE cell with an UNCERTAIN (26–74) cardinal neighbour** and no wall within
  a 5×5 window; found by BFS over free cells from the robot. (Not "next to -1" – this is
  intentional for Cartographer's probability map; some docstrings still say "unknown".)
- Groups: 8-connected clusters. Goal per group: nearest cell to the centroid in the
  robot's footprint-safe, cardinal-connected FREE region, ≥ 15 cells (0.75 m)
  from the robot (`MIN_GOAL_GRID_DIST`) to avoid instant "success". Clearance is
  `GOAL_CLEARANCE_M` (0.22 m, rounded up to cells; conservative square), matching
  Nav2's robot radius rather than treating its whole inflation layer as lethal.
  FREE cells within the robot's current footprint allow it to leave an incompletely
  mapped start; the target itself always has full clearance. Missing/non-FREE robot
  poses produce no goals. Narrow corridors produce a reachable approach target on
  the robot's side; duplicate approach targets share their frontier group sizes.
- `grid_to_world` uses a **0.4999** offset on purpose (Nav2 rounding) – don't "fix" to 0.5.

### Classic explorer (`frontier_explorer.py`)

Ranks all fresh centroids by confirmed Nav2 plan length and goes to the nearest.
If the planner isn't up, waits for the next map; after 8 s of ranking, uses only
confirmed paths (no Euclidean fallback to unverified goals).
The queued world goal is rechecked against footprint-safe reachability in the latest
map immediately before dispatch, preserving its world position across origin shifts.
`_tried` blacklists failed goals until a goal succeeds with real motion (≥ 0.2 m).
An independent timer
cancels a goal after 8 s without 0.1 m of translation; moving goals retain the 45 s
overall timeout. Generation IDs ignore delayed ranking/navigation responses after
timeouts, so an old cancellation cannot clear a new goal. See issue #21.

### RL explorer (`rl_agent.py`)

Pointwise DQN: each reachable centroid → features `[group_size/100, path_len/10, coverage]`
→ MLP (3→64→32→1) → Q; ε-greedy (1.0 → 0.05, ×0.98 per gradient step). Reward =
Δcoverage / Δt × 100, computed at the next decision. Transitions are committed to the
replay buffer only when the episode succeeds (stuck/timeout episodes are discarded).
Model/replay/meta persisted to **hard-coded** `~/projects/slam_rl/rl_data/`. No blacklist of
failed goals; spins via `/cmd_vel` when idle 15 s.

## World generator (`tb3_slam/utils.py::generate_random_world`)

Square box of edge length `world_size`, 0.15 m walls, house-like recursive BSP. Sizes are
always edge lengths: "12 m" means 12 × 12 m = 144 m². **Only the study sizes in
`WORLD_SIZES` (8, 12) are valid** – the generator raises `ValueError` otherwise, and the
launch file, `training_runner.py` and `check_world_connectivity.py` reject other values.
Default (`DEFAULT_WORLD_SIZE`) is 8.

- **Stop criterion, not a split count:** a room is split while it is > `MAX_ROOM_AREA`
  (9 m²) or more elongated than `MAX_ROOM_ASPECT` (2.5); no cut leaves a room
  < `MIN_ROOM_AREA` (5 m²) or narrower than `MIN_ROOM_SPAN` (2 m). The room count therefore
  follows the area (≈ 8 rooms on 8 × 8 m, ≈ 18 on 12 × 12 m, ~8 m² each).
- Cuts run across the room's long side at 35–65 % of it (`SPLIT_FRACTION`) – no slivers/strips.
  Only if no valid cut exists there (e.g. the middle of the square 8 × 8 m world is taken by
  the spawn keep-out) is the other axis tried; otherwise the room stays unsplit.
- Each split wall gets one door of `MIN_PATH_WIDTH = 1.1 m`, either flush with a corner or
  with ≥ `MIN_DOOR_SIDE_WALL` (0.8 m) wall on both sides; later walls keep that distance
  from existing doors (no wall stubs, door never narrowed). The room graph is a tree
  (no loops, no open-plan openings – deliberately out of scope).
- Split walls may not enter a keep-out box of ±`SPAWN_KEEP_OUT` (0.5 m) **around both
  spawn points** (one box spanning both), and no cut may pass between the spawns
  (`separates_spawns`), so both robots always start in the same room.
- Writes the metadata (traversable cells, `spawns: [[x, y], [x, y]]`, `world_size`).

Nothing in Cartographer (`turtlebot3_lds_2d.lua`, unbounded submaps) or Nav2 default params
(global costmap sized from `/map`, rolling 3 × 3 m local costmap) depends on world size.

**Study sizes (Studienbericht 2): 8 × 8 m (64 m²) and 12 × 12 m (144 m²).** Verified fully
connected with `check_world_connectivity.py --world-size 8 12` (clearances 0.22–0.40 m).
The episode timeout for these sizes is decided in #10 / #15 (600 s was tuned for 5 × 5 m).

**Spawns (two robots, #6):** derived deterministically from `world_size` by `spawn_poses()`
in utils.py – side by side at the same entry, `SPAWN_SEPARATION` (1.0 m) apart, mirrored
around x = 0, just inside the south wall – and written to the world metadata. The launch file
and `check_world_connectivity.py` both read them via `load_spawns()` (falls back to the
default geometry for missing/legacy single-spawn metadata), so they scale with the world and
cannot drift. To move the spawns, change `spawn_poses` / `SPAWN_WALL_MARGIN` /
`SPAWN_SEPARATION` only. The connectivity check reports per robot whether the spawn is
blocked (in inflated wall) or stranded (not in the main free region), and whether the two
robots' clearance discs overlap.

## Known issues (state at start of WS 2026/27)

1. **No merged map yet** (#9): two namespaced stacks run (#8), but frontier detection,
   coverage and episode end still see single-robot maps.

## Studienbericht 2 – multi-robot exploration

### Why

On 5 × 5 m worlds with one robot, nearest-frontier is already near-optimal, so there is no
room to tell whether RL beats a classical allocator. Larger, room-like worlds explored by
two robots create coordination choices where a smarter allocator can matter. The graded
deliverable is a paper of ~8000 words, so scope stays tight: anything that does not serve
the research question is out of scope. The project is standalone – no single-robot baseline.

### Research question

> In two-robot exploration of large, room-like worlds, does an RL-based frontier allocator
> outperform a good classical allocator?

Two methods (classical vs RL), two robots, two map sizes – 8 × 8 m and 12 × 12 m (map size
is the only independent variable besides method), one metric set.

### The system

Gazebo simulation of two namespaced TurtleBot3 Burgers in random room-like worlds larger
than 5 × 5 m. Each robot runs its own Cartographer, Nav2 and `cmd_vel_relay`; the local maps
are merged into one global map on which frontier detection, coverage and episode end run. A
single allocator node sees the merged map and both robots' poses and assigns a frontier goal
to each robot through a shared interface `(merged_map, poses) -> (goal_r1, goal_r2)`; the only
difference between the two methods is the allocator behind that interface – a cost-utility
classical allocator and a centralised pointwise-DQN allocator. Both methods run on identical
worlds, frontier detection, Nav2 config, episode-end rules and metrics; evaluation compares
time to coverage, stuck/timeout rate and distance travelled across many episodes.

## Docker (headless)

Headless-only containerisation for the macOS co-developer and AWS training (no
GUI – no Gazebo client, no RViz). Files at repo root: `Dockerfile`,
`compose.yaml`, `.dockerignore`, plus `docker/entrypoint.sh`, `docker/train.sh`,
`docker/README.md`.

- Base `ros:jazzy-ros-base` (multi-arch amd64 + arm64, Python 3.12); sim stack
  (TurtleBot3, `ros-gz`, Nav2, Cartographer, `tf-transformations`) installed via
  apt – it is **not** in the `package.xml` files. CPU-only torch in a venv at
  `<workspace>/.venv` (the launch file injects `python3.12/site-packages`).
- Workspace lives at `/root/projects/slam_rl`; RL persistence goes to
  `SLAM_RL_DATA_DIR` (set by the entrypoints to `<workspace>/rl_data`).
- Two compose services share the image: `dev` (interactive shell, mounts `src/`
  for live edits → rebuild inside) and `train` (runs `training_runner.py` via
  `docker/train.sh`, which forwards all args; `src/` **not** mounted, so runs are
  reproducible from the built image). Both use `init: true` to reap the process
  tree the runner spawns/kills each episode.
- Usage: `docker compose build`, then
  `docker compose run --rm train --episodes 1 --headless` or
  `docker compose run --rm dev`. `results/`, `worlds/`, `rl_data/` are bind-mounted.

## Conventions

- Python, ROS 2 `rclpy`; type hints with `X | None`; package tests via `colcon test`
  (ament flake8/pep257 run too).
- Shared logic goes in `tb3_slam`, not duplicated in the explorer packages.
- Keep `frontier_utils.py` free of ROS message imports beyond `OccupancyGrid` (viz lives
  in `frontier_viz.py`).
- After any change, check whether the Docker image or its commands/docs need updating:
  new apt/pip dependencies or a changed Python version → `Dockerfile`; new launch
  args, entrypoint behaviour or data paths → `docker/entrypoint.sh` /
  `docker/train.sh` / `compose.yaml`; changed usage → `docker/README.md` (and the
  Docker section above). The sim stack is apt-installed in the `Dockerfile`, not via
  `package.xml`, so new ROS/sim dependencies must be added there too.
