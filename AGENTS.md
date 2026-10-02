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
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic   # or :=rl, headless:=true
python3 training_runner.py --mode classic --episodes 20 --headless --randomize-worlds
python3 training_runner.py --world worlds/<id>.sdf --episodes 5     # reuse a saved world
colcon test --packages-select tb3_slam && colcon test-result --verbose
pkill -9 -f gz                                        # when Gazebo survives Ctrl+C
python3 tools/check_world_connectivity.py            # offline world-generator check (no ROS)
```

`training_runner.py` options: `--mode classic|rl`, `--episodes`, `--timeout` (600 s),
`--threshold` (0.99), `--headless`, `--world PATH`, `--randomize-worlds`
(otherwise ONE world is reused for all episodes), `--save-worlds` (persist to `worlds/`).

## Architecture

Three ament_python packages in `src/`:

| Package | Contents |
|---|---|
| `tb3_slam` | launch file, world generator (`utils.py`), `episode_manager`, `cmd_vel_relay`, **shared** `frontier_utils.py`, `frontier_viz.py`, `coverage_monitor.py` |
| `tb3_exploration` | `frontier_explorer` node (classic) |
| `tb3_reinforced_exploration` | `reinforced_exploration` node (`rl_agent.py`, DQN) |

**Launch sequence** (`src/tb3_slam/launch/sim_slam.launch.py`, timer-based):
world generated at launch-description time (unless env `SLAM_WORLD_FILE` is set) →
gz server (+ GUI unless headless) → spawn robot near the south wall (edge entry,
scales with world size; written to world metadata by the generator) → 2 s robot_state_publisher
→ 5 s Cartographer (`turtlebot3_cartographer`) → 6 s startup spin (4 s of TwistStamped on
`/cmd_vel`) → 10 s Nav2 (`nav2_bringup/navigation_launch.py`, default params) →
20 s exploration node. `episode_manager` and `cmd_vel_relay` start immediately.

**Data flow**
- `/map` (OccupancyGrid from Cartographer) → explorers + episode_manager.
- Nav2 actions used: `navigate_to_pose`, `compute_path_to_pose` (for ranking candidates).
- `cmd_vel_relay`: Nav2 `/cmd_vel_smoothed` (Twist) → `/cmd_vel` (TwistStamped, Jazzy gz bridge).
- `/frontiers_viz` (MarkerArray) for RViz; `/episode/coverage` (Float32).
- All topics/frames are **global, single-robot** (`/map`, `/cmd_vel`, `map`→`base_link`).

**Inter-process files in `/tmp`**
- `/tmp/slam_world_meta.json` – `expected_traversable_cells` (+ spawn), written by
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
- Groups: 8-connected clusters. Goal per group: nearest valid FREE cell to the centroid,
  ≥ 15 cells (0.75 m) from the robot (`MIN_GOAL_GRID_DIST`) to avoid instant "success".
- `grid_to_world` uses a **0.4999** offset on purpose (Nav2 rounding) – don't "fix" to 0.5.

### Classic explorer (`frontier_explorer.py`)

Ranks all fresh centroids by Nav2 plan length (Euclidean fallback if the planner
isn't up or ranking takes > 8 s), goes to the nearest. `_tried` set blacklists failed goals
until a goal succeeds with real motion (≥ 0.2 m). Goals cancelled after 45 s.

### RL explorer (`rl_agent.py`)

Pointwise DQN: each reachable centroid → features `[group_size/100, path_len/10, coverage]`
→ MLP (3→64→32→1) → Q; ε-greedy (1.0 → 0.05, ×0.98 per gradient step). Reward =
Δcoverage / Δt × 100, computed at the next decision. Transitions are committed to the
replay buffer only when the episode succeeds (stuck/timeout episodes are discarded).
Model/replay/meta persisted to **hard-coded** `~/projects/slam_rl/rl_data/`. No blacklist of
failed goals; spins via `/cmd_vel` when idle 15 s.

## World generator (`tb3_slam/utils.py::generate_random_world`)

5 × 5 m box (`world_size`), 0.15 m walls, recursive BSP with 3–5 splits; each split wall
gets one door of `MIN_PATH_WIDTH = 1.1 m`; min room span 1.5 m. Split walls may not enter
a keep-out square of ±0.5 m **around the spawn point** (near the south wall). Also writes
the metadata file (traversable cells + spawn x/y).

The spawn is derived from `world_size` by `spawn_pose()` in utils.py (centred in x, just
inside the south wall) and written to the world metadata. The launch file and
`check_world_connectivity.py` both read it from there, so it scales with the world and
cannot drift. To move the spawn, change `spawn_pose` / `SPAWN_WALL_MARGIN` only.

## Known issues (state at start of WS 2026/27)

1. **World size** is hard-coded (5 m) and not parameterised; spawn point is independent.
2. **Single-robot everything** (no namespaces, one Cartographer, one Nav2, one `/map`).

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

Two methods (classical vs RL), two robots, one or two map sizes (map size is the only
independent variable besides method), one metric set.

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

## Conventions

- Python, ROS 2 `rclpy`; type hints with `X | None`; package tests via `colcon test`
  (ament flake8/pep257 run too).
- Shared logic goes in `tb3_slam`, not duplicated in the explorer packages.
- Keep `frontier_utils.py` free of ROS message imports beyond `OccupancyGrid` (viz lives
  in `frontier_viz.py`).
