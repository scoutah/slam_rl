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
  `.venv/lib/python3.12/site-packages` into `PYTHONPATH`).
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
gz server (+ GUI unless headless) → spawn robot at **(2.0, 0.0)** → 2 s robot_state_publisher
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
- `/tmp/slam_world_meta.json` – `expected_traversable_cells`, written by `generate_random_world`,
  read by `coverage_monitor.load_expected_cells()`.
- `/tmp/slam_episode_done` – written by `episode_manager` (and by `frontier_explorer` on
  reaching threshold); content is the reason (`coverage=97.3%`, `stuck (coverage=…)`,
  `timeout`). `training_runner.py` polls it, then kills the whole launch process group
  plus a list of known process names.
- Results: `results/<dd-mm-yyyy>-<mode>-<runid>.csv` (`;`-separated).

**Episode end** (`episode_manager.py`): coverage ≥ threshold, or coverage gained
< 0.5 % within 90 s (`stuck`), or 600 s wall clock. First 15 s of `/map` are ignored
(stale TRANSIENT_LOCAL map from a previous Cartographer).

**Coverage** = known-free cells (value 0–25) / expected free cells from wall-area
geometry, capped at 1.0. It does not know about unreachable rooms.

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
a keep-out square of ±0.5 m **around the origin**. Also writes the metadata file.

Coupled constants to change together: `world_size` / keep-out (utils.py), spawn
`x_pose`/`y_pose` (launch file), `SPAWN_XY` (tools/check_world_connectivity.py),
Cartographer/Nav2 map size assumptions.

## Known issues (state at start of WS 2026/27)

1. **Unpassable passages.** `tools/check_world_connectivity.py`, 300 worlds, seed 0:

   | clearance | worlds with an unreachable region | spawn inside wall/clearance |
   |---|---|---|
   | 0.22 m | 4 | 20 |
   | 0.30 m | 41 | 29 |
   | 0.40 m | 102 | 31 |

   Causes: (a) a later split wall spans its whole sub-room, so it can end *inside the
   door gap* of the parent wall and cut it into two narrow gaps; (b) the keep-out zone is
   at (0, 0) but the robot spawns at (2.0, 0.0).
2. **World size** is hard-coded (5 m) and not parameterised; spawn point is independent.
3. **Single-robot everything** (no namespaces, one Cartographer, one Nav2, one `/map`).
4. `--world PATH` reuses a saved SDF but `/tmp/slam_world_meta.json` still describes the
   last *generated* world → wrong coverage denominator.
5. `src/tb3_exploration/test/test_frontier_utils.py` is stale (imports
   `tb3_exploration.frontier_utils`, `find_nearest_frontier` – neither exist).
6. README drift: says threshold 95 % (code: 99 %), omits `--world/--randomize-worlds/--save-worlds`,
   lists `frontier_utils.py` under `tb3_exploration`, links Gazebo "Jetty" (Jazzy pairs with Harmonic).
7. Minor: launch file appends the venv path twice and hard-codes `python3.12`; unused
   `world_file` launch arg; SDF output has whitespace before `<?xml`.

## Semester goals

1. Fix the generator so every room is reachable (verify with the connectivity script;
   aim for 0 disconnected worlds at ≥ 0.30 m clearance and 0 blocked spawns).
2. Bigger worlds (parameterised size, room count scaled with area).
3. Two-robot exploration. Expect to touch: namespaced spawning + robot_state_publisher,
   per-robot Cartographer (or multi-robot SLAM) plus map merging, per-robot Nav2
   (namespaced action servers/costmaps), a frontier allocator that avoids both robots
   picking the same goal, `cmd_vel_relay`/startup spin per namespace, coverage on the
   merged map, and `training_runner`'s kill list.

## Conventions

- Python, ROS 2 `rclpy`; type hints with `X | None`; package tests via `colcon test`
  (ament flake8/pep257 run too).
- Shared logic goes in `tb3_slam`, not duplicated in the explorer packages.
- Keep `frontier_utils.py` free of ROS message imports beyond `OccupancyGrid` (viz lives
  in `frontier_viz.py`).
