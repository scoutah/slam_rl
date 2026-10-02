# Studienbericht 2 – Multi-robot exploration: classical vs RL

Read this first in a new session. `CLAUDE.md` (repo root) describes the existing
single-robot code in detail; this file describes **what we are building next and why**.

## Context

- University project (TurtleBot3 Burger, Gazebo, Cartographer, Nav2). Studienbericht 1 covered
  single-robot exploration with a classical frontier explorer and a pointwise DQN.
- Problem found: the 5 × 5 m worlds are too small to tell whether RL beats classical, because
  with one robot and a tiny map nearest-frontier is already near-optimal.
- **Graded deliverable: a paper of ~8000 words.** Scope must stay small enough to be written up
  in that budget. Anything that does not serve the research question is out of scope.
- The new project is **standalone**: no single-robot comparison/baseline.

## Research question

> In two-robot exploration of large, room-like worlds, does an RL-based frontier allocator
> outperform a good classical allocator?

Exactly **two methods**, **two robots**, **one or two map sizes** (map size is the only
independent variable besides method), one metric set.

## Target system

- **Simulation:** Gazebo, 2 TurtleBot3 Burgers, random room-like worlds, larger than 5 × 5 m
  (parameterised size, room count scaled with area).
- **Per-robot stack:** namespaced spawn + robot_state_publisher, own Cartographer, own Nav2,
  own `cmd_vel_relay`/startup spin.
- **Map merging:** local maps merged into one global map. Frontier detection, coverage and
  episode end run on the **merged** map.
- **Allocator (the only thing that differs between the two methods):** one node sees the merged
  map and both robots' poses, picks a frontier goal for each robot, sends it to that robot's Nav2.
  Both methods implement the same interface: `(map, poses) -> (goal_robot1, goal_robot2)`.
  - **Classical:** cost-utility allocation (Burgard et al. style). Cost = Nav2 path length per
    robot-frontier pair; utility reduced for frontiers near another robot's goal; solved
    greedily or with the Hungarian method. Must be a genuinely good baseline, not naive
    independent nearest-frontier.
  - **RL:** **centralised** pointwise DQN assigner (one agent, one replay buffer, one team reward
    = Δcoverage/Δt). Existing features `[group_size, path_len, coverage]` extended with
    team-aware features (e.g. other robot's distance to the frontier / proximity to its goal).
    Chosen over independent per-robot agents because it fits the existing code, gives a clean
    comparison (same interface), and avoids multi-agent non-stationarity.
- **Fairness:** identical worlds (fixed seeds/saved SDFs), frontier detection, Nav2 config,
  episode-end rules and metrics for both methods.

## Evaluation

- Many episodes per method on the same set of worlds, at 1–2 map sizes.
- Metrics: time to coverage threshold, stuck/timeout rate, total distance travelled
  (per robot and team), optionally robot overlap/idle time.
- Coverage denominator must be correct for the world used (see prerequisites).

## Paper outline (≈ 8000 words)

1. Introduction / motivation (why bigger maps, why multi-robot) 
2. Related work (frontier exploration, multi-robot task allocation, RL exploration)
3. System (sim, world generator, per-robot stack, map merging)
4. Allocators (classical cost-utility; centralised DQN, features, reward)
5. Experimental setup and metrics
6. Results
7. Discussion and limitations (incl. why centralised RL, not MARL)
8. Conclusion

## Prerequisites (known bugs from `CLAUDE.md`, to be fixed before evaluation)

Hard-coded world size, stale `/tmp/slam_world_meta.json` when using `--world`, README drift,
stale `tb3_exploration` tests. Tracked as GitHub issues.

## Work packages

All tracked as GitHub issues in `scoutah/slam_rl` (label `studienbericht-2`). Rough order:

1. Parameterised/larger worlds (+ spawn points for 2 robots)
2. Multi-robot simulation stack (namespaces, SLAM, Nav2, relay)
3. Map merging + coverage/episode manager on the merged map
4. Allocator interface, classical cost-utility allocator
5. RL allocator (features, reward, training)
6. Training/evaluation runner for two robots, metrics, plots
7. Experiments, paper

## Open decisions

- Map merging approach (merge Cartographer maps vs a multi-robot SLAM package).
- Concrete map sizes and episode timeout for large worlds.
- Greedy vs Hungarian assignment in the classical method.
- Exact RL team-aware features.
