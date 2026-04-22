# Frontier & Goal Visualization in RViz

**Status:** Draft
**Date:** 2026-04-21
**Branch:** `feature/frontier-troubleshooting`

## Purpose

Make frontier detection visually debuggable in RViz. Today, the frontier pipeline (`tb3_slam/frontier_utils.py`) runs entirely in-memory — there is no way to see which cells the explorer considers frontiers or which cell is the current Nav2 goal. This makes it hard to tell whether problems ("robot doesn't move", "wrong direction") come from bad detection, bad centroid selection, Nav2 pathing, or TF drift.

Both exploration strategies (`FrontierExplorer`, `RLExplorer`) must gain this visualization so the same RViz display works regardless of `exploration_mode`.

## Scope

**In scope:**
- Publish detected frontier cells as **red** cube markers in RViz.
- Publish the current Nav2 goal cell as a **blue** cube marker in RViz.
- Works for both `FrontierExplorer` and `RLExplorer`.
- Markers clear themselves when there are no frontiers / no active goal.

**Out of scope:**
- Visualizing centroids that are not the current goal (only the one active target is highlighted blue).
- Changing the frontier detection algorithm itself.
- Committing an RViz config — the RViz instance is started externally by Cartographer/turtlebot3 and its config is not managed in this repo. The user adds the marker display manually once.
- Showing dropped / non-navigable centroids in a third color (can be added later if needed).

## Architecture

### Data flow

```
/map (OccupancyGrid)
  → centroids_from_map_debug(msg, robot_grid_pos)
      → stats['frontiers']  (raw (row, col) cells)   ← NEW
      → centroids           (existing, unchanged)
  → build_frontier_markers(stats['frontiers'], goal_cell, msg)
  → publish MarkerArray on /frontiers_viz
```

### Components

**1. `tb3_slam/frontier_utils.py` — add frontiers to stats**

`centroids_from_map_debug` already computes `frontiers = find_frontiers(...)`. Add `'frontiers': frontiers` to the returned `stats` dict so callers can reuse the result without re-running the BFS.

No change to the return signature of `centroids_from_map` (the non-debug entrypoint). `RLExplorer` currently uses `centroids_from_map`; it will switch to `centroids_from_map_debug` so it can get the frontier list. The centroid output of both functions is identical.

**2. `tb3_slam/frontier_viz.py` — new module**

Single public function:

```python
def build_frontier_markers(
    frontiers: list[tuple[int, int]],
    goal: tuple[int, int] | None,
    map_msg: OccupancyGrid,
    frame_id: str = 'map',
    stamp: builtin_interfaces.msg.Time | None = None,
) -> MarkerArray
```

Behaviour:
- Always returns a `MarkerArray` with exactly two markers (ids 0 and 1), so RViz keeps a stable display.
- **Marker id=0 (frontiers):** `type=CUBE_LIST`, `action=ADD`, color red (r=1.0, g=0.0, b=0.0, a=0.7), `scale.x = scale.y = map_msg.info.resolution`, `scale.z = 0.02`. One `Point` per frontier cell at its world-space center (reuse `grid_to_world`). If `frontiers` is empty, the marker still uses `action=ADD` with an empty `points` list — RViz renders nothing, which is what we want (no delete/re-add flicker).
- **Marker id=1 (goal):** `type=CUBE`, color blue (r=0.0, g=0.0, b=1.0, a=0.9), `scale.x = scale.y = 1.5 * resolution`, `scale.z = 0.05` (slightly taller + wider than red cells so it's visible on top of them). When `goal` is a cell, `action=ADD` and `pose.position` is set to the goal cell's world-space center. When `goal is None`, the marker uses `action=DELETE` so the previous blue cube is removed.
- Both markers set `pose.orientation.w = 1.0` (identity quaternion) to avoid RViz "uninitialized quaternion" warnings.
- Header: `frame_id` (default `'map'`). `stamp` defaults to `None`, in which case the header stamp is left at zero (RViz uses the latest transform).
- Each marker sets `lifetime = Duration()` (0 = persist until overwritten) and `ns = 'frontier_viz'`.

This module imports `visualization_msgs.msg` and `geometry_msgs.msg` — kept out of `frontier_utils.py` to preserve its pure-computation role.

**3. `FrontierExplorer` changes (`src/tb3_exploration/tb3_exploration/frontier_explorer.py`)**

- Add publisher in `__init__`:
  ```python
  self._viz_pub = self.create_publisher(MarkerArray, '/frontiers_viz', 10)
  ```
- In `_map_callback`, after `centroids_from_map_debug(...)` returns, build and publish markers:
  ```python
  markers = build_frontier_markers(
      stats['frontiers'],
      self._current_target,
      msg,
      stamp=self.get_clock().now().to_msg(),
  )
  self._viz_pub.publish(markers)
  ```
- `self._current_target` is already maintained (set in `_try_next_goal`, cleared on success / abort / timeout). No state changes needed.

**4. `RLExplorer` changes (`src/tb3_reinforced_exploration/tb3_reinforced_exploration/rl_agent.py`)**

- Add publisher in `__init__` (same as above).
- Switch `centroids_from_map(...)` → `centroids_from_map_debug(...)` to get `stats['frontiers']`.
- Add new field `self._current_target_cell: tuple[int, int] | None = None`, set in `_map_callback` when dispatching the Nav2 goal (assign `target_cell`), cleared in `_on_goal_result`.
- Publish marker array after detection in `_map_callback`, analogous to FrontierExplorer.

**5. Package dependencies**

- `tb3_slam/package.xml`: add `<depend>visualization_msgs</depend>` if not already declared (the new `frontier_viz.py` lives in this package).
- `tb3_exploration/package.xml` and `tb3_reinforced_exploration/package.xml`: add `<depend>visualization_msgs</depend>` so ROS imports resolve at runtime.

## QoS & topic choice

- Topic: `/frontiers_viz`.
- Message type: `visualization_msgs/MarkerArray`.
- QoS: default (`depth=10`, reliable, volatile). Markers are cheap and re-published every map callback (~1 Hz), so `transient_local` is not needed.

## Error handling

- `stats['frontiers']` being empty → marker id=0 carries an empty `points` list → RViz renders nothing. No error.
- `_current_target is None` → marker id=1 uses `action=DELETE`. No error.
- Publisher creation failure would raise at construction time — handled like any other ROS publisher (node fails to start, already-existing behaviour).
- `build_frontier_markers` does not swallow exceptions; a malformed `map_msg` would propagate. This matches the rest of `frontier_utils.py`, which assumes a well-formed `OccupancyGrid`.

## Testing

- **Unit tests** (in `src/tb3_exploration/test/` or a new `src/tb3_slam/test/`):
  - `test_build_frontier_markers_empty` — empty frontiers + `goal=None` returns a `MarkerArray` with two markers, red one has no points, blue one has `action=DELETE`.
  - `test_build_frontier_markers_with_cells` — red marker has one point per frontier cell, at expected world coordinates (reuse `grid_to_world` reference values).
  - `test_build_frontier_markers_with_goal` — blue marker has `action=ADD` and correct world position.
- **Manual verification**:
  1. Launch `ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic`.
  2. In RViz: Add → `By topic` → `/frontiers_viz` → `MarkerArray`.
  3. Confirm red cells appear at the frontier band and a blue cube highlights the current Nav2 goal. Goal cube moves when a new goal is dispatched and disappears on completion.
  4. Repeat with `exploration_mode:=rl`.

## Open items

None. Design complete.
