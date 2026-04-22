# Frontier & Goal Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish detected frontier cells (red) and the currently targeted Nav2 goal cell (blue) as RViz markers from both `FrontierExplorer` and `RLExplorer`, so frontier detection can be visually debugged.

**Architecture:** `frontier_utils.centroids_from_map_debug` already computes raw frontier cells — expose them through the stats dict. A new shared helper `tb3_slam/frontier_viz.py` builds a two-marker `MarkerArray` (red `CUBE_LIST` for frontiers, blue `CUBE` for the goal). Each explorer owns a `/frontiers_viz` publisher and calls the helper on every `/map` callback.

**Tech Stack:** ROS 2 (rclpy), `visualization_msgs/MarkerArray`, `nav_msgs/OccupancyGrid`, `geometry_msgs/Point`, Python 3, pytest.

**Spec:** `docs/superpowers/specs/2026-04-21-frontier-viz-design.md`

**Conventions / assumptions:**
- Branch is already checked out (`feature/frontier-troubleshooting`).
- Tests run via `pytest` from the repo root: `pytest src/tb3_slam/test/...`. ROS 2 messages (`visualization_msgs`, `nav_msgs`, `geometry_msgs`) must be importable — i.e. the ROS 2 environment is sourced (`source /opt/ros/<distro>/setup.bash` and any workspace overlay).
- `tb3_slam` does not currently have a Python `test/` directory for unit tests — only lint tests. Task 2 creates `src/tb3_slam/test/test_frontier_viz.py`; pytest discovers it via `find_packages(exclude=['test'])` in `setup.py` (tests are excluded from install but still discoverable by pytest).

---

## Task 1: Expose raw frontiers in `centroids_from_map_debug` stats

**Files:**
- Modify: `src/tb3_slam/tb3_slam/frontier_utils.py:247-322` (add `'frontiers'` key to returned stats dict)
- Test: `src/tb3_slam/test/test_frontier_utils_stats.py` (new)

**Why:** Both explorers already call (or will call) `centroids_from_map_debug`. Returning the raw frontier list in `stats` avoids a second BFS pass in the visualization path.

- [ ] **Step 1: Write the failing test**

Create `src/tb3_slam/test/test_frontier_utils_stats.py`:

```python
"""Tests for centroids_from_map_debug stats dict."""
import numpy as np
from nav_msgs.msg import OccupancyGrid

from tb3_slam.frontier_utils import centroids_from_map_debug


def _make_grid(data: list[list[int]]) -> OccupancyGrid:
    arr = np.array(data, dtype=np.int8)
    msg = OccupancyGrid()
    msg.info.resolution = 0.05
    msg.info.width = arr.shape[1]
    msg.info.height = arr.shape[0]
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.data = arr.flatten().tolist()
    return msg


def test_stats_includes_raw_frontiers():
    # 3x4 grid: column 0 is UNKNOWN, rest is FREE → frontier cells are column-0
    # cells adjacent to FREE (column 1). Seed BFS from a FREE cell.
    grid = [
        [-1, 0, 0, 0],
        [-1, 0, 0, 0],
        [-1, 0, 0, 0],
    ]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(1, 2))

    assert 'frontiers' in stats, 'stats dict must include frontiers key'
    assert set(stats['frontiers']) == {(0, 0), (1, 0), (2, 0)}


def test_stats_frontiers_empty_when_no_unknown():
    grid = [[0, 0], [0, 0]]
    msg = _make_grid(grid)
    _, stats = centroids_from_map_debug(msg, robot_grid_pos=(0, 0))

    assert stats['frontiers'] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tb3_slam/test/test_frontier_utils_stats.py -v`
Expected: FAIL with `KeyError: 'frontiers'` or `AssertionError: 'frontiers' in stats`.

- [ ] **Step 3: Add `frontiers` to the stats dict**

Edit `src/tb3_slam/tb3_slam/frontier_utils.py`. Locate the stats-dict construction around line 307 and add the `frontiers` entry. The full replacement block (look for `stats = {` inside `centroids_from_map_debug`):

```python
    stats = {
        'grid_shape': (rows, cols),
        'robot_grid_pos': robot_grid_pos,
        'robot_cell_value': robot_cell_value,
        'n_free': n_free,
        'n_uncertain': n_uncertain,
        'n_wall': n_wall,
        'n_unknown': n_unknown,
        'n_frontiers': len(frontiers),
        'n_groups': len(groups),
        'n_centroids': len(centroids),
        'n_dropped_groups': len(groups) - len(centroids),
        'group_sizes': [len(g) for g in groups],
        'dropped_group_sample': dropped_samples,
        'frontiers': frontiers,
    }
    return centroids, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/tb3_slam/test/test_frontier_utils_stats.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/tb3_slam/tb3_slam/frontier_utils.py src/tb3_slam/test/test_frontier_utils_stats.py
git commit -m "feat(frontier): expose raw frontier cells in debug stats"
```

---

## Task 2: Create `frontier_viz.build_frontier_markers` helper

**Files:**
- Create: `src/tb3_slam/tb3_slam/frontier_viz.py`
- Test: `src/tb3_slam/test/test_frontier_viz.py`

- [ ] **Step 1: Write the failing test**

Create `src/tb3_slam/test/test_frontier_viz.py`:

```python
"""Tests for build_frontier_markers."""
import numpy as np
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from tb3_slam.frontier_viz import build_frontier_markers


def _make_map(resolution: float = 0.05, origin=(0.0, 0.0), size=(4, 4)) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.info.resolution = resolution
    msg.info.width = size[1]
    msg.info.height = size[0]
    msg.info.origin.position.x = origin[0]
    msg.info.origin.position.y = origin[1]
    msg.data = (np.zeros(size[0] * size[1], dtype=np.int8)).tolist()
    return msg


def test_markerarray_has_two_markers_with_ids_0_and_1():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    assert len(arr.markers) == 2
    ids = sorted(m.id for m in arr.markers)
    assert ids == [0, 1]


def test_empty_frontiers_yields_add_action_with_no_points():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert frontier_marker.type == Marker.CUBE_LIST
    assert frontier_marker.action == Marker.ADD
    assert list(frontier_marker.points) == []
    # Red colour
    assert frontier_marker.color.r == 1.0
    assert frontier_marker.color.g == 0.0
    assert frontier_marker.color.b == 0.0
    assert frontier_marker.color.a == 0.7
    # Scale matches map resolution
    assert frontier_marker.scale.x == msg.info.resolution
    assert frontier_marker.scale.y == msg.info.resolution


def test_frontiers_become_cube_list_points():
    msg = _make_map(resolution=0.1, origin=(0.0, 0.0))
    # cell (row=1, col=2) → world (x = (2 + 0.4999) * 0.1, y = (1 + 0.4999) * 0.1)
    arr = build_frontier_markers([(1, 2)], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert len(frontier_marker.points) == 1
    p = frontier_marker.points[0]
    assert abs(p.x - (2 + 0.4999) * 0.1) < 1e-6
    assert abs(p.y - (1 + 0.4999) * 0.1) < 1e-6


def test_goal_none_sets_delete_action():
    msg = _make_map()
    arr = build_frontier_markers([], None, msg)
    goal_marker = next(m for m in arr.markers if m.id == 1)
    assert goal_marker.action == Marker.DELETE


def test_goal_sets_blue_cube_at_world_position():
    msg = _make_map(resolution=0.1, origin=(0.0, 0.0))
    arr = build_frontier_markers([], (2, 3), msg)
    goal_marker = next(m for m in arr.markers if m.id == 1)
    assert goal_marker.type == Marker.CUBE
    assert goal_marker.action == Marker.ADD
    # Blue
    assert goal_marker.color.r == 0.0
    assert goal_marker.color.g == 0.0
    assert goal_marker.color.b == 1.0
    assert goal_marker.color.a == 0.9
    # Scale = 1.5 * resolution in x/y
    assert abs(goal_marker.scale.x - 1.5 * 0.1) < 1e-9
    assert abs(goal_marker.scale.y - 1.5 * 0.1) < 1e-9
    # Position at cell (row=2, col=3)
    assert abs(goal_marker.pose.position.x - (3 + 0.4999) * 0.1) < 1e-6
    assert abs(goal_marker.pose.position.y - (2 + 0.4999) * 0.1) < 1e-6
    # Identity orientation
    assert goal_marker.pose.orientation.w == 1.0


def test_frame_id_and_namespace():
    msg = _make_map()
    arr = build_frontier_markers([(0, 0)], (0, 0), msg, frame_id='custom_frame')
    for m in arr.markers:
        assert m.header.frame_id == 'custom_frame'
        assert m.ns == 'frontier_viz'


def test_frontier_marker_has_identity_orientation():
    msg = _make_map()
    arr = build_frontier_markers([(0, 0)], None, msg)
    frontier_marker = next(m for m in arr.markers if m.id == 0)
    assert frontier_marker.pose.orientation.w == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/tb3_slam/test/test_frontier_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tb3_slam.frontier_viz'`.

- [ ] **Step 3: Implement `frontier_viz.py`**

Create `src/tb3_slam/tb3_slam/frontier_viz.py`:

```python
"""Build RViz markers for detected frontier cells and the current Nav2 goal.

Kept separate from ``frontier_utils`` so the pure-computation module does not
pull in ``visualization_msgs``/``geometry_msgs``.
"""
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from tb3_slam.frontier_utils import grid_to_world

_NAMESPACE = 'frontier_viz'
_FRONTIER_ID = 0
_GOAL_ID = 1


def _new_marker(
    marker_id: int,
    marker_type: int,
    frame_id: str,
    stamp: Time | None,
) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = _NAMESPACE
    m.id = marker_id
    m.type = marker_type
    m.pose.orientation.w = 1.0
    return m


def build_frontier_markers(
    frontiers: list[tuple[int, int]],
    goal: tuple[int, int] | None,
    map_msg: OccupancyGrid,
    frame_id: str = 'map',
    stamp: Time | None = None,
) -> MarkerArray:
    """Return a MarkerArray with a red CUBE_LIST for frontiers + blue CUBE for the goal.

    - ``frontiers``: list of (row, col) cells that the explorer flagged as frontiers.
    - ``goal``: current Nav2 goal cell (row, col), or ``None`` if no active goal.
    - ``map_msg``: the OccupancyGrid the cells were computed from (used for
      resolution + origin).

    The returned array always contains exactly two markers (ids 0 and 1). When
    ``goal`` is ``None`` the goal marker uses ``Marker.DELETE`` so the previous
    blue cube disappears in RViz.
    """
    res = map_msg.info.resolution

    frontier_marker = _new_marker(_FRONTIER_ID, Marker.CUBE_LIST, frame_id, stamp)
    frontier_marker.action = Marker.ADD
    frontier_marker.scale.x = res
    frontier_marker.scale.y = res
    frontier_marker.scale.z = 0.02
    frontier_marker.color.r = 1.0
    frontier_marker.color.g = 0.0
    frontier_marker.color.b = 0.0
    frontier_marker.color.a = 0.7
    for cell in frontiers:
        x, y = grid_to_world(cell, map_msg)
        p = Point()
        p.x = x
        p.y = y
        p.z = 0.0
        frontier_marker.points.append(p)

    goal_marker = _new_marker(_GOAL_ID, Marker.CUBE, frame_id, stamp)
    if goal is None:
        goal_marker.action = Marker.DELETE
    else:
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 1.5 * res
        goal_marker.scale.y = 1.5 * res
        goal_marker.scale.z = 0.05
        goal_marker.color.r = 0.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        goal_marker.color.a = 0.9
        gx, gy = grid_to_world(goal, map_msg)
        goal_marker.pose.position.x = gx
        goal_marker.pose.position.y = gy
        goal_marker.pose.position.z = 0.025  # sit on top of frontier cubes

    arr = MarkerArray()
    arr.markers = [frontier_marker, goal_marker]
    return arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/tb3_slam/test/test_frontier_viz.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tb3_slam/tb3_slam/frontier_viz.py src/tb3_slam/test/test_frontier_viz.py
git commit -m "feat(viz): add build_frontier_markers helper"
```

---

## Task 3: Declare ROS package dependencies for the new imports

**Files:**
- Modify: `src/tb3_slam/package.xml:13` (add `visualization_msgs`, `geometry_msgs`)
- Modify: `src/tb3_reinforced_exploration/package.xml:15` (add `visualization_msgs`)

`tb3_exploration/package.xml` already declares `visualization_msgs` — no change.

- [ ] **Step 1: Update `tb3_slam/package.xml`**

Replace the dependency block (lines 10-12):

```xml
  <depend>rclpy</depend>
  <depend>nav_msgs</depend>
  <depend>python3-numpy</depend>
```

with:

```xml
  <depend>rclpy</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>python3-numpy</depend>
```

- [ ] **Step 2: Update `tb3_reinforced_exploration/package.xml`**

Replace the dependency block (lines 10-15):

```xml
  <depend>rclpy</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>nav2_msgs</depend>
  <depend>python3-numpy</depend>
```

with:

```xml
  <depend>rclpy</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>nav2_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>python3-numpy</depend>
```

- [ ] **Step 3: Commit**

```bash
git add src/tb3_slam/package.xml src/tb3_reinforced_exploration/package.xml
git commit -m "chore: declare visualization_msgs dependency for frontier viz"
```

---

## Task 4: Wire `/frontiers_viz` publisher into `FrontierExplorer`

**Files:**
- Modify: `src/tb3_exploration/tb3_exploration/frontier_explorer.py` (imports, `__init__`, `_map_callback`)

- [ ] **Step 1: Add the MarkerArray import and viz helper import**

At the top of `src/tb3_exploration/tb3_exploration/frontier_explorer.py`, replace the import block (lines 9-26) with:

```python
import rclpy
import tf_transformations
from tf2_ros import Buffer, TransformListener
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from visualization_msgs.msg import MarkerArray

from tb3_slam.coverage_monitor import (
    DEFAULT_COVERAGE_THRESHOLD,
    check_coverage,
    load_expected_cells,
)
from tb3_slam.frontier_utils import centroids_from_map_debug, grid_to_world, world_to_grid
from tb3_slam.frontier_viz import build_frontier_markers
```

- [ ] **Step 2: Create the publisher in `__init__`**

Locate the ROS-interfaces block in `__init__` (currently lines 75-78):

```python
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
```

Replace with:

```python
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._viz_pub = self.create_publisher(MarkerArray, '/frontiers_viz', 10)
```

- [ ] **Step 3: Publish markers in `_map_callback`**

In `_map_callback` locate the block that calls `centroids_from_map_debug` (currently around lines 100-107):

```python
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids, stats = centroids_from_map_debug(msg, robot_grid_pos=robot_grid_pos)
        fresh = [c for c in centroids if c not in self._tried]
        self.get_logger().info(
            f'[frontier] {len(centroids)} centroid(s), {len(fresh)} fresh',
            throttle_duration_sec=5.0,
        )
```

Replace with:

```python
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids, stats = centroids_from_map_debug(msg, robot_grid_pos=robot_grid_pos)
        self._viz_pub.publish(
            build_frontier_markers(
                stats['frontiers'],
                self._current_target,
                msg,
                stamp=self.get_clock().now().to_msg(),
            )
        )
        fresh = [c for c in centroids if c not in self._tried]
        self.get_logger().info(
            f'[frontier] {len(centroids)} centroid(s), {len(fresh)} fresh',
            throttle_duration_sec=5.0,
        )
```

Note: `self._current_target` is already initialised to `None` in `__init__` (line 46), set in `_try_next_goal`, cleared on success/abort/timeout — no additional state work needed.

- [ ] **Step 4: Build and smoke-check the import**

Run from the repo root:

```bash
python3 -c "from tb3_exploration.frontier_explorer import FrontierExplorer; print('ok')"
```

(Requires the workspace to be built and sourced: `colcon build --packages-select tb3_slam tb3_exploration` then `source install/setup.bash`.)

Expected: `ok` printed. If you see `ModuleNotFoundError: visualization_msgs`, the ROS environment is not sourced.

- [ ] **Step 5: Commit**

```bash
git add src/tb3_exploration/tb3_exploration/frontier_explorer.py
git commit -m "feat(frontier_explorer): publish frontier + goal markers on /frontiers_viz"
```

---

## Task 5: Wire `/frontiers_viz` publisher into `RLExplorer`

**Files:**
- Modify: `src/tb3_reinforced_exploration/tb3_reinforced_exploration/rl_agent.py` (imports, `__init__`, `_map_callback`, `_on_goal_result`)

- [ ] **Step 1: Swap imports to use the debug variant and add viz imports**

At the top of `src/tb3_reinforced_exploration/tb3_reinforced_exploration/rl_agent.py`, replace the import block covering lines 17-33 with:

```python
import numpy as np
import rclpy
import tf_transformations
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from tb3_slam.coverage_monitor import (
    DEFAULT_COVERAGE_THRESHOLD,
    check_coverage,
    load_expected_cells,
)
from tb3_slam.frontier_utils import centroids_from_map_debug, grid_to_world, world_to_grid
from tb3_slam.frontier_viz import build_frontier_markers
```

(We remove the `centroids_from_map` import and use `centroids_from_map_debug` instead.)

- [ ] **Step 2: Add `_current_target_cell` state and publisher in `__init__`**

In `__init__`, locate the navigation-state block (currently around lines 145-147):

```python
        # ── Navigation state ───────────────────────────────────────
        self._goal_active = False
        self._current_map: OccupancyGrid | None = None
```

Replace with:

```python
        # ── Navigation state ───────────────────────────────────────
        self._goal_active = False
        self._current_map: OccupancyGrid | None = None
        self._current_target_cell: tuple[int, int] | None = None
```

Then locate the ROS2-interfaces block (currently around lines 169-173):

```python
        # ── ROS2 interfaces ────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
```

Replace with:

```python
        # ── ROS2 interfaces ────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._viz_pub = self.create_publisher(MarkerArray, '/frontiers_viz', 10)
```

- [ ] **Step 3: Switch `_map_callback` to the debug variant, publish markers, track goal cell**

In `_map_callback`, locate the detection block (currently around lines 196-204):

```python
        # ── Record reward for the previous transition ──────────────
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids = centroids_from_map(msg, robot_grid_pos=robot_grid_pos)
        if robot_pos is None:
            self.get_logger().warning('[rl] Could not get robot position – skipping step')
            return

        next_state = self._build_state(centroids, robot_pos)
```

Replace with:

```python
        # ── Record reward for the previous transition ──────────────
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids, stats = centroids_from_map_debug(msg, robot_grid_pos=robot_grid_pos)
        self._viz_pub.publish(
            build_frontier_markers(
                stats['frontiers'],
                self._current_target_cell,
                msg,
                stamp=self.get_clock().now().to_msg(),
            )
        )
        if robot_pos is None:
            self.get_logger().warning('[rl] Could not get robot position – skipping step')
            return

        next_state = self._build_state(centroids, robot_pos)
```

Then locate the action-dispatch block further down (currently around lines 221-237):

```python
        n_valid = min(len(centroids), MAX_CENTROIDS)
        action = self._select_action(next_state, n_valid)
        target_cell = centroids[action]
        self._current_map = msg

        x, y = grid_to_world(target_cell, msg)
        self.get_logger().info(
            f'[rl] action={action}  → ({x:.2f}, {y:.2f})'
            f'  ε={self._epsilon:.3f}  replay={len(self._replay)}'
        )
        self._send_goal(x, y)

        # Store current step info for the next reward computation
        self._last_state = next_state
        self._last_action = action
        self._last_coverage = current_coverage
        self._goal_active = True
```

Replace with:

```python
        n_valid = min(len(centroids), MAX_CENTROIDS)
        action = self._select_action(next_state, n_valid)
        target_cell = centroids[action]
        self._current_map = msg
        self._current_target_cell = target_cell

        x, y = grid_to_world(target_cell, msg)
        self.get_logger().info(
            f'[rl] action={action}  → ({x:.2f}, {y:.2f})'
            f'  ε={self._epsilon:.3f}  replay={len(self._replay)}'
        )
        self._send_goal(x, y)

        # Store current step info for the next reward computation
        self._last_state = next_state
        self._last_action = action
        self._last_coverage = current_coverage
        self._goal_active = True
```

- [ ] **Step 4: Clear `_current_target_cell` when the goal ends**

Locate `_on_goal_result` (currently around lines 362-371):

```python
    def _on_goal_result(self, future: Future) -> None:
        status: int = future.result().status
        status_name = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'status={status}')
        self.get_logger().info(f'[goal] {status_name}')
        self._goal_active = False
        # The next _map_callback will compute the reward and select a new action.
```

Replace with:

```python
    def _on_goal_result(self, future: Future) -> None:
        status: int = future.result().status
        status_name = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'status={status}')
        self.get_logger().info(f'[goal] {status_name}')
        self._goal_active = False
        self._current_target_cell = None
        # The next _map_callback will compute the reward and select a new action.
```

- [ ] **Step 5: Smoke-check the import**

Run (after `colcon build --packages-select tb3_slam tb3_reinforced_exploration` and sourcing):

```bash
python3 -c "from tb3_reinforced_exploration.rl_agent import RLExplorer; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/tb3_reinforced_exploration/tb3_reinforced_exploration/rl_agent.py
git commit -m "feat(rl_agent): publish frontier + goal markers on /frontiers_viz"
```

---

## Task 6: Manual verification in RViz

**Files:** none (runtime check only)

- [ ] **Step 1: Rebuild the workspace**

```bash
cd /home/tombielert/projects/slam_rl
colcon build --packages-select tb3_slam tb3_exploration tb3_reinforced_exploration
source install/setup.bash
```

Expected: build succeeds for all three packages.

- [ ] **Step 2: Re-run existing pytest suites**

```bash
pytest src/tb3_slam/test/ -v
```

Expected: the two new test files pass (`test_frontier_utils_stats.py`, `test_frontier_viz.py`). Lint tests (`test_copyright.py` etc.) behave as before.

- [ ] **Step 3: Launch the classic explorer**

```bash
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic
```

- [ ] **Step 4: Add the marker display in RViz**

In the RViz window that opens automatically:

1. Click **Add** → tab **By topic** → pick `MarkerArray` under `/frontiers_viz`.
2. Confirm red cubes appear along the boundary between mapped and unmapped areas as Cartographer progresses.
3. Confirm a blue cube sits on the currently active Nav2 goal cell and disappears / moves when the goal changes.

If red cubes do not appear: run `ros2 topic hz /frontiers_viz` — it should tick roughly at map-update rate. If it ticks but nothing renders, check that the RViz Marker display's "Fixed Frame" is `map`.

- [ ] **Step 5: Repeat for the RL explorer**

Stop the previous launch, then:

```bash
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=rl
```

Repeat the RViz steps — same red/blue behaviour should be visible. The blue cube updates whenever the DQN picks a new action.

- [ ] **Step 6: Final commit (if no code changes needed)**

If verification surfaced no fixes, nothing to commit. Otherwise, fix and commit as appropriate.

---

## Self-review checklist (already applied)

- Spec sections covered: frontiers exposed in stats (Task 1), marker helper (Task 2), deps (Task 3), FrontierExplorer wiring (Task 4), RLExplorer wiring + `_current_target_cell` state (Task 5), manual verification (Task 6).
- No placeholders — every code block is complete.
- Type consistency — `build_frontier_markers(frontiers, goal, map_msg, frame_id, stamp)` signature is used identically everywhere; `_current_target_cell` is named the same in state, dispatch, and clear.
- Spec requirement for identity quaternion on both markers implemented via `_new_marker` helper and asserted in tests.
