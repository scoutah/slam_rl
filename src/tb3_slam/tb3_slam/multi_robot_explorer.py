"""Shared base node for multi-robot frontier exploration.

Common infrastructure for the classical and the RL exploration methods. Each
method lives in its own package and subclasses :class:`MultiRobotExplorerNode`,
overriding only :meth:`select_goals`. The base owns all ROS I/O; the subclass
owns the strategy. The two methods share no strategy code and may diverge
freely.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import BackUp, ComputePathToPose, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.client import Client
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from tb3_slam.coverage_monitor import (
    check_coverage,
    DEFAULT_COVERAGE_THRESHOLD,
    load_expected_cells,
)
from tb3_slam.frontier_utils import (
    centroids_from_map_debug,
    grid_to_world,
    world_to_grid,
)
from tb3_slam.frontier_viz import build_frontier_markers
from tb3_slam.multi_robot import frame, MERGED_MAP_FRAME, ROBOT_NAMESPACES, topic
from tb3_slam.path_cost import cost_key, path_length
from tf2_geometry_msgs import do_transform_pose
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from visualization_msgs.msg import MarkerArray

_TF_QOS = QoSProfile(
    depth=100,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
)
_TF_STATIC_QOS = QoSProfile(
    depth=100,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)

# Per-robot goal-marker colours matching the robot scan colours in sim_slam.rviz
# (r1 green, r2 orange); robots beyond the table fall back to green.
_GOAL_COLOR_FALLBACK = (0.0, 1.0, 0.0)
_ROBOT_GOAL_COLORS: dict[str, tuple[float, float, float]] = defaultdict(
    lambda: _GOAL_COLOR_FALLBACK,
    {'r1': (0.0, 1.0, 0.0), 'r2': (1.0, 160 / 255, 0.0)},
)


@dataclass(frozen=True)
class Goal:
    """A navigation goal in the merged ``map`` frame (metres)."""

    x: float
    y: float
    group_size: int | None = None
    path_len: float | None = None


@dataclass(frozen=True)
class RobotState:
    """Pose and goal state of one robot in the merged ``map`` frame."""

    name: str
    x: float
    y: float
    busy: bool = False
    current_goal: 'Goal | None' = None


@dataclass
class FrontierCandidate:
    """An approach goal and its underlying frontier cells on the merged map.

    The group cells retain frontier identity when robots need different
    approach points for the same group.
    """

    cell: tuple[int, int]
    x: float
    y: float
    group_size: int
    frontier_cells: frozenset[tuple[int, int]] = frozenset()


# Last few frontier analyses, keyed by (map message identity, robot cell). The RViz
# markers and the goal selection analyse the same merged map for the same idle
# robot; the pipeline is pure Python and costly, so it runs once. Map messages are
# treated as immutable; the cache holds a reference so an id is never reused.
_ANALYSIS_CACHE_SIZE = 4
_analysis_cache: dict[tuple[int, tuple[int, int] | None], tuple[OccupancyGrid, tuple]] = {}


def frontier_analysis(
    merged_map: OccupancyGrid,
    robot_grid_pos: tuple[int, int] | None,
) -> tuple[list[tuple[int, int]], dict]:
    """``centroids_from_map_debug`` memoised per map message and robot cell.

    The returned centroids/stats are shared between callers: do not mutate them.
    """
    key = (id(merged_map), robot_grid_pos)
    hit = _analysis_cache.get(key)
    if hit is not None:
        return hit[1]
    result = centroids_from_map_debug(merged_map, robot_grid_pos=robot_grid_pos)
    if len(_analysis_cache) >= _ANALYSIS_CACHE_SIZE:
        del _analysis_cache[next(iter(_analysis_cache))]    # oldest first
    _analysis_cache[key] = (merged_map, result)
    return result


def candidates_for_robot(
    merged_map: OccupancyGrid,
    robot: RobotState,
) -> list[FrontierCandidate]:
    """Navigable frontier centroids reachable from ``robot`` on the merged map."""
    robot_grid_pos = world_to_grid(robot.x, robot.y, merged_map)
    centroids, stats = frontier_analysis(merged_map, robot_grid_pos)
    sizes = stats.get('centroid_group_sizes', [])
    candidates: list[FrontierCandidate] = []
    for i, cell in enumerate(centroids):
        wx, wy = grid_to_world(cell, merged_map)
        group_size = sizes[i] if i < len(sizes) else 0
        candidates.append(FrontierCandidate(
            cell=cell, x=wx, y=wy, group_size=group_size,
            frontier_cells=frozenset(stats['centroid_frontiers'][i]),
        ))
    return candidates


class IdleTracker:
    """How long the whole team has been without any active goal (sim seconds).

    Feeds the ``exhausted`` verdict: if no robot is navigating or recovering for
    a while although coverage is short of the threshold, the strategy has no
    assignable frontier left and more waiting will not add progress.
    """

    def __init__(self) -> None:
        self._idle_since: float | None = None

    def update(self, busy: bool, now: float) -> float:
        """Record the team state at ``now``; return seconds idle so far (0 if busy)."""
        if busy:
            self._idle_since = None
            return 0.0
        if self._idle_since is None:
            self._idle_since = now
        return now - self._idle_since


@dataclass
class _RobotIO:
    ns: str
    base_frame: str
    map_frame: str
    nav_client: ActionClient
    path_client: ActionClient
    backup_client: ActionClient
    spin_client: ActionClient
    clear_local: Client
    clear_global: Client
    goal_active: bool = False
    goal_handle: ClientGoalHandle | None = None
    goal_generation: int = 0
    current_goal: Goal | None = None
    dispatch_pos: tuple[float, float] | None = None
    dispatch_time: float | None = None
    last_progress_pos: tuple[float, float] | None = None
    last_progress_time: float | None = None
    # Consecutive stalled goals without a success; drives the escalating recovery.
    stall_streak: int = 0
    recovering: bool = False
    recovery_deadline: float = 0.0


class MultiRobotExplorerNode(Node):
    """Base node owning all ROS I/O; subclasses implement :meth:`select_goals`."""

    def __init__(self, node_name: str = 'multi_robot_explorer') -> None:
        super().__init__(node_name)

        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')
        self.declare_parameter('map_frame', MERGED_MAP_FRAME)
        self.declare_parameter('decision_period', 2.0)
        self.declare_parameter('goal_timeout', 45.0)
        self.declare_parameter('no_progress_timeout', 8.0)
        self.declare_parameter('progress_step', 0.1)
        self.declare_parameter('min_progress', 0.2)
        self.declare_parameter('recovery_timeout', 10.0)
        # Team idle this long (sim s) before any goal was ever assigned again =>
        # episode ends as "exhausted". 0 disables. Must stay below the episode
        # manager's stuck window (90 s) minus the time a last goal can still run.
        self.declare_parameter('exhausted_idle_s', 15.0)
        self.declare_parameter('data_dir', os.environ.get('SLAM_RL_DATA_DIR', ''))
        self.declare_parameter('cost_cache_ttl', 5.0)
        self.declare_parameter('viz_min_period', 2.0)

        self._coverage_threshold: float = self.get_parameter('coverage_threshold').value
        self._episode_file: str = self.get_parameter('episode_file').value
        self._map_frame: str = self.get_parameter('map_frame').value
        self._goal_timeout: float = self.get_parameter('goal_timeout').value
        self._no_progress_timeout: float = self.get_parameter('no_progress_timeout').value
        self._progress_step: float = self.get_parameter('progress_step').value
        self._min_progress: float = self.get_parameter('min_progress').value
        self._recovery_timeout: float = self.get_parameter('recovery_timeout').value
        self._exhausted_idle_s: float = self.get_parameter('exhausted_idle_s').value
        self._idle = IdleTracker()
        self._ever_dispatched = False
        self._coverage = 0.0
        self._data_dir: str = self.get_parameter('data_dir').value
        self._cost_cache_ttl: float = self.get_parameter('cost_cache_ttl').value
        self._viz_min_period: float = self.get_parameter('viz_min_period').value
        self._viz_last_wall = -math.inf

        self._coverage_done = False
        self._latest_map: OccupancyGrid | None = None
        self._cost_cache: dict[tuple, tuple[float | None, float]] = {}
        self._cost_pending: set[tuple] = set()
        self._cycle_positions: dict[str, tuple[float, float]] = {}

        self._tf_buffer = Buffer()
        self._setup_tf_listeners()

        self._robots: dict[str, _RobotIO] = {}
        for ns in ROBOT_NAMESPACES:
            self._robots[ns] = _RobotIO(
                ns=ns,
                base_frame=frame(ns, 'base_link'),
                map_frame=frame(ns, 'map'),
                nav_client=ActionClient(self, NavigateToPose, topic(ns, 'navigate_to_pose')),
                path_client=ActionClient(
                    self, ComputePathToPose, topic(ns, 'compute_path_to_pose')
                ),
                backup_client=ActionClient(self, BackUp, topic(ns, 'backup')),
                spin_client=ActionClient(self, Spin, topic(ns, 'spin')),
                clear_local=self.create_client(
                    ClearEntireCostmap, topic(ns, 'local_costmap/clear_entirely_local_costmap')),
                clear_global=self.create_client(
                    ClearEntireCostmap, topic(ns, 'global_costmap/clear_entirely_global_costmap')),
            )

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning(
                '[coverage] No world metadata – coverage monitoring disabled'
            )
            self._expected_cells = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected cells: {expected} '
                f'(threshold {self._coverage_threshold:.0%})'
            )

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, map_qos,
        )
        self._viz_pub = self.create_publisher(MarkerArray, '/frontiers_viz', 10)

        self.create_timer(self.get_parameter('decision_period').value, self._decide)
        self.create_timer(1.0, self._supervise_goals)
        self.create_timer(1.0, self._watch_episode_end)
        self.create_timer(1.0, self._check_exhausted)

        self.get_logger().info(f'{node_name} ready – waiting for merged /map ...')

    def _watch_episode_end(self) -> None:
        """Stop acting once the episode is over (file written by episode_manager).

        The base only ends on its own coverage check; episode_manager also ends
        on stuck/timeout. Honour that file so robots stop being re-tasked after
        the episode is declared over.
        """
        if self._coverage_done or not self._episode_file:
            return
        if Path(self._episode_file).exists():
            self._coverage_done = True
            for io in self._robots.values():
                if io.goal_active:
                    self._cancel_goal(io, reason='episode ended')

    def _setup_tf_listeners(self) -> None:
        """Feed the shared TF buffer from each robot's /<ns>/tf[_static] tree.

        Robots keep their tree on /<ns>/tf (not global /tf); the static
        map→<ns>/map offset lives on global /tf_static. Subscribing to all
        three resolves map→<ns>/base_link without the viz-only tf_relay.
        """
        self.create_subscription(
            TFMessage, '/tf_static', self._tf_static_cb, _TF_STATIC_QOS,
        )
        for ns in ROBOT_NAMESPACES:
            self.create_subscription(
                TFMessage, f'/{ns}/tf', self._tf_cb, _TF_QOS,
            )
            self.create_subscription(
                TFMessage, f'/{ns}/tf_static', self._tf_static_cb, _TF_STATIC_QOS,
            )

    def _tf_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform(transform, 'explorer')

    def _tf_static_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform_static(transform, 'explorer')

    def select_goals(
        self,
        merged_map: OccupancyGrid,
        robots: list[RobotState],
    ) -> dict[str, Goal | None]:
        """Return a goal (or ``None`` to leave unchanged) for each robot.

        Baseline: selects nothing, so robots stay put. Subclasses override this
        with their strategy, using :meth:`candidates_for` and :meth:`path_cost`.
        """
        return {robot.name: None for robot in robots}

    def candidates_for(self, robot: RobotState) -> list[FrontierCandidate]:
        """Frontier centroids reachable from ``robot`` on the latest merged map."""
        if self._latest_map is None:
            return []
        return candidates_for_robot(self._latest_map, robot)

    def path_cost(self, robot: RobotState, x: float, y: float) -> float | None:
        """Return the cached Nav2 path length from ``robot`` to ``(x, y)``, or ``None``.

        A cache miss returns ``None`` and triggers a background plan request so
        the value is ready on a later decision cycle; callers fall back to a
        geometric estimate while it is unknown.
        """
        start = self._cycle_positions.get(robot.name)
        if start is None:
            return None
        key = cost_key(robot.name, start[0], start[1], x, y)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached[0]
        self._request_path_cost(robot.name, start, (x, y), key)
        return None

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self._latest_map = msg
        self._publish_frontier_markers(msg)
        if self._coverage_done or self._expected_cells <= 0:
            return
        coverage, _known = check_coverage(msg, self._expected_cells)
        self._coverage = coverage
        if coverage >= self._coverage_threshold:
            self._finish(coverage)

    def _publish_frontier_markers(self, msg: OccupancyGrid) -> None:
        """Publish the union of per-robot frontiers/centroids for RViz.

        Each robot's active goal is drawn in that robot's RViz colour (r1 green,
        r2 orange) so the two goals are distinguishable.

        Display only, so it costs nothing without a subscriber and is rate
        limited (wall clock, ``viz_min_period``) otherwise: at higher sim speed
        maps arrive faster than the pipeline can analyse them.
        """
        if self._viz_pub.get_subscription_count() == 0:
            return
        wall = time.monotonic()
        if wall - self._viz_last_wall < self._viz_min_period:
            return
        self._viz_last_wall = wall
        frontier_cells: set[tuple[int, int]] = set()
        centroid_cells: set[tuple[int, int]] = set()
        extra_goals: list[
            tuple[tuple[int, int] | None, tuple[float, float, float]]
        ] = []

        for ns, io in self._robots.items():
            pos = self._robot_pos(io)
            if pos is None:
                continue
            robot_grid_pos = world_to_grid(pos[0], pos[1], msg)
            if robot_grid_pos is None:
                continue
            centroids, stats = frontier_analysis(msg, robot_grid_pos)
            frontier_cells.update(stats['frontiers'])
            centroid_cells.update(centroids)
            goal_cell: tuple[int, int] | None = None
            if io.current_goal is not None:
                goal_cell = world_to_grid(
                    io.current_goal.x, io.current_goal.y, msg,
                )
            extra_goals.append((goal_cell, _ROBOT_GOAL_COLORS[ns]))

        self._viz_pub.publish(
            build_frontier_markers(
                sorted(frontier_cells),
                None,
                msg,
                frame_id=self._map_frame,
                stamp=self.get_clock().now().to_msg(),
                centroids=sorted(centroid_cells),
                extra_goals=extra_goals,
            )
        )

    def _robot_pos(self, io: _RobotIO) -> tuple[float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform(
                self._map_frame, io.base_frame, rclpy.time.Time(),
            )
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:  # noqa: BLE001 - TF raises many unrelated types
            self.get_logger().warning(
                f'[tf] no pose for {io.ns}: {e}', throttle_duration_sec=5.0,
            )
            return None

    def _decide(self) -> None:
        if self._coverage_done or self._latest_map is None:
            return

        states: list[RobotState] = []
        positions: dict[str, tuple[float, float]] = {}
        for ns, io in self._robots.items():
            pos = self._robot_pos(io)
            if pos is None:
                continue
            positions[ns] = pos
            states.append(RobotState(
                name=ns,
                x=pos[0],
                y=pos[1],
                busy=io.goal_active or io.recovering,
                current_goal=io.current_goal,
            ))

        if not states:
            self.get_logger().warning(
                '[explorer] no robot poses yet – waiting for Cartographer to '
                'publish <ns>/map->odom (benign for the first few seconds)',
                throttle_duration_sec=5.0,
            )
            return

        self._expire_cost_cache()
        self._cycle_positions = positions
        try:
            goals = self.select_goals(self._latest_map, states)
        except Exception as e:  # noqa: BLE001 - a strategy bug must not kill the node
            self.get_logger().error(f'[explorer] select_goals raised: {e}')
            return

        for ns, goal in goals.items():
            io = self._robots.get(ns)
            if io is None or goal is None:
                continue
            if io.goal_active:
                self._cancel_goal(io, reason='re-tasked by strategy')
            self._dispatch(io, goal, positions.get(ns))

    def _expire_cost_cache(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        stale = [k for k, (_, ts) in self._cost_cache.items() if now - ts > self._cost_cache_ttl]
        for k in stale:
            del self._cost_cache[k]

    def _request_path_cost(
        self, ns: str, start: tuple[float, float], goal: tuple[float, float], key: tuple,
    ) -> None:
        if key in self._cost_pending:
            return
        io = self._robots.get(ns)
        if io is None or not io.path_client.server_is_ready():
            return
        req = ComputePathToPose.Goal()
        req.use_start = True
        start_pose = self._goal_in_robot_frame(io, start[0], start[1])
        goal_pose = self._goal_in_robot_frame(io, goal[0], goal[1])
        if start_pose is None or goal_pose is None:
            return
        req.start = start_pose
        req.goal = goal_pose
        self._cost_pending.add(key)
        future = io.path_client.send_goal_async(req)
        future.add_done_callback(lambda f, k=key: self._on_path_goal_response(f, k))

    def _on_path_goal_response(self, future: Future, key: tuple) -> None:
        try:
            handle: ClientGoalHandle = future.result()
        except Exception:  # noqa: BLE001
            self._cost_pending.discard(key)
            return
        if not handle.accepted:
            self._cost_pending.discard(key)
            return
        handle.get_result_async().add_done_callback(
            lambda f, k=key: self._on_path_result(f, k)
        )

    def _on_path_result(self, future: Future, key: tuple) -> None:
        self._cost_pending.discard(key)
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            path = future.result().result.path
        except Exception:  # noqa: BLE001
            self._cost_cache[key] = (None, now)
            return
        self._cost_cache[key] = (path_length(path), now)

    def _pose_stamped(self, x: float, y: float, frame_id: str | None = None) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id or self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        return pose

    def _goal_in_robot_frame(self, io: _RobotIO, x: float, y: float) -> PoseStamped | None:
        """Convert a merged-``map`` point to a goal in the robot's ``<ns>/map`` frame.

        Each robot's Nav2 only knows its own ``<ns>/map`` tree (TF on
        ``/<ns>/tf``), so a goal stamped ``map`` fails to transform. Returns
        ``None`` if ``map → <ns>/map`` is not yet in the buffer.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                io.map_frame, self._map_frame, rclpy.time.Time(),
            )
        except Exception as e:  # noqa: BLE001 - TF raises many unrelated types
            self.get_logger().warning(
                f'[{io.ns}] no {self._map_frame}->{io.map_frame} transform: {e}',
                throttle_duration_sec=5.0,
            )
            return None
        pose = do_transform_pose(self._pose_stamped(x, y).pose, tf)
        out = PoseStamped()
        out.header.frame_id = io.map_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose = pose
        return out

    def _dispatch(
        self, io: _RobotIO, goal: Goal, robot_pos: tuple[float, float] | None,
    ) -> None:
        if io.recovering:
            return      # a recovery behaviour owns the drive; the strategy retries next cycle
        if not io.nav_client.server_is_ready():
            self.get_logger().warning(
                f'[{io.ns}] Nav2 not ready – skipping dispatch', throttle_duration_sec=5.0,
            )
            return

        goal_pose = self._goal_in_robot_frame(io, goal.x, goal.y)
        if goal_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        self._ever_dispatched = True
        io.goal_active = True
        io.current_goal = goal
        io.dispatch_pos = robot_pos
        io.dispatch_time = now
        io.last_progress_pos = robot_pos
        io.last_progress_time = now
        io.goal_generation += 1
        generation = io.goal_generation

        msg = NavigateToPose.Goal()
        msg.pose = goal_pose

        dist_str = (
            f'{math.hypot(goal.x - robot_pos[0], goal.y - robot_pos[1]):.2f}m'
            if robot_pos is not None else 'n/a'
        )
        self.get_logger().info(
            f'[{io.ns}] → ({goal.x:.2f}, {goal.y:.2f}) dist={dist_str} '
            f'group={goal.group_size}'
        )
        future: Future = io.nav_client.send_goal_async(msg)
        future.add_done_callback(
            lambda f, ns=io.ns, g=generation: self._on_goal_response(f, ns, g)
        )

    def _on_goal_response(self, future: Future, ns: str, generation: int) -> None:
        io = self._robots[ns]
        handle: ClientGoalHandle = future.result()
        if generation != io.goal_generation:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.get_logger().warning(f'[{ns}] goal REJECTED')
            io.goal_handle = None
            io.goal_active = False
            io.current_goal = None
            return
        io.goal_handle = handle
        self.get_logger().info(f'[{ns}] goal ACCEPTED – navigating ...')
        handle.get_result_async().add_done_callback(
            lambda f, n=ns, g=generation: self._on_goal_result(f, n, g)
        )

    def _on_goal_result(self, future: Future, ns: str, generation: int) -> None:
        io = self._robots[ns]
        if generation != io.goal_generation:
            return
        result = future.result()
        status: int = result.status
        io.goal_handle = None
        io.goal_active = False
        moved = self._progress_since_dispatch(io)
        moved_str = f'{moved:.2f}m' if moved is not None else 'n/a'

        if status == GoalStatus.STATUS_SUCCEEDED:
            if moved is not None and moved < self._min_progress:
                self.get_logger().warning(
                    f'[{ns}] SUCCEEDED but moved only {moved_str} '
                    f'(< {self._min_progress}m)'
                )
            else:
                io.stall_streak = 0
                self.get_logger().info(f'[{ns}] SUCCEEDED moved={moved_str}')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning(
                f'[{ns}] goal ABORTED (moved={moved_str}) {self._nav_error(result)}'
            )
        else:
            self.get_logger().warning(f'[{ns}] goal ended status={status}')
        io.current_goal = None

    @staticmethod
    def _nav_error(result) -> str:
        """Nav2 ``error_code``/``error_msg`` from a NavigateToPose result, if any."""
        res = getattr(result, 'result', None)
        code = getattr(res, 'error_code', 0)
        msg = getattr(res, 'error_msg', '') or ''
        return f'code={code} msg={msg!r}' if code or msg else 'no error detail'

    def _cancel_goal(self, io: _RobotIO, reason: str) -> None:
        self.get_logger().info(f'[{io.ns}] cancelling goal – {reason}')
        io.goal_generation += 1
        try:
            if io.goal_handle is not None:
                io.goal_handle.cancel_goal_async()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f'[{io.ns}] cancel failed: {e}')
        io.goal_handle = None
        io.goal_active = False
        io.current_goal = None

    def _supervise_goals(self) -> None:
        if self._coverage_done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        for io in self._robots.values():
            if io.recovering and now > io.recovery_deadline:
                io.recovering = False       # watchdog: behaviour result never arrived
            if not io.goal_active or io.dispatch_time is None:
                continue
            pos = self._robot_pos(io)
            if pos is not None:
                if (io.last_progress_pos is None
                        or math.dist(pos, io.last_progress_pos) >= self._progress_step):
                    io.last_progress_pos = pos
                    io.last_progress_time = now
            elapsed = now - io.dispatch_time
            stalled = (
                pos is not None and io.last_progress_time is not None
                and now - io.last_progress_time >= self._no_progress_timeout
            )
            if stalled or elapsed > self._goal_timeout:
                reason = 'NO PROGRESS' if stalled else 'TIMEOUT'
                self.get_logger().warning(
                    f'[{io.ns}] {reason} after {elapsed:.0f}s{self._stall_context(io, pos)}')
                self._cancel_goal(io, reason=reason)
                if stalled:
                    self._recover(io)

    def _recover(self, io: _RobotIO) -> None:
        """Escalating self-rescue so a stalled robot never just sits there.

        Always clears both costmaps (stale/phantom obstacles). On the 2nd stall in
        a row it also backs up, on the 3rd it spins, then the cycle repeats until
        a goal succeeds again (``stall_streak`` reset). While a behaviour runs
        ``io.recovering`` blocks new goals so Nav2 and the behaviour server do
        not fight over ``cmd_vel``.
        """
        io.stall_streak += 1
        step = io.stall_streak % 3
        self._clear_costmaps(io)
        if step == 1:
            self.get_logger().info(
                f'[{io.ns}] recovery #{io.stall_streak}: cleared costmaps')
            return
        if step == 2:
            client, goal, name = io.backup_client, BackUp.Goal(), 'back up'
            goal.target = Point(x=0.3)
            goal.speed = 0.1
            goal.time_allowance = Duration(sec=8)
        else:
            client, goal, name = io.spin_client, Spin.Goal(), 'spin'
            goal.target_yaw = 1.57
            goal.time_allowance = Duration(sec=8)
        if not client.server_is_ready():
            return
        self.get_logger().info(f'[{io.ns}] recovery #{io.stall_streak}: {name}')
        io.recovering = True
        io.recovery_deadline = self.get_clock().now().nanoseconds * 1e-9 + self._recovery_timeout
        client.send_goal_async(goal).add_done_callback(
            lambda f, n=io.ns: self._on_recovery_response(f, n))

    def _clear_costmaps(self, io: _RobotIO) -> None:
        for client in (io.clear_local, io.clear_global):
            if client.service_is_ready():
                client.call_async(ClearEntireCostmap.Request())

    def _on_recovery_response(self, future: Future, ns: str) -> None:
        io = self._robots[ns]
        try:
            handle: ClientGoalHandle = future.result()
        except Exception:  # noqa: BLE001
            handle = None
        if handle is None or not handle.accepted:
            io.recovering = False
            return
        handle.get_result_async().add_done_callback(
            lambda f, n=ns: self._on_recovery_done(f, n))

    def _on_recovery_done(self, future: Future, ns: str) -> None:
        io = self._robots[ns]
        io.recovering = False
        try:
            self.get_logger().info(
                f'[{ns}] recovery finished status={future.result().status}')
        except Exception:  # noqa: BLE001
            pass

    def _stall_context(self, io: _RobotIO, pos: tuple[float, float] | None) -> str:
        """Where the robot, its goal and its teammates were when a goal was abandoned."""
        if pos is None:
            return ''
        text = f' at ({pos[0]:.2f}, {pos[1]:.2f})'
        if io.current_goal is not None:
            text += f' goal=({io.current_goal.x:.2f}, {io.current_goal.y:.2f})'
        for other in self._robots.values():
            other_pos = self._robot_pos(other) if other is not io else None
            if other_pos is not None:
                text += (f' {other.ns}=({other_pos[0]:.2f}, {other_pos[1]:.2f}) '
                         f'{math.dist(pos, other_pos):.2f}m away')
        return text

    def _progress_since_dispatch(self, io: _RobotIO) -> float | None:
        if io.dispatch_pos is None:
            return None
        now = self._robot_pos(io)
        if now is None:
            return None
        return math.hypot(now[0] - io.dispatch_pos[0], now[1] - io.dispatch_pos[1])

    def _check_exhausted(self) -> None:
        """End the episode as ``exhausted`` once the team has nothing left to do.

        Distinct from ``stuck`` (coverage stagnates, robots may still be trying):
        here every robot has been idle for ``exhausted_idle_s`` because the
        strategy assigns no goal any more, so the remaining unknown area is out
        of reach for this method. Saved worlds of such episodes are the ones worth
        re-running (see ``training_runner.py``).
        """
        if (self._coverage_done or self._exhausted_idle_s <= 0 or not self._ever_dispatched
                or self._latest_map is None or self._expected_cells <= 0):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        busy = any(io.goal_active or io.recovering for io in self._robots.values())
        if self._idle.update(busy, now) < self._exhausted_idle_s:
            return
        self.get_logger().info(
            f'[coverage] EXHAUSTED at {self._coverage:.1%}: no goal assigned for '
            f'{self._exhausted_idle_s:.0f}s'
        )
        self._coverage_done = True
        self._write_episode_file(f'exhausted (coverage={self._coverage:.1%})')

    def _write_episode_file(self, text: str) -> None:
        """Write the episode-end file unless one (e.g. from episode_manager) exists."""
        if self._episode_file and not Path(self._episode_file).exists():
            Path(self._episode_file).write_text(text)

    def _finish(self, coverage: float) -> None:
        self.get_logger().info(
            f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%}'
        )
        self._coverage_done = True
        self._write_episode_file(f'coverage={coverage:.1%}')
