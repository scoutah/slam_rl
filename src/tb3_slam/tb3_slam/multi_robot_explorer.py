"""Shared map/TF, Nav2 dispatch and goal supervision for standalone explorers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
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
from tb3_slam.multi_robot import frame, MERGED_MAP_FRAME, ROBOT_NAMESPACES, topic
from tb3_slam.path_cost import cost_key, path_length
from tf2_ros import Buffer, TransformListener


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


def candidates_for_robot(
    merged_map: OccupancyGrid,
    robot: RobotState,
) -> list[FrontierCandidate]:
    """Navigable frontier centroids reachable from ``robot`` on the merged map."""
    robot_grid_pos = world_to_grid(robot.x, robot.y, merged_map)
    centroids, stats = centroids_from_map_debug(merged_map, robot_grid_pos=robot_grid_pos)
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


@dataclass
class _RobotIO:
    ns: str
    base_frame: str
    nav_client: ActionClient
    path_client: ActionClient
    goal_active: bool = False
    goal_handle: ClientGoalHandle | None = None
    goal_generation: int = 0
    current_goal: Goal | None = None
    dispatch_pos: tuple[float, float] | None = None
    dispatch_time: float | None = None
    last_progress_pos: tuple[float, float] | None = None
    last_progress_time: float | None = None


class MultiRobotExplorerNode(Node):
    """Reusable map/TF, Nav2 dispatch and goal-supervision infrastructure."""

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
        self.declare_parameter('data_dir', os.environ.get('SLAM_RL_DATA_DIR', ''))
        self.declare_parameter('cost_cache_ttl', 5.0)

        self._coverage_threshold: float = self.get_parameter('coverage_threshold').value
        self._episode_file: str = self.get_parameter('episode_file').value
        self._map_frame: str = self.get_parameter('map_frame').value
        self._goal_timeout: float = self.get_parameter('goal_timeout').value
        self._no_progress_timeout: float = self.get_parameter('no_progress_timeout').value
        self._progress_step: float = self.get_parameter('progress_step').value
        self._min_progress: float = self.get_parameter('min_progress').value
        self._data_dir: str = self.get_parameter('data_dir').value
        self._cost_cache_ttl: float = self.get_parameter('cost_cache_ttl').value

        self._coverage_done = False
        self._latest_map: OccupancyGrid | None = None
        self._cost_cache: dict[tuple, tuple[float | None, float]] = {}
        self._cost_pending: set[tuple] = set()
        self._cycle_positions: dict[str, tuple[float, float]] = {}

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._robots: dict[str, _RobotIO] = {}
        for ns in ROBOT_NAMESPACES:
            self._robots[ns] = _RobotIO(
                ns=ns,
                base_frame=frame(ns, 'base_link'),
                nav_client=ActionClient(self, NavigateToPose, topic(ns, 'navigate_to_pose')),
                path_client=ActionClient(
                    self, ComputePathToPose, topic(ns, 'compute_path_to_pose')
                ),
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

        self.create_timer(self.get_parameter('decision_period').value, self._decide)
        self.create_timer(1.0, self._supervise_goals)

        self.get_logger().info(f'{node_name} ready – waiting for merged /map ...')

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
        if self._coverage_done or self._expected_cells <= 0:
            return
        coverage, _known = check_coverage(msg, self._expected_cells)
        if coverage >= self._coverage_threshold:
            self._finish(coverage)

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
                busy=io.goal_active,
                current_goal=io.current_goal,
            ))

        if not states:
            self.get_logger().warning(
                '[explorer] no robot poses yet – waiting for TF',
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
        req.start = self._pose_stamped(start[0], start[1])
        req.goal = self._pose_stamped(goal[0], goal[1])
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

    def _pose_stamped(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        return pose

    def _dispatch(
        self, io: _RobotIO, goal: Goal, robot_pos: tuple[float, float] | None,
    ) -> None:
        if not io.nav_client.server_is_ready():
            self.get_logger().warning(
                f'[{io.ns}] Nav2 not ready – skipping dispatch', throttle_duration_sec=5.0,
            )
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        io.goal_active = True
        io.current_goal = goal
        io.dispatch_pos = robot_pos
        io.dispatch_time = now
        io.last_progress_pos = robot_pos
        io.last_progress_time = now
        io.goal_generation += 1
        generation = io.goal_generation

        msg = NavigateToPose.Goal()
        msg.pose = self._pose_stamped(goal.x, goal.y)

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
        status: int = future.result().status
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
                self.get_logger().info(f'[{ns}] SUCCEEDED moved={moved_str}')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning(f'[{ns}] goal ABORTED (moved={moved_str})')
        else:
            self.get_logger().warning(f'[{ns}] goal ended status={status}')
        io.current_goal = None

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
                self.get_logger().warning(f'[{io.ns}] {reason} after {elapsed:.0f}s')
                self._cancel_goal(io, reason=reason)

    def _progress_since_dispatch(self, io: _RobotIO) -> float | None:
        if io.dispatch_pos is None:
            return None
        now = self._robot_pos(io)
        if now is None:
            return None
        return math.hypot(now[0] - io.dispatch_pos[0], now[1] - io.dispatch_pos[1])

    def _finish(self, coverage: float) -> None:
        self.get_logger().info(
            f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%}'
        )
        self._coverage_done = True
        if self._episode_file:
            Path(self._episode_file).write_text(f'coverage={coverage:.1%}')
