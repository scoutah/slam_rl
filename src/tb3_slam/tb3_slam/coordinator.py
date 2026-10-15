"""Centralised exploration coordinator (issue #11).

One node sees the merged ``/map`` and both robots' poses, asks a swappable
:class:`~tb3_slam.allocator.Allocator` for a goal per robot, and dispatches each
goal to that robot's own Nav2 (``/<ns>/navigate_to_pose``). All ROS I/O — map
subscription, TF lookups, Nav2 action clients, failure/cancel/timeout handling —
lives here, so the classical and RL methods differ *only* in the allocator
object behind the shared interface.

The allocator is selected by name (``allocator`` parameter, default ``dummy``)
via :func:`tb3_slam.allocator_registry.make_allocator`.

Per-robot goal supervision mirrors the single-robot ``frontier_explorer``:
a goal is cancelled after ``goal_timeout`` seconds, or after
``no_progress_timeout`` seconds without ``progress_step`` metres of motion; a
"SUCCEEDED" goal that moved the robot less than ``min_progress`` metres is
treated as a non-move so the allocator picks something else next cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from tb3_slam.allocator import Goal, RobotState
from tb3_slam.allocator_registry import make_allocator
from tb3_slam.coverage_monitor import (
    check_coverage,
    DEFAULT_COVERAGE_THRESHOLD,
    load_expected_cells,
)
from tb3_slam.multi_robot import frame, MERGED_MAP_FRAME, ROBOT_NAMESPACES, topic
from tf2_ros import Buffer, TransformListener
import tf_transformations


@dataclass
class _RobotIO:
    """Per-robot ROS clients and live goal-tracking state."""

    ns: str
    base_frame: str
    nav_client: ActionClient
    goal_active: bool = False
    goal_handle: ClientGoalHandle | None = None
    goal_generation: int = 0
    current_goal: Goal | None = None
    dispatch_pos: tuple[float, float] | None = None
    dispatch_time: float | None = None
    last_progress_pos: tuple[float, float] | None = None
    last_progress_time: float | None = None


class Coordinator(Node):
    """Dispatch allocator goals to each robot's Nav2 on the merged map."""

    def __init__(self) -> None:
        super().__init__('coordinator')

        self.declare_parameter('allocator', 'dummy')
        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')
        self.declare_parameter('map_frame', MERGED_MAP_FRAME)
        self.declare_parameter('decision_period', 2.0)
        self.declare_parameter('goal_timeout', 45.0)
        self.declare_parameter('no_progress_timeout', 8.0)
        self.declare_parameter('progress_step', 0.1)
        self.declare_parameter('min_progress', 0.2)

        self._allocator_name: str = self.get_parameter('allocator').value
        self._coverage_threshold: float = self.get_parameter('coverage_threshold').value
        self._episode_file: str = self.get_parameter('episode_file').value
        self._map_frame: str = self.get_parameter('map_frame').value
        self._goal_timeout: float = self.get_parameter('goal_timeout').value
        self._no_progress_timeout: float = self.get_parameter('no_progress_timeout').value
        self._progress_step: float = self.get_parameter('progress_step').value
        self._min_progress: float = self.get_parameter('min_progress').value

        self._allocator = make_allocator(self._allocator_name)
        self.get_logger().info(f"[coord] allocator = '{self._allocator_name}'")

        self._coverage_done = False
        self._latest_map: OccupancyGrid | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Per-robot Nav2 clients and goal state.
        self._robots: dict[str, _RobotIO] = {}
        for ns in ROBOT_NAMESPACES:
            self._robots[ns] = _RobotIO(
                ns=ns,
                base_frame=frame(ns, 'base_link'),
                nav_client=ActionClient(self, NavigateToPose, topic(ns, 'navigate_to_pose')),
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

        # Merged /map is latched TRANSIENT_LOCAL by the merger.
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

        self.get_logger().info('Coordinator ready – waiting for merged /map ...')

    # ------------------------------------------------------------------
    # Map / coverage
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self._latest_map = msg
        if self._coverage_done or self._expected_cells <= 0:
            return
        coverage, _known = check_coverage(msg, self._expected_cells)
        if coverage >= self._coverage_threshold:
            self._finish(coverage)

    # ------------------------------------------------------------------
    # Decision cycle: ask the allocator, dispatch per robot
    # ------------------------------------------------------------------

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
            states.append(RobotState(name=ns, x=pos[0], y=pos[1], busy=io.goal_active))

        if not states:
            self.get_logger().warning(
                '[coord] no robot poses yet – waiting for TF', throttle_duration_sec=5.0,
            )
            return

        try:
            goals = self._allocator.allocate(self._latest_map, states)
        except Exception as e:  # noqa: BLE001 - never let an allocator crash the node
            self.get_logger().error(f'[coord] allocator raised: {e}')
            return

        for ns, goal in goals.items():
            io = self._robots.get(ns)
            if io is None or goal is None:
                continue
            if io.goal_active:
                # Allocator chose to re-task a busy robot: cancel first.
                self._cancel_goal(io, reason='re-tasked by allocator')
            self._dispatch(io, goal, positions.get(ns))

    # ------------------------------------------------------------------
    # Nav2 dispatch + result handling
    # ------------------------------------------------------------------

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
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        q = tf_transformations.quaternion_from_euler(0, 0, 0.0)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        msg.pose = pose

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
        # The next decision cycle re-tasks this now-idle robot.

    def _cancel_goal(self, io: _RobotIO, reason: str) -> None:
        self.get_logger().info(f'[{io.ns}] cancelling goal – {reason}')
        io.goal_generation += 1  # invalidate pending callbacks
        try:
            if io.goal_handle is not None:
                io.goal_handle.cancel_goal_async()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f'[{io.ns}] cancel failed: {e}')
        io.goal_handle = None
        io.goal_active = False
        io.current_goal = None

    # ------------------------------------------------------------------
    # Per-robot goal supervision (timeout / no-progress)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Episode completion
    # ------------------------------------------------------------------

    def _finish(self, coverage: float) -> None:
        self.get_logger().info(
            f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%}'
        )
        self._coverage_done = True
        if self._episode_file:
            Path(self._episode_file).write_text(f'coverage={coverage:.1%}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Coordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
