"""Frontier-based exploration node.

Detects frontier centroids from the map and drives the robot to each one via Nav2
until the coverage threshold is reached.
"""
import math
from pathlib import Path

import rclpy
import tf_transformations
from tf2_ros import Buffer, TransformListener
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import ComputePathToPose, NavigateToPose
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


class FrontierExplorer(Node):
    """Navigate to frontier centroids until the map is sufficiently explored."""

    def __init__(self) -> None:
        super().__init__('frontier_explorer')

        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')

        self._coverage_threshold: float = self.get_parameter('coverage_threshold').value
        self._episode_file: str = self.get_parameter('episode_file').value
        self._coverage_done = False

        # Navigation state
        self._goal_active = False
        self._current_map: OccupancyGrid | None = None
        self._centroid_queue: list[tuple[int, int]] = []
        self._current_target: tuple[int, int] | None = None
        # Centroids that failed in the current round (cleared on success that
        # actually moved the robot — see _on_goal_result)
        self._tried: set[tuple[int, int]] = set()
        # Robot world position captured at goal dispatch, used to detect
        # "Nav2 succeeded but robot didn't move" scenarios (TF/tolerance mismatch)
        self._goal_dispatch_pos: tuple[float, float] | None = None
        self._goal_dispatch_world: tuple[float, float] | None = None
        self._goal_dispatch_time: float | None = None
        self._goal_handle: ClientGoalHandle | None = None
        # Meters — if a "SUCCEEDED" goal moved the robot less than this, the
        # goal was likely reached by coincidence/tolerance rather than real
        # navigation; mark it tried to break repeat cycles.
        self._MIN_PROGRESS_M = 0.2
        # Seconds — cancel a goal that Nav2 has not completed in this time.
        # Covers infinite recovery loops (spin → wait → backup) that never
        # return SUCCEEDED/ABORTED and would otherwise silence the explorer.
        self._GOAL_TIMEOUT_S = 45.0

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning('[coverage] No world metadata – coverage monitoring disabled')
            self._expected_cells = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected cells: {expected} (threshold {self._coverage_threshold:.0%})'
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._planner_client: ActionClient = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose'
        )
        self._viz_pub = self.create_publisher(MarkerArray, '/frontiers_viz', 10)

        # Ranking state. We ask Nav2 for a full plan to every candidate centroid
        # so we can rank by real path length (respects walls) instead of
        # Euclidean distance. Ranking is async: one ComputePathToPose goal per
        # candidate; when all results are in we sort and dispatch the nearest.
        self._ranking_in_progress = False
        self._ranking_map: OccupancyGrid | None = None
        self._ranking_pending: int = 0
        self._ranking_results: list[tuple[tuple[int, int], float]] = []

        self.get_logger().info('FrontierExplorer ready – waiting for /map ...')

    # ------------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._coverage_done:
            return

        if self._expected_cells > 0:
            coverage, known = check_coverage(msg, self._expected_cells)
            if coverage >= self._coverage_threshold:
                self._finish(coverage)
                return

        # Run detection + publish markers on EVERY map update so RViz shows
        # frontier cells and the active goal while Nav2 is navigating. If we
        # gated this behind `_goal_active`, the MarkerArray would stop
        # updating the moment the first goal is dispatched.
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids, stats = centroids_from_map_debug(msg, robot_grid_pos=robot_grid_pos)
        self._viz_pub.publish(
            build_frontier_markers(
                stats['frontiers'],
                self._current_target,
                msg,
                stamp=self.get_clock().now().to_msg(),
                centroids=centroids,
                goal_color=(0.0, 1.0, 0.0),
            )
        )

        if self._goal_active:
            self._supervise_active_goal()
            return

        if self._ranking_in_progress:
            return

        fresh = [c for c in centroids if c not in self._tried]
        self.get_logger().info(
            f'[frontier] {len(centroids)} centroid(s), {len(fresh)} fresh',
            throttle_duration_sec=5.0,
        )

        # Detailed diagnostics when the pipeline produces nothing navigable —
        # tells us which stage swallowed the frontiers (detection, grouping,
        # centroid validation) or whether the robot pose is broken.
        if not fresh:
            self._log_pipeline_stats(stats)

        if fresh:
            self._start_ranking(fresh, msg)

    def _log_pipeline_stats(self, stats: dict) -> None:
        log = self.get_logger()
        log.warning(
            '[diag] pipeline produced no fresh centroids — '
            f'tried={len(self._tried)} '
            f'robot_grid={stats["robot_grid_pos"]} '
            f'robot_cell_val={stats["robot_cell_value"]} '
            f'grid={stats["grid_shape"]} '
            f'cells free={stats["n_free"]} unc={stats["n_uncertain"]} '
            f'wall={stats["n_wall"]} unk={stats["n_unknown"]}',
            throttle_duration_sec=5.0,
        )
        log.warning(
            f'[diag] pipeline frontiers={stats["n_frontiers"]} '
            f'groups={stats["n_groups"]} centroids={stats["n_centroids"]} '
            f'dropped_groups={stats["n_dropped_groups"]} '
            f'group_sizes={stats["group_sizes"][:10]}',
            throttle_duration_sec=5.0,
        )
        for i, d in enumerate(stats['dropped_group_sample']):
            log.warning(
                f'[diag] dropped#{i} size={d["size"]} '
                f'centroid_rc={d["centroid_rc"]} centroid_val={d["centroid_value"]} '
                f'samples={d["samples"]}',
                throttle_duration_sec=5.0,
            )

    def _get_robot_pos(self) -> tuple[float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            self.get_logger().warning(f'[tf] Could not get robot position: {e}')
            return None

    # ------------------------------------------------------------------
    # Nav2-based centroid ranking
    # ------------------------------------------------------------------

    def _start_ranking(
        self,
        fresh: list[tuple[int, int]],
        map_msg: OccupancyGrid,
    ) -> None:
        """Ask Nav2 for a real plan to every fresh centroid, then dispatch nearest.

        If the planner action server isn't up yet we fall back to Euclidean
        sorting so the robot still moves while Nav2 is initialising.
        """
        if not self._planner_client.server_is_ready():
            robot = self._get_robot_pos()
            if robot is not None:
                fresh.sort(
                    key=lambda c: math.hypot(
                        grid_to_world(c, map_msg)[0] - robot[0],
                        grid_to_world(c, map_msg)[1] - robot[1],
                    )
                )
            self.get_logger().warning(
                '[rank] compute_path_to_pose not ready — using Euclidean fallback',
                throttle_duration_sec=5.0,
            )
            self._centroid_queue = fresh
            self._current_map = map_msg
            self._try_next_goal()
            return

        self._ranking_in_progress = True
        self._ranking_map = map_msg
        self._ranking_results = []
        self._ranking_pending = len(fresh)

        for cell in fresh:
            wx, wy = grid_to_world(cell, map_msg)
            goal = ComputePathToPose.Goal()
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            goal.goal = pose
            goal.use_start = False  # let Nav2 use the current robot pose
            future = self._planner_client.send_goal_async(goal)
            future.add_done_callback(
                lambda f, c=cell: self._on_rank_goal_response(f, c)
            )

    def _on_rank_goal_response(self, future: Future, cell: tuple[int, int]) -> None:
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            self._record_rank_result(cell, None)
            return
        handle.get_result_async().add_done_callback(
            lambda f, c=cell: self._on_rank_result(f, c)
        )

    def _on_rank_result(self, future: Future, cell: tuple[int, int]) -> None:
        wrapper = future.result()
        status: int = wrapper.status
        length: float | None = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            path = wrapper.result.path
            if wrapper.result.error_code == 0 and len(path.poses) >= 2:
                length = 0.0
                prev = path.poses[0].pose.position
                for ps in path.poses[1:]:
                    cur = ps.pose.position
                    length += math.hypot(cur.x - prev.x, cur.y - prev.y)
                    prev = cur
        self._record_rank_result(cell, length)

    def _record_rank_result(
        self,
        cell: tuple[int, int],
        length: float | None,
    ) -> None:
        if length is not None:
            self._ranking_results.append((cell, length))
        self._ranking_pending -= 1
        if self._ranking_pending > 0:
            return
        self._finish_ranking()

    def _finish_ranking(self) -> None:
        reachable = sorted(self._ranking_results, key=lambda t: t[1])
        ranked = [cell for cell, _ in reachable]
        if reachable:
            self.get_logger().info(
                f'[rank] {len(ranked)} reachable centroids '
                f'(nearest path_len={reachable[0][1]:.2f}m)'
            )
        else:
            self.get_logger().warning(
                '[rank] No centroid is reachable — waiting for next map update'
            )
        self._ranking_in_progress = False
        if not ranked:
            self._ranking_map = None
            return
        self._centroid_queue = ranked
        self._current_map = self._ranking_map
        self._ranking_map = None
        self._try_next_goal()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _try_next_goal(self) -> None:
        """Pop the next centroid and send it to Nav2."""
        if not self._centroid_queue or self._current_map is None:
            self.get_logger().warning(
                '[goal] Queue empty – waiting for new frontiers',
                throttle_duration_sec=5.0,
            )
            self._goal_active = False
            return

        target = self._centroid_queue.pop(0)
        self._current_target = target
        x, y = grid_to_world(target, self._current_map)
        robot = self._get_robot_pos()
        dist_str = 'n/a'
        if robot is not None:
            d = math.hypot(x - robot[0], y - robot[1])
            dist_str = f'{d:.2f}m'
        self._goal_dispatch_pos = robot
        self._goal_dispatch_world = (x, y)
        self._goal_dispatch_time = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            f'[goal] → ({x:.2f}, {y:.2f})  robot=({robot[0]:.2f}, {robot[1]:.2f})  '
            f'dist={dist_str}  ({len(self._centroid_queue)} remaining)'
            if robot is not None else
            f'[goal] → ({x:.2f}, {y:.2f})  robot=unknown  ({len(self._centroid_queue)} remaining)'
        )
        self._send_goal(x, y)

    def _send_goal(self, x: float, y: float, theta: float = 0.0) -> None:
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('[goal] Nav2 action server not reachable')
            return

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        goal.pose = pose

        self._goal_active = True
        future: Future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future: Future) -> None:
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            self.get_logger().warning('[goal] REJECTED – trying next centroid')
            self._goal_handle = None
            self._mark_tried_and_continue()
            return
        self._goal_handle = handle
        self.get_logger().info('[goal] ACCEPTED – navigating ...')
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future: Future) -> None:
        status: int = future.result().status
        self._goal_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            moved = self._progress_since_dispatch()
            moved_str = f'{moved:.2f}m' if moved is not None else 'n/a'
            if moved is not None and moved < self._MIN_PROGRESS_M:
                # Nav2 says success but the robot barely moved — happens when
                # Nav2's internal pose is already inside its goal_tolerance of
                # the target even though our TF said otherwise. Mark this
                # target as tried so we pick a different frontier next round
                # instead of looping on the same coordinates.
                self.get_logger().warning(
                    f'[goal] SUCCEEDED but robot moved only {moved_str} '
                    f'(< {self._MIN_PROGRESS_M}m) — marking target tried'
                )
                if self._current_target is not None:
                    self._tried.add(self._current_target)
                self._current_target = None
                self._goal_active = False
                self._centroid_queue.clear()
            else:
                self.get_logger().info(f'[goal] SUCCEEDED  moved={moved_str}')
                self._tried.clear()
                self._current_target = None
                self._goal_active = False
                self._centroid_queue.clear()

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning(
                f'[goal] ABORTED – {len(self._centroid_queue)} fallback(s) left'
            )
            self._mark_tried_and_continue()

        else:
            self.get_logger().warning(f'[goal] Status {status} – clearing queue')
            self._current_target = None
            self._goal_active = False
            self._centroid_queue.clear()

    def _mark_tried_and_continue(self) -> None:
        if self._current_target is not None:
            self._tried.add(self._current_target)
            self._current_target = None
        self._try_next_goal()

    def _supervise_active_goal(self) -> None:
        """Periodic watchdog for goals stuck in Nav2 recovery loops.

        Nav2's recovery subtree (spin → wait → backup) can iterate for a long
        time without ever returning SUCCEEDED or ABORTED. Without a timeout,
        ``_goal_active`` stays True forever and the explorer goes silent —
        the user sees no frontier/goal logs and the episode runs down its
        clock. We log progress periodically and force-cancel after a limit.
        """
        if self._goal_dispatch_time is None:
            return
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._goal_dispatch_time
        moved = self._progress_since_dispatch()
        moved_str = f'{moved:.2f}m' if moved is not None else 'n/a'
        target_str = (
            f'{self._goal_dispatch_world[0]:.2f}, {self._goal_dispatch_world[1]:.2f}'
            if self._goal_dispatch_world is not None else 'n/a'
        )
        self.get_logger().info(
            f'[goal] active {elapsed:.0f}s target=({target_str}) moved={moved_str}',
            throttle_duration_sec=5.0,
        )
        if elapsed > self._GOAL_TIMEOUT_S and self._goal_handle is not None:
            self.get_logger().warning(
                f'[goal] TIMEOUT after {elapsed:.0f}s — cancelling and trying next'
            )
            try:
                self._goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warning(f'[goal] cancel failed: {e}')
            # Defensive: mark tried locally so we skip this centroid even if
            # the cancel callback is slow to fire.
            if self._current_target is not None:
                self._tried.add(self._current_target)
            self._current_target = None
            self._goal_handle = None
            self._goal_active = False
            self._centroid_queue.clear()

    def _progress_since_dispatch(self) -> float | None:
        """Return distance (m) the robot actually moved since the goal was sent."""
        if self._goal_dispatch_pos is None:
            return None
        now = self._get_robot_pos()
        if now is None:
            return None
        dx = now[0] - self._goal_dispatch_pos[0]
        dy = now[1] - self._goal_dispatch_pos[1]
        return math.hypot(dx, dy)

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


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
