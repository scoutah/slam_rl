"""Frontier-based exploration node.

Detects frontier centroids from the map and drives the robot to each one via Nav2
until the coverage threshold is reached.
"""
from pathlib import Path

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

from tb3_slam.coverage_monitor import (
    DEFAULT_COVERAGE_THRESHOLD,
    check_coverage,
    load_expected_cells,
)
from tb3_slam.frontier_utils import centroids_from_map, grid_to_world


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
        # Centroids that failed in the current round (cleared on success)
        self._tried: set[tuple[int, int]] = set()

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning('[coverage] No world metadata – coverage monitoring disabled')
            self._expected_cells = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected cells: {expected} (threshold {self._coverage_threshold:.0%})'
            )

        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('FrontierExplorer ready – waiting for /map ...')

    # ------------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._coverage_done:
            return

        if self._expected_cells > 0:
            coverage, known = check_coverage(msg, self._expected_cells)
            self.get_logger().info(
                f'[coverage] {coverage:.1%}  ({known}/{self._expected_cells} cells)',
                throttle_duration_sec=5.0,
            )
            if coverage >= self._coverage_threshold:
                self._finish(coverage)
                return

        if self._goal_active:
            return

        centroids = centroids_from_map(msg)
        fresh = [c for c in centroids if c not in self._tried]
        self.get_logger().info(f'[frontier] {len(centroids)} centroid(s), {len(fresh)} fresh')

        if fresh:
            self._centroid_queue = fresh
            self._current_map = msg
            self._try_next_goal()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _try_next_goal(self) -> None:
        """Pop the next centroid and send it to Nav2."""
        if not self._centroid_queue or self._current_map is None:
            self.get_logger().warning('[goal] Queue empty – waiting for new frontiers')
            self._goal_active = False
            return

        target = self._centroid_queue.pop(0)
        self._current_target = target
        x, y = grid_to_world(target, self._current_map)
        self.get_logger().info(
            f'[goal] → ({x:.2f}, {y:.2f})  ({len(self._centroid_queue)} remaining)'
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
            self._mark_tried_and_continue()
            return
        self.get_logger().info('[goal] ACCEPTED – navigating ...')
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future: Future) -> None:
        status: int = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('[goal] SUCCEEDED')
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
