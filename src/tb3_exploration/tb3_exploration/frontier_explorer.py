from pathlib import Path

import tf_transformations
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from tb3_exploration.frontier_utils import *
from tb3_slam.coverage_monitor import (
    check_coverage,
    load_expected_cells,
    DEFAULT_COVERAGE_THRESHOLD,
)


class FrontierExplorer(Node):

    def __init__(self) -> None:
        """
        Initialize the FrontierExplorer node. This node subscribes to the /map topic to receive
        occupancy grid updates, processes the map to identify frontiers, and sends navigation
        goals to Nav2 to explore those frontiers.
        """
        super().__init__('frontier_explorer')

        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')

        self._coverage_threshold: float = (
            self.get_parameter('coverage_threshold').value
        )
        self._episode_file: str = self.get_parameter('episode_file').value

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning(
                '[coverage] World metadata file not found – '
                'coverage monitoring disabled until a new world is generated'
            )
            self._expected_cells: int = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected traversable cells: {self._expected_cells} '
                f'(threshold {self._coverage_threshold:.0%})'
            )

        self._coverage_done: bool = False
        self.goal_active: bool = False
        self._centroid_queue: list[tuple[int, int]] = []
        self._current_map: OccupancyGrid | None = None
        # Centroids that were tried and failed in the current round.
        # Only cleared when a goal succeeds. map_callback skips any centroid
        # already in this set, so the full queue is exhausted before the same
        # positions are ever retried.
        self._tried_centroids: set[tuple[int, int]] = set()
        self._current_target: tuple[int, int] | None = None

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10,
        )

        self.nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('FrontierExplorer gestartet, warte auf /map ...')

    def map_callback(self, msg: OccupancyGrid) -> None:
        """
        Triggered whenever a new map is received on /map. Checks coverage progress,
        then processes frontiers and sends navigation goals to Nav2.
        """
        if self._coverage_done:
            return

        # --- Coverage check (always, regardless of goal state) ---
        if self._expected_cells > 0:
            coverage, known = check_coverage(msg, self._expected_cells)
            self.get_logger().info(
                f'[coverage] {coverage:.1%}  ({known}/{self._expected_cells} cells)',
                throttle_duration_sec=5.0,
            )
            if coverage >= self._coverage_threshold:
                self.get_logger().info(
                    f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%} – stopping exploration'
                )
                self._coverage_done = True
                if self._episode_file:
                    Path(self._episode_file).write_text(f'coverage={coverage:.1%}')
                return

        if self.goal_active:
            self.get_logger().debug('Goal already active, skipping map update')
            return

        grid = np.array(msg.data, dtype=int).reshape((msg.info.height, msg.info.width))
        self.get_logger().info(f'map size: {grid.shape}')

        # '-1' = unknown cell, cell value = probability of being occupied (0–100)
        unknown = int(np.sum(grid == -1))
        free = int(np.sum((FREE_LOWER_BOUND <= grid) & (grid <= FREE_UPPER_BOUND)))
        uncertain = int(np.sum((FRONTIER_LOWER_BOUND <= grid) & (grid <= FRONTIER_UPPER_BOUND)))
        occupied = int(np.sum((OCCUPIED_LOWER_BOUND <= grid) & (grid <= OCCUPIED_UPPER_BOUND)))
        self.get_logger().info(
            f'unknown: {unknown}, free: {free}, uncertain: {uncertain}, occupied: {occupied}'
        )

        frontier_cells = find_frontiers(grid)
        frontier_groups = find_groups(frontier_cells)
        self.get_logger().info(f'frontier_groups: {len(frontier_groups)}')

        centroids: list[tuple[int, int]] = []
        for frontier in frontier_groups:
            centroid = find_centroid(frontier, grid)
            if centroid:
                centroids.append(centroid)
        self.get_logger().info(f'centroids: {len(centroids)}')

        if centroids:
            fresh = [c for c in centroids if c not in self._tried_centroids]
            if not fresh:
                self.get_logger().debug(
                    f'[frontier] All {len(centroids)} centroid(s) already tried – '
                    'waiting for new frontiers'
                )
                return
            self._centroid_queue = fresh
            self._current_map = msg
            self._try_next_goal()

    def _try_next_goal(self) -> None:
        """Pop the next centroid from the queue and send it as a Nav2 goal."""
        if not self._centroid_queue or self._current_map is None:
            self.get_logger().warning('[goal] No more frontiers in queue – waiting for new map update')
            self.goal_active = False
            return

        target_cell = self._centroid_queue.pop(0)
        self._current_target = target_cell
        remaining = len(self._centroid_queue)
        self.get_logger().info(
            f'[goal] Trying centroid (grid): row={target_cell[0]}, col={target_cell[1]}'
            f' ({remaining} remaining in queue)'
        )
        x, y = self.grid_to_world(target_cell, self._current_map)
        self.send_goal(x, y)

    def send_goal(self, x: float, y: float, theta: float = 0.0) -> None:
        """Send a navigation goal to Nav2."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Nav2 NavigateToPose action server not reachable after 5 s – goal NOT sent'
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y

        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]

        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'[goal] Sending to Nav2: x={x:.3f}, y={y:.3f}, theta={theta:.3f}'
        )
        self.goal_active = True
        future: Future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        """Called when Nav2 accepts or rejects the navigation goal."""
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(
                f'[goal] REJECTED by Nav2 – {len(self._centroid_queue)} fallback(s) left'
            )
            if self._current_target is not None:
                self._tried_centroids.add(self._current_target)
                self._current_target = None
            self._try_next_goal()
            return

        self.get_logger().info('[goal] ACCEPTED by Nav2 – waiting for result ...')
        result_future: Future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future: Future) -> None:
        """Called when Nav2 finishes (success, abort, or cancel)."""
        result = future.result()
        status: int = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('[goal] SUCCEEDED – robot reached the target')
            self._tried_centroids.clear()
            self._current_target = None
            self.goal_active = False
            self._centroid_queue.clear()
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning(
                f'[goal] ABORTED by Nav2 (path blocked or planning failed)'
                f' – {len(self._centroid_queue)} fallback(s) left'
            )
            if self._current_target is not None:
                self._tried_centroids.add(self._current_target)
                self._current_target = None
            self._try_next_goal()
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning('[goal] CANCELLED')
            self._current_target = None
            self.goal_active = False
            self._centroid_queue.clear()
        else:
            self.get_logger().error(f'[goal] Unexpected status code: {status}')
            self._current_target = None
            self.goal_active = False
            self._centroid_queue.clear()

    def grid_to_world(self, centroid: tuple[int, int], msg: OccupancyGrid) -> tuple[float, float]:
        """Convert grid cell (row, col) to world (x, y) using the map metadata."""
        x = msg.info.origin.position.x + (centroid[1] + 0.5) * msg.info.resolution
        y = msg.info.origin.position.y + (centroid[0] + 0.5) * msg.info.resolution
        return x, y



def main(args=None):
    """Main function to initialize the FrontierExplorer node and start spinning."""
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
