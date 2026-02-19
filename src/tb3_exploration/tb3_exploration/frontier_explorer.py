from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

from tb3_exploration.frontier_utils import *


class FrontierExplorer(Node):

    def __init__(self):
        """
        Initialize the FrontierExplorer node. This node subscribes to the /map topic to receive occupancy grid updates, processes the map to identify frontiers, and sends navigation goals to Nav2 to explore those frontiers.
        """
        super().__init__('frontier_explorer')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('FrontierExplorer gestartet, warte auf /map ...')


    def map_callback(self, msg: OccupancyGrid):
        """
        This callback is triggered whenever a new map is received on the /map topic. It processes the occupancy grid to identify frontiers and sends a navigation goal to Nav2 for the first frontier centroid found.
        """

        grid = np.array(msg.data, dtype=int).reshape((msg.info.height, msg.info.width))
        self.get_logger().info(f'map size: {grid.shape}')

        # '-1' = unknown cell, cell number is the probability of being occupied (0-100)
        unknown = np.sum(grid == -1)
        free = np.sum((FREE_LOWER_BOUND <= grid) & (grid <= FREE_UPPER_BOUND))
        uncertain = np.sum((FRONTIER_LOWER_BOUND <= grid) & (grid <= FRONTIER_UPPER_BOUND))
        occupied = np.sum((OCCUPIED_LOWER_BOUND <= grid) & (grid <= OCCUPIED_UPPER_BOUND))
        self.get_logger().info(f'unknown: {unknown}, free: {free}, uncertain: {uncertain}, occupied: {occupied}')

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
            self.get_logger().info(f'Centroid[0]: {centroids[0]}')
            target_cell = centroids[0]  # currently a random centroid (not the nearest or biggest frontier)
            x, y = self.grid_to_world(target_cell, msg)
            self.send_goal(x, y)


    def send_goal(self, x, y, theta=0.0):
        """
        Send a navigation goal to Nav2. Nav2 will attempt to navigate the robot to the specified (x, y) coordinates with the given orientation (theta).
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose Action Server not reachable')
            return

        goal_msg = NavigateToPose.Goal()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y

        # Quaternion from yaw
        import tf_transformations
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]

        goal_msg.pose = goal_pose

        self.get_logger().info(f'Send goal to Nav2: x={x:.2f}, y={y:.2f}, theta={theta:.2f}')
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        This callback is triggered when Nav2 responds to the navigation goal. It checks if the goal was accepted and, if so, waits for the result (i.e., whether the robot reached the goal).
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            self.goal_active = False
            return

        self.get_logger().info('Goal accepted by Nav2')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)


    def goal_result_callback(self, future):
        """
        This callback is triggered when Nav2 finishes processing the navigation goal.
        It logs whether the goal was successfully reached or not.
        """
        self.get_logger().info('Goal reached')
        self.goal_active = False


    def grid_to_world(self, centroid: tuple[int, int], msg: OccupancyGrid):
        """Convert grid cell coordinates to world coordinates using the map metadata."""
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
