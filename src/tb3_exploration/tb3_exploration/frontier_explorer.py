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
        super().__init__('frontier_explorer')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('MapListener gestartet, warte auf /map ...')


    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # -1 unknown cell, 0 freely movable cell, 100 occupied (non-movable)
        grid = np.array(msg.data, dtype=int).reshape((height, width))
        self.get_logger().info(f'map size: {grid.shape}')

        unknown = np.sum(grid == -1)
        free = np.sum((FREE_LOWER_BOUND <= grid) & (grid <= FREE_UPPER_BOUND))
        uncertain = np.sum((FRONTIER_LOWER_BOUND <= grid) & (grid <= FRONTIER_UPPER_BOUND))
        occupied = np.sum((OCCUPIED_LOWER_BOUND <= grid) & (grid <= OCCUPIED_UPPER_BOUND))

        self.get_logger().info(f'unknown: {unknown}, free: {free}, uncertain: {uncertain}, occupied: {occupied}')

        # currently only finds hard cuts between -1 and 0, but most cells are ranged between 0 and 100
        frontier_cells = find_frontiers(grid)
        frontiers = find_groups(frontier_cells)
        self.get_logger().info(f'frontiers: {len(frontiers)}')


        centroids: list[tuple[int, int]] = []
        for frontier in frontiers:
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
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            self.goal_active = False
            return

        self.get_logger().info('Goal accepted by Nav2')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)


    def goal_result_callback(self, future):
        self.get_logger().info('Goal reached')
        self.goal_active = False


    def grid_to_world(self, centroid: tuple[int, int], msg: OccupancyGrid):
        x = msg.info.origin.position.x + (centroid[1] + 0.5) * msg.info.resolution
        y = msg.info.origin.position.y + (centroid[0] + 0.5) * msg.info.resolution
        return x, y



def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
