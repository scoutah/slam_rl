import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info('MapListener gestartet, warte auf /map ...')

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        data = msg.data # -1 unknown cell, 0 freely movable cell, 100 occupied (non-movable)

        unknown = data.count(-1)
        free = data.count(0)
        occupied = data.count(100)

        self.get_logger().info(
            f'Map erhalten | '
            f'Größe: {width}x{height} | '
            f'Auflösung: {resolution:.3f} m | '
            f'Unknown: {unknown}, Free: {free}, Occupied: {occupied}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
