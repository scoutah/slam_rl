from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node


class CmdVelRelay(Node):

    def __init__(self) -> None:
        super().__init__('cmd_vel_relay')

        self.declare_parameter('input_topic', '/cmd_vel_smoothed')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('frame_id', 'base_link')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self._publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self._subscription = self.create_subscription(
            Twist,
            input_topic,
            self._on_cmd_vel,
            10
        )

        self.get_logger().info(
            f'Relaying Twist from {input_topic} to TwistStamped on {output_topic}'
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self._frame_id
        stamped.twist = msg
        self._publisher.publish(stamped)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
