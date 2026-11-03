"""Mirror per-robot ``/<ns>/tf`` onto the global ``/tf`` so RViz can display it.

Robots publish transforms only on ``/<ns>/tf`` (Nav2 multi-robot convention), so
the global ``/tf`` stays empty and RViz displays that only subscribe to ``/tf``
(e.g. RobotModel) show nothing. Frames are prefixed, so merging is collision-free.
Visualisation aid only: launched with RViz, never fed back into a robot stack.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from tf2_msgs.msg import TFMessage


class TfRelay(Node):
    def __init__(self) -> None:
        super().__init__('tf_relay')
        namespaces = self.declare_parameter(
            'namespaces', ['r1', 'r2']).get_parameter_value().string_array_value

        self._dyn_pub = self.create_publisher(TFMessage, '/tf', 100)
        # tf_static is latched: keep every sample so late joiners (RViz) get it.
        static_qos = QoSProfile(
            depth=100,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._static_pub = self.create_publisher(TFMessage, '/tf_static', static_qos)

        for ns in namespaces:
            self.create_subscription(
                TFMessage, f'/{ns}/tf',
                lambda msg: self._dyn_pub.publish(msg), 100)
            self.create_subscription(
                TFMessage, f'/{ns}/tf_static',
                lambda msg: self._static_pub.publish(msg), static_qos)

        self.get_logger().info(f'TF relay running for namespaces: {list(namespaces)}')


def main() -> None:
    rclpy.init()
    node = TfRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
