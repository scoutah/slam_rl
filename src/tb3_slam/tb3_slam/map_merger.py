from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from tb3_slam.map_merging import fuse_grids, merged_grid_spec
from tb3_slam.multi_robot import MERGED_MAP_FRAME, ROBOT_NAMESPACES, topic


class MapMerger(Node):
    """Merge every robot's Cartographer map into one global ``/map``.

    Parameters (ROS):
      spawns      flat [x1, y1, x2, y2, ...] spawn poses (metres, ``map`` frame),
                  one per robot in ROBOT_NAMESPACES order; ``<ns>/map`` starts there.
      world_size  edge length of the square world (m); fixes the merged extent.
      rate        publish rate in Hz.

    Publishes ``/map`` (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL) in frame ``map``.
    """

    def __init__(self) -> None:
        super().__init__('map_merger')
        self.declare_parameter('spawns', [0.0] * (2 * len(ROBOT_NAMESPACES)))
        self.declare_parameter('world_size', 8.0)
        self.declare_parameter('rate', 2.0)

        flat = list(self.get_parameter('spawns').value)
        if len(flat) != 2 * len(ROBOT_NAMESPACES):
            raise ValueError(f'spawns needs {2 * len(ROBOT_NAMESPACES)} values, got {len(flat)}')
        self._spawns = {
            ns: (flat[2 * i], flat[2 * i + 1]) for i, ns in enumerate(ROBOT_NAMESPACES)
        }
        world_size = float(self.get_parameter('world_size').value)
        self._spec = merged_grid_spec(world_size)
        self._maps: dict[str, OccupancyGrid] = {}
        # Persistent merged state: the fusion is stateful so a cell confirmed FREE
        # once is not lost when Cartographer later reports it UNKNOWN/UNCERTAIN.
        self._merged: np.ndarray | None = None

        # Cartographer's occupancy grid node latches its map (TRANSIENT_LOCAL).
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        for ns in ROBOT_NAMESPACES:
            self.create_subscription(
                OccupancyGrid, topic(ns, 'map'),
                lambda msg, ns=ns: self._maps.__setitem__(ns, msg), qos,
            )
        self._pub = self.create_publisher(OccupancyGrid, '/map', qos)
        self.create_timer(1.0 / float(self.get_parameter('rate').value), self._publish)

        self.get_logger().info(
            f'Merging {[topic(ns, "map") for ns in ROBOT_NAMESPACES]} into /map '
            f'({self._spec.width}x{self._spec.height} cells, world_size={world_size:g} m)'
        )

    def _publish(self) -> None:
        if not self._maps:
            return
        layers = []
        for ns, msg in self._maps.items():
            info = msg.info
            if info.width == 0 or info.height == 0:
                continue
            data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
            spawn_x, spawn_y = self._spawns[ns]
            layers.append((
                data,
                info.origin.position.x + spawn_x,
                info.origin.position.y + spawn_y,
                info.resolution,
            ))
        if not layers:
            return
        try:
            merged = fuse_grids(self._spec, layers, base=self._merged)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
            return
        self._merged = merged

        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = MERGED_MAP_FRAME
        out.info.map_load_time = out.header.stamp
        out.info.resolution = self._spec.resolution
        out.info.width = self._spec.width
        out.info.height = self._spec.height
        out.info.origin.position.x = self._spec.origin_x
        out.info.origin.position.y = self._spec.origin_y
        out.info.origin.orientation.w = 1.0
        out.data = merged.ravel().tolist()
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
