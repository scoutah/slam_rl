"""ROS node that fuses each robot's map into the global /map via map_merging."""
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tb3_slam.map_merging import clear_robot_footprints, fuse_grids, merged_grid_spec
from tb3_slam.multi_robot import frame, MERGED_MAP_FRAME, ROBOT_NAMESPACES, topic
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer


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
        self._tf_buffer = Buffer()
        self._setup_tf_listeners()

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

    def _setup_tf_listeners(self) -> None:
        """Feed the TF buffer from each /<ns>/tf[_static] plus global /tf_static.

        Same scheme as the explorer base node: robot trees never reach global
        /tf (tf_relay is viz-only), the map→<ns>/map offsets are on /tf_static.
        """
        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        dyn_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(TFMessage, '/tf_static', self._tf_static_cb, static_qos)
        for ns in ROBOT_NAMESPACES:
            self.create_subscription(TFMessage, f'/{ns}/tf', self._tf_cb, dyn_qos)
            self.create_subscription(
                TFMessage, f'/{ns}/tf_static', self._tf_static_cb, static_qos)

    def _tf_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform(transform, 'map_merger')

    def _tf_static_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform_static(transform, 'map_merger')

    def _robot_positions(self) -> list[tuple[float, float]]:
        """Return current robot positions in ``map``, skipping robots without TF yet."""
        positions = []
        for ns in ROBOT_NAMESPACES:
            try:
                transform = self._tf_buffer.lookup_transform(
                    MERGED_MAP_FRAME, frame(ns, 'base_link'), rclpy.time.Time())
            except Exception:  # noqa: BLE001 - TF may not be ready at startup
                continue
            point = transform.transform.translation
            positions.append((point.x, point.y))
        return positions

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
        # Teammates see each other as walls; free each robot's own footprint.
        clear_robot_footprints(merged, self._spec, self._robot_positions())
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
