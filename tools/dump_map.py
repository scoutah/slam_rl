#!/usr/bin/env python3
"""Save the merged /map and both robots' cells to .npz snapshots while a sim runs.

Needs a sourced ROS 2 environment and the system Python (``/usr/bin/python3``).
Replay a snapshot offline with ``tools/analyze_map_dump.py``.
"""
import argparse
from pathlib import Path

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

ROBOTS = ('r1', 'r2')
_TF_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
_STATIC_QOS = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
_MAP_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class MapDumper(Node):
    """Write ``map_NNN.npz`` (grid, res, origin, robot cells) every ``period`` s."""

    def __init__(self, out_dir: Path, period: float) -> None:
        """Subscribe to the merged map and all TF trees; start the save timer."""
        super().__init__('map_dumper')
        self._out = out_dir
        self._out.mkdir(parents=True, exist_ok=True)
        self._buf = Buffer()
        self._map: OccupancyGrid | None = None
        self._n = 0
        self.create_subscription(TFMessage, '/tf_static', self._static_cb, _STATIC_QOS)
        for ns in ROBOTS:
            self.create_subscription(TFMessage, f'/{ns}/tf', self._tf_cb, _TF_QOS)
            self.create_subscription(
                TFMessage, f'/{ns}/tf_static', self._static_cb, _STATIC_QOS)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, _MAP_QOS)
        self.create_timer(period, self._save)

    def _tf_cb(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self._buf.set_transform(t, 'dump')

    def _static_cb(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self._buf.set_transform_static(t, 'dump')

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _save(self) -> None:
        msg = self._map
        if msg is None:
            return
        info = msg.info
        cells: dict[str, tuple[int, int]] = {}
        for ns in ROBOTS:
            try:
                t = self._buf.lookup_transform(
                    'map', f'{ns}/base_link', rclpy.time.Time())
            except Exception:  # noqa: BLE001 - pose not available yet
                continue
            x, y = t.transform.translation.x, t.transform.translation.y
            cells[ns] = (int((y - info.origin.position.y) / info.resolution),
                         int((x - info.origin.position.x) / info.resolution))
        grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
        path = self._out / f'map_{self._n:03d}.npz'
        np.savez(
            path, grid=grid, res=info.resolution,
            ox=info.origin.position.x, oy=info.origin.position.y,
            names=np.array(sorted(cells)),
            robots=np.array([cells[k] for k in sorted(cells)]).reshape(-1, 2),
        )
        self.get_logger().info(f'saved {path} robots={cells}')
        self._n += 1


def main() -> None:
    """Parse arguments and spin the dumper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/tmp/slam_map_dump'),
                        help='output directory (default: /tmp/slam_map_dump)')
    parser.add_argument('--period', type=float, default=10.0,
                        help='seconds between snapshots (default: 10)')
    args = parser.parse_args()
    rclpy.init()
    try:
        rclpy.spin(MapDumper(args.out, args.period))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
