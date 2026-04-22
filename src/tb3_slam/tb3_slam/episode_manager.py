import time
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Float32

from tb3_slam.coverage_monitor import check_coverage, load_expected_cells


class EpisodeManager(Node):
    """
    Monitors map coverage and signals episode completion.

    Subscribes to /map, computes the fraction of known cells, and writes a
    done-signal file once coverage exceeds the configured threshold (default
    95 %).  An outer process (training_runner.py) watches that file and
    restarts the simulation for the next episode.

    Published topics:
      /episode/coverage  (std_msgs/Float32)  – current coverage [0, 1]
    """

    def __init__(self) -> None:
        super().__init__('episode_manager')

        self.declare_parameter('coverage_threshold', 0.99)
        self.declare_parameter('episode_file', '/tmp/slam_episode_done')
        self.declare_parameter('max_episode_duration', 600.0)  # seconds
        # Ignore /map messages for this many seconds after launch. A previous
        # episode's cartographer that survives SIGINT keeps publishing the old
        # (nearly complete) map on /map with TRANSIENT_LOCAL QoS, which would
        # make the new episode terminate instantly at ~99% coverage.
        self.declare_parameter('startup_grace_s', 15.0)

        self.threshold: float = self.get_parameter('coverage_threshold').value
        self.episode_file: str = self.get_parameter('episode_file').value
        self.max_duration: float = self.get_parameter('max_episode_duration').value
        self.startup_grace_s: float = self.get_parameter('startup_grace_s').value

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning(
                'World metadata file not found – coverage denominator will fall back '
                'to total map size on first map message'
            )
            self._expected_cells: int = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(f'Expected traversable cells: {self._expected_cells}')

        self._start_time: float = time.time()
        self._done: bool = False

        self._coverage_pub = self.create_publisher(Float32, '/episode/coverage', 10)

        self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)

        # Watchdog: check wall-clock timeout every 10 s
        self.create_timer(10.0, self._timeout_check)

        self.get_logger().info(
            f'EpisodeManager started – threshold={self.threshold:.0%}, '
            f'max_duration={self.max_duration:.0f}s, '
            f'episode_file={self.episode_file}'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._done:
            return

        elapsed = time.time() - self._start_time
        if elapsed < self.startup_grace_s:
            self.get_logger().info(
                f'[warmup] ignoring /map ({self.startup_grace_s - elapsed:.0f}s left)',
                throttle_duration_sec=5.0,
            )
            return

        # Fall back to total map size if no world metadata was available
        expected = self._expected_cells if self._expected_cells > 0 else len(msg.data)
        coverage, known = check_coverage(msg, expected)

        self._coverage_pub.publish(Float32(data=float(coverage)))
        self.get_logger().info(
            f'Coverage: {coverage:.1%}  ({known}/{expected} cells)',
            throttle_duration_sec=5.0,
        )

        if coverage >= self.threshold:
            elapsed = time.time() - self._start_time
            self._signal_done(reason=f'coverage={coverage:.1%}', elapsed=elapsed)

    def _timeout_check(self) -> None:
        if self._done:
            return
        elapsed = time.time() - self._start_time
        if elapsed >= self.max_duration:
            self._signal_done(reason='timeout', elapsed=elapsed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _signal_done(self, *, reason: str, elapsed: float) -> None:
        self._done = True
        self.get_logger().info(
            f'Episode done – reason={reason}, elapsed={elapsed:.1f}s'
        )
        Path(self.episode_file).write_text(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
