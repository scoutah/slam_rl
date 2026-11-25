"""Watch the merged map and signal episode end on coverage, stuck or timeout."""
import os
from pathlib import Path

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from tb3_slam.coverage_monitor import (
    check_coverage,
    load_episode_timeout,
    load_expected_cells,
)


SIM_ELAPSED_SUFFIX = '.sim_s'


class EpisodeManager(Node):
    """
    Monitors map coverage and signals episode completion.

    Subscribes to /map, computes the fraction of known cells, and writes a
    done-signal file once coverage exceeds the configured threshold (default
    97 %).  An outer process (training_runner.py) watches that file and
    restarts the simulation for the next episode.

    All durations (grace, stuck window, timeout, elapsed) run on the node
    clock, i.e. *simulation* time under ``use_sim_time``, so they mean the same
    thing whatever real-time factor Gazebo achieves.  The current sim-time
    elapsed is mirrored to ``<episode_file>.sim_s`` for the runner.

    Published topics:
      /episode/coverage  (std_msgs/Float32)  – current coverage [0, 1]
    """

    def __init__(self) -> None:
        super().__init__('episode_manager')

        self.declare_parameter('coverage_threshold', 0.97)
        self.declare_parameter('episode_file', '/tmp/slam_episode_done')
        # Ignore /map messages for this many seconds after launch. A previous
        # episode's cartographer that survives SIGINT keeps publishing the old
        # (nearly complete) map on /map with TRANSIENT_LOCAL QoS, which would
        # make the new episode terminate instantly at ~99% coverage.
        self.declare_parameter('startup_grace_s', 15.0)
        # Abort episode if coverage does not increase by at least
        # stuck_min_delta within stuck_timeout_s seconds.
        self.declare_parameter('stuck_timeout_s', 90.0)
        self.declare_parameter('stuck_min_delta', 0.005)

        self.threshold: float = self.get_parameter('coverage_threshold').value
        self.episode_file: str = self.get_parameter('episode_file').value
        self.max_duration: float = load_episode_timeout()
        self.startup_grace_s: float = self.get_parameter('startup_grace_s').value
        self._stuck_timeout: float = self.get_parameter('stuck_timeout_s').value
        self._stuck_min_delta: float = self.get_parameter('stuck_min_delta').value

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

        self.sim_elapsed_file: str = self.episode_file + SIM_ELAPSED_SUFFIX
        self._start_time: float = self._now()
        self._done: bool = False
        self._last_coverage: float = 0.0
        self._last_coverage_change_time: float = self._now()

        # A file present at startup is always stale (this node is its sole
        # writer); leaving it makes explorers terminate instantly on the first
        # watch tick.
        self._clear_stale_episode_file()

        self._coverage_pub = self.create_publisher(Float32, '/episode/coverage', 10)

        self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)

        # Watchdog (sim-time timer): timeout check every 10 sim-s.
        self.create_timer(10.0, self._timeout_check)
        # Keep the runner's sim-time elapsed fresh even if the explorer ends
        # the episode (it writes the done file itself) before this node does.
        self.create_timer(1.0, self._write_sim_elapsed)

        self.get_logger().info(
            f'EpisodeManager started – threshold={self.threshold:.0%}, '
            f'max_duration={self.max_duration:.0f}s, '
            f'stuck_timeout={self._stuck_timeout:.0f}s, '
            f'episode_file={self.episode_file}'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._done:
            return

        elapsed = self._elapsed()
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

        # Stuck detection: abort if coverage stagnates
        if coverage - self._last_coverage >= self._stuck_min_delta:
            self._last_coverage = coverage
            self._last_coverage_change_time = self._now()
        else:
            stall_duration = self._now() - self._last_coverage_change_time
            if stall_duration >= self._stuck_timeout:
                elapsed = self._elapsed()
                self._signal_done(
                    reason=f'stuck (coverage={coverage:.1%})',
                    elapsed=elapsed,
                )
                return

        if coverage >= self.threshold:
            elapsed = self._elapsed()
            self._signal_done(reason=f'coverage={coverage:.1%}', elapsed=elapsed)

    def _timeout_check(self) -> None:
        if self._done:
            return
        elapsed = self._elapsed()
        if elapsed >= self.max_duration:
            self._signal_done(reason='timeout', elapsed=elapsed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """Node-clock time in seconds (sim time under ``use_sim_time``)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _elapsed(self) -> float:
        """Seconds of node-clock time since this node started."""
        return self._now() - self._start_time

    def _write_sim_elapsed(self) -> None:
        """Atomically publish the sim-time elapsed for training_runner.py."""
        tmp = self.sim_elapsed_file + '.tmp'
        Path(tmp).write_text(f'{self._elapsed():.3f}')
        os.replace(tmp, self.sim_elapsed_file)

    def _clear_stale_episode_file(self) -> None:
        stale = Path(self.episode_file)
        if stale.exists():
            stale.unlink()
            self.get_logger().info(
                f'Removed stale episode file {self.episode_file}'
            )
        Path(self.sim_elapsed_file).unlink(missing_ok=True)

    def _signal_done(self, *, reason: str, elapsed: float) -> None:
        self._done = True
        self.get_logger().info(
            f'Episode done – reason={reason}, elapsed={elapsed:.1f}s'
        )
        self._write_sim_elapsed()
        # The explorer may have ended the episode first (e.g. "exhausted"); keep
        # its more specific reason.
        if not Path(self.episode_file).exists():
            Path(self.episode_file).write_text(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
