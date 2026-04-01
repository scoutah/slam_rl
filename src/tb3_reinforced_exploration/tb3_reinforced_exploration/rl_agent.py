from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from tb3_slam.coverage_monitor import (
    check_coverage,
    load_expected_cells,
    DEFAULT_COVERAGE_THRESHOLD,
)


class ReinforcedExplorationAgent(Node):
    def __init__(self) -> None:
        """
        Initialize the ReinforcedExplorationAgent node.
        """
        super().__init__('reinforced_exploration_agent')

        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')

        self._coverage_threshold: float = (
            self.get_parameter('coverage_threshold').value
        )
        self._episode_file: str = self.get_parameter('episode_file').value

        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning(
                '[coverage] World metadata file not found – '
                'coverage monitoring disabled until a new world is generated'
            )
            self._expected_cells: int = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected traversable cells: {self._expected_cells} '
                f'(threshold {self._coverage_threshold:.0%})'
            )

        self._coverage_done: bool = False

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10,
        )
        self.get_logger().info('Initializing Reinforced Exploration Agent')

    def map_callback(self, msg: OccupancyGrid) -> None:
        """
        Triggered whenever a new map is received on /map. Checks coverage progress
        and updates the agent's state for RL-based action selection.
        """
        if self._coverage_done:
            return

        # --- Coverage check ---
        if self._expected_cells > 0:
            coverage, known = check_coverage(msg, self._expected_cells)
            self.get_logger().info(
                f'[coverage] {coverage:.1%}  ({known}/{self._expected_cells} cells)',
                throttle_duration_sec=5.0,
            )
            if coverage >= self._coverage_threshold:
                self.get_logger().info(
                    f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%} – stopping exploration'
                )
                self._coverage_done = True
                if self._episode_file:
                    Path(self._episode_file).write_text(f'coverage={coverage:.1%}')
                return

        # Implement state update logic here

    def select_action(self, state):
        """
        Implement action selection logic here (e.g., epsilon-greedy, softmax, etc.)
        """
        pass

    def update_policy(self, reward, next_state):
        """
        Update the agent's policy based on the reward received and the next state.
        """
        pass


def main(args=None):
    """Main function to initialize the ReinforcedExplorationAgent and start the exploration process."""
    rclpy.init(args=args)
    agent = ReinforcedExplorationAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()