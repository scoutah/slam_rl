"""Centralised DQN multi-robot explorer (node ``rl_explorer``).

Baseline inherits the base node's empty ``select_goals``; the learned policy is
layered on later. Features must stay size-invariant so it transfers to the unseen
12 m world. Persist location comes from the ``data_dir`` param (``$SLAM_RL_DATA_DIR``).
"""
import rclpy
from tb3_slam.multi_robot_explorer import MultiRobotExplorerNode


class RLExplorer(MultiRobotExplorerNode):
    """Learned centralised frontier assignment over both robots."""

    def __init__(self) -> None:
        super().__init__('rl_explorer')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
