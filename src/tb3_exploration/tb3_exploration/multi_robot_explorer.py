"""Classical cost-utility multi-robot explorer (node ``classic_explorer``).

Baseline inherits the base node's empty ``select_goals``; the cost-utility
assignment is layered on later. No learned parameters, so the same logic runs
unchanged on any world size.
"""
import rclpy
from tb3_slam.multi_robot_explorer import MultiRobotExplorerNode


class ClassicExplorer(MultiRobotExplorerNode):
    """Cost-utility frontier assignment over both robots."""

    def __init__(self) -> None:
        super().__init__('classic_explorer')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClassicExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
