"""Block until one robot's Nav2 is ready, then exit — the launch serialiser.

Run as a short-lived process per robot (``nav2_gate --ns r1``). It waits for
that robot's ``navigate_to_pose`` action server to come up and exits 0 the
moment it does, or exits 1 on timeout. The launch file chains these with
``OnProcessExit`` so the robots' Nav2 lifecycle bringups run strictly one after
another (r2's only starts once r1's is active) and the explorer starts the
instant both are ready — replacing the old fixed wall-clock stagger/delay.
"""
import argparse
import sys

from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from tb3_slam.multi_robot import topic


class Nav2Gate(Node):
    """Minimal node that waits for ``/<ns>/navigate_to_pose`` to be available."""

    def __init__(self, ns: str) -> None:
        super().__init__(f'nav2_gate_{ns}')
        self._client = ActionClient(self, NavigateToPose, topic(ns, 'navigate_to_pose'))


def wait_for_nav2(ns: str, timeout_s: float) -> bool:
    """Return True once ``<ns>``'s navigate_to_pose server is ready, else False.

    Spins the node while ``wait_for_server`` blocks so action discovery can make
    progress; a non-positive ``timeout_s`` waits indefinitely.
    """
    node = Nav2Gate(ns)
    try:
        node.get_logger().info(f'[gate:{ns}] waiting for navigate_to_pose ...')
        deadline = None if timeout_s <= 0 else timeout_s
        ready = node._client.wait_for_server(timeout_sec=deadline)
        if ready:
            node.get_logger().info(f'[gate:{ns}] Nav2 ready')
        else:
            node.get_logger().error(
                f'[gate:{ns}] Nav2 not ready after {timeout_s:.0f}s'
            )
        return ready
    finally:
        node.destroy_node()


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', required=True, help='robot namespace, e.g. r1')
    parser.add_argument(
        '--timeout', type=float, default=120.0,
        help='seconds to wait before giving up (<=0 waits forever)',
    )
    # Strip ROS args (e.g. --ros-args) that launch may append.
    parsed, _ = parser.parse_known_args(args=args if args is not None else sys.argv[1:])

    rclpy.init()
    try:
        ready = wait_for_nav2(parsed.ns, parsed.timeout)
    finally:
        rclpy.shutdown()
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
