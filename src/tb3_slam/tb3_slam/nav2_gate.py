"""Bring one robot's Nav2 up, then exit — lifecycle driver and launch serialiser.

Run as a short-lived process per robot (``nav2_gate --ns r1``). It walks the
robot's Nav2 lifecycle nodes (``NAV2_LIFECYCLE_NODES``, in order) through
configure → activate, waits for ``navigate_to_pose`` and exits 0, or exits 1
once ``--timeout`` is spent. The launch file chains these with
``OnProcessExit`` so the robots' Nav2 bringups run strictly one after another
(r2's only starts once r1's is active) and the explorer starts the instant both
are ready.

It replaces Nav2's lifecycle_manager, which blocks forever on every
``change_state`` reply. Under load Fast DDS drops such a reply when the
caller's reply reader is not matched yet (node log: "failed to send response
... client will not receive response"), which hung r2's bringup in
"Configuring behavior_server". Here every call has a timeout and the node's
actual state is re-read before the next step, so a lost reply only costs one
retry. Without a lifecycle_manager no bonds are monitored (no respawn on
crash), which the episodic runner does not need.
"""
import argparse
import sys
import time

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.node import Node

from tb3_slam.multi_robot import NAV2_LIFECYCLE_NODES, topic

GET_STATE_TIMEOUT_S = 3.0      # reply wait for get_state before re-asking
CHANGE_STATE_TIMEOUT_S = 10.0  # reply wait for configure/activate before re-checking
RETRY_PAUSE_S = 0.5            # pause after a failed/transitional step

# Primary state → transition that moves it towards ACTIVE.
_NEXT_TRANSITION = {
    State.PRIMARY_STATE_UNCONFIGURED: Transition.TRANSITION_CONFIGURE,
    State.PRIMARY_STATE_INACTIVE: Transition.TRANSITION_ACTIVATE,
}


def next_transition(state_id: int) -> int | None:
    """Return the transition id to send for ``state_id``, or None to just re-poll.

    None covers ACTIVE (done), transitional states (still busy) and FINALIZED
    (dead); the caller tells those apart.
    """
    return _NEXT_TRANSITION.get(state_id)


class Nav2Gate(Node):
    """Minimal node that activates ``/<ns>/``'s Nav2 nodes and awaits navigate_to_pose."""

    def __init__(self, ns: str) -> None:
        super().__init__(f'nav2_gate_{ns}')
        self.ns = ns
        self._action = ActionClient(self, NavigateToPose, topic(ns, 'navigate_to_pose'))

    def _call(self, client: Client, request, timeout_s: float, deadline: float):
        """Call ``client`` and return the reply, or None on timeout / lost reply."""
        timeout_s = min(timeout_s, deadline - time.monotonic())
        if timeout_s <= 0 or not client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            client.remove_pending_request(future)
            return None
        return future.result()

    def activate(self, name: str, deadline: float) -> bool:
        """Drive lifecycle node ``name`` to ACTIVE; return False on deadline/finalized."""
        log = self.get_logger()
        get_state = self.create_client(GetState, topic(self.ns, f'{name}/get_state'))
        change_state = self.create_client(
            ChangeState, topic(self.ns, f'{name}/change_state'))
        try:
            while time.monotonic() < deadline:
                reply = self._call(
                    get_state, GetState.Request(), GET_STATE_TIMEOUT_S, deadline)
                if reply is None:
                    continue  # not launched yet, or the reply was lost: ask again
                state = reply.current_state
                if state.id == State.PRIMARY_STATE_ACTIVE:
                    return True
                if state.id == State.PRIMARY_STATE_FINALIZED:
                    log.error(f'[gate:{self.ns}] {name} is finalized')
                    return False
                transition = next_transition(state.id)
                if transition is None:  # mid-transition: wait for it to settle
                    time.sleep(RETRY_PAUSE_S)
                    continue
                verb = 'Configuring' if transition == Transition.TRANSITION_CONFIGURE \
                    else 'Activating'
                log.info(f'[gate:{self.ns}] {verb} {name}')
                result = self._call(
                    change_state,
                    ChangeState.Request(transition=Transition(id=transition)),
                    CHANGE_STATE_TIMEOUT_S, deadline,
                )
                if result is None:
                    log.warning(
                        f'[gate:{self.ns}] no reply from {name}/change_state, '
                        're-checking its state')
                elif not result.success:
                    log.warning(f'[gate:{self.ns}] {verb} {name} failed, retrying')
                    time.sleep(RETRY_PAUSE_S)
            return False
        finally:
            self.destroy_client(get_state)
            self.destroy_client(change_state)

    def wait_for_navigate(self, deadline: float) -> bool:
        """Return True once navigate_to_pose is discoverable before ``deadline``."""
        return self._action.wait_for_server(
            timeout_sec=max(0.0, deadline - time.monotonic()))


def bring_up_nav2(ns: str, timeout_s: float) -> bool:
    """Activate ``<ns>``'s Nav2 nodes in order; True once navigate_to_pose is ready.

    The deadline covers waiting for the nodes to be launched too; a
    non-positive ``timeout_s`` waits indefinitely.
    """
    deadline = float('inf') if timeout_s <= 0 else time.monotonic() + timeout_s
    node = Nav2Gate(ns)
    log = node.get_logger()
    try:
        log.info(f'[gate:{ns}] waiting for Nav2 nodes ...')
        ready = all(node.activate(name, deadline) for name in NAV2_LIFECYCLE_NODES) \
            and node.wait_for_navigate(deadline)
        if ready:
            log.info(f'[gate:{ns}] Nav2 ready')
        else:
            log.error(f'[gate:{ns}] Nav2 not ready after {timeout_s:.0f}s')
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
        ready = bring_up_nav2(parsed.ns, parsed.timeout)
    except KeyboardInterrupt:
        # Ctrl+C / episode teardown while waiting: exit quietly (launch's SIGINT
        # already shut the context down, so skip a second rclpy.shutdown()).
        sys.exit(1)
    rclpy.try_shutdown()
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()
