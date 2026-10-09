"""Exercise the real explorer against controlled clock and action boundaries."""
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.task import Future
from rclpy.time import Time

from tb3_exploration.frontier_explorer import FrontierExplorer


class _Handle:
    accepted = True

    def __init__(self):
        self.result = Future()
        self.cancelled = False

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancelled = True
        return Future()


class _Navigation:
    def __init__(self):
        self.requests = []

    def wait_for_server(self, timeout_sec):
        return True

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        response = Future()
        self.requests.append(response)
        return response


@pytest.fixture
def explorer(monkeypatch):
    rclpy.init()
    node = FrontierExplorer()
    boundary = SimpleNamespace(seconds=0.0, position=(0.0, 0.0))
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(seconds=boundary.seconds)))
    monkeypatch.setattr(node, '_get_robot_pos', lambda: boundary.position)
    node._goal_active = True
    node._current_target = (20, 40)
    node._goal_dispatch_time = 0.0
    node._goal_dispatch_pos = (0.0, 0.0)
    node._goal_dispatch_world = (3.0, 0.0)
    node._goal_last_progress_time = 0.0
    node._goal_last_progress_pos = (0.0, 0.0)
    node._goal_handle = _Handle()
    try:
        yield node, boundary
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stationary_goal_cancelled_without_waiting_45_seconds(explorer):
    node, boundary = explorer
    handle = node._goal_handle
    boundary.seconds = 9.0
    node._supervise_active_goal()
    assert handle.cancelled
    assert not node._goal_active
    assert (20, 40) in node._tried


def test_real_motion_restarts_no_progress_window(explorer):
    node, boundary = explorer
    boundary.seconds = 7.0
    boundary.position = (0.3, 0.0)
    node._supervise_active_goal()
    boundary.seconds = 14.0
    node._supervise_active_goal()
    assert node._goal_active
    boundary.seconds = 16.0
    node._supervise_active_goal()
    assert not node._goal_active


def test_late_cancel_result_cannot_clear_a_new_goal(explorer, monkeypatch):
    node, boundary = explorer
    navigation = _Navigation()
    monkeypatch.setattr(node, '_nav_client', navigation)
    node._send_goal(3.0, 0.0)
    old = _Handle()
    navigation.requests[0].set_result(old)
    boundary.seconds = 9.0
    node._supervise_active_goal()
    node._send_goal(4.0, 0.0)
    navigation.requests[1].set_result(_Handle())
    old.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    assert node._goal_active


def _ranking_boundaries(node, monkeypatch):
    planner = _Navigation()
    navigation = _Navigation()
    monkeypatch.setattr(node, '_planner_client', planner)
    monkeypatch.setattr(node, '_nav_client', navigation)
    msg = OccupancyGrid()
    msg.info.resolution = 0.05
    return planner, navigation, msg


def test_planner_timeout_does_not_dispatch_an_unverified_goal(explorer, monkeypatch):
    node, boundary = explorer
    planner, navigation, msg = _ranking_boundaries(node, monkeypatch)
    node._start_ranking([(20, 40), (30, 50)], msg)
    planner.requests[0].set_result(SimpleNamespace(accepted=False))
    boundary.seconds = 9.0
    node._supervise_ranking()
    assert not navigation.requests
    assert not node._ranking_in_progress


def test_late_planner_reply_does_not_finish_a_new_ranking(explorer, monkeypatch):
    node, boundary = explorer
    planner, _, msg = _ranking_boundaries(node, monkeypatch)
    node._start_ranking([(20, 40)], msg)
    boundary.seconds = 9.0
    node._supervise_ranking()
    node._start_ranking([(30, 50)], msg)
    planner.requests[0].set_result(SimpleNamespace(accepted=False))
    assert node._ranking_in_progress
    assert node._ranking_pending == 1


def test_missing_planner_does_not_dispatch_without_a_plan(explorer, monkeypatch):
    node, _ = explorer
    planner, navigation, msg = _ranking_boundaries(node, monkeypatch)
    monkeypatch.setattr(planner, 'server_is_ready', lambda: False)
    node._start_ranking([(20, 40)], msg)
    assert not navigation.requests


@pytest.mark.parametrize('changed', [
    'wall', 'disconnected', 'origin_shift', 'unchanged', 'shifted_free',
])
def test_queued_goal_is_rechecked_in_latest_map(explorer, monkeypatch, changed):
    node, boundary = explorer
    navigation = _Navigation()
    monkeypatch.setattr(node, '_nav_client', navigation)
    boundary.position = (0.525, 0.775)
    old = OccupancyGrid()
    old.info.resolution = 0.05
    old.info.height = 40
    old.info.width = 70
    old.data = [0] * (40 * 70)
    latest = OccupancyGrid()
    latest.info.resolution = 0.05
    latest.info.height = 40
    latest.info.width = 70
    grid = np.zeros((40, 70), dtype=np.int8)
    if changed == 'disconnected':
        grid[:, 25:30] = 50
    elif changed in ('origin_shift', 'shifted_free'):
        latest.info.origin.position.x = -0.5
        if changed == 'origin_shift':
            grid[15, 50] = 95  # Old cell (15,40) is now (15,50) in the same world frame.
    elif changed == 'wall':
        grid[15, 40] = 95
    latest.data = grid.ravel().tolist()
    node._current_map = old
    node._latest_map = latest
    node._centroid_queue = [(15, 40)]
    node._try_next_goal()
    if changed in ('unchanged', 'shifted_free'):
        assert len(navigation.requests) == 1
        assert node._goal_active
        assert node._goal_dispatch_world == pytest.approx((2.024995, 0.774995))
    else:
        assert not navigation.requests
        assert not node._goal_active


def test_queued_goal_waits_for_robot_pose(explorer, monkeypatch):
    node, boundary = explorer
    navigation = _Navigation()
    monkeypatch.setattr(node, '_nav_client', navigation)
    boundary.position = None
    node._current_map = OccupancyGrid()
    node._centroid_queue = [(15, 40)]
    node._try_next_goal()
    assert not navigation.requests
    assert not node._goal_active
