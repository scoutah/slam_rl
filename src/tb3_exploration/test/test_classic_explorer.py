"""Exercise the classical node with real maps and controlled Nav2 boundaries."""
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.task import Future
from rclpy.time import Time

from tb3_exploration.multi_robot_explorer import ClassicExplorer
from tb3_slam.multi_robot_explorer import candidates_for_robot, RobotState


def _map():
    grid = np.full((40, 120), 95, dtype=np.int8)
    grid[5:35, 5:115] = 0
    grid[5:35, 10] = 50
    grid[5:35, 109] = 50
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.height, msg.info.width = grid.shape
    msg.info.resolution = 0.05
    msg.data = grid.ravel().tolist()
    return msg


class _Handle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.cancelled = False
        self.result = Future()

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancelled = True
        return Future()


class _Action:
    def __init__(self):
        self.requests = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, request):
        response = Future()
        self.requests.append(SimpleNamespace(request=request, response=response, handle=None))
        return response

    def finish_plan(self, index, length=2.0, status=GoalStatus.STATUS_SUCCEEDED,
                    error_code=0, accepted=True):
        entry = self.requests[index]
        handle = _Handle(accepted)
        entry.handle = handle
        entry.response.set_result(handle)
        result = ComputePathToPose.Result()
        result.error_code = error_code
        if length is not None:
            a, b = PoseStamped(), PoseStamped()
            b.pose.position.x = float(length)
            result.path.poses = [a, b]
        handle.result.set_result(SimpleNamespace(status=status, result=result))
        return handle


@pytest.fixture
def explorer(monkeypatch):
    rclpy.init()
    node = ClassicExplorer()
    boundary = SimpleNamespace(seconds=0.0,
                               positions={'r1': (1.25, 1.0), 'r2': (4.75, 1.0)})
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(
        now=lambda: Time(seconds=boundary.seconds)))
    monkeypatch.setattr(node, '_robot_pos', lambda io: boundary.positions.get(io.ns))
    node._latest_map = _map()
    # Identity map -> <ns>/map so request poses keep merged-map coordinates;
    # the frame (not the offset) is what Nav2 needs, see test below.
    for ns in node._robots:
        tf = TransformStamped()
        tf.header.frame_id = 'map'
        tf.child_frame_id = f'{ns}/map'
        tf.transform.rotation.w = 1.0
        node._tf_buffer.set_transform_static(tf, 'test')
    for io in node._robots.values():
        io.path_client = _Action()
        io.nav_client = _Action()
    try:
        yield node, boundary
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _finish_plans(node):
    for io in node._robots.values():
        count = 0
        while count < len(io.path_client.requests):
            entry = io.path_client.requests[count]
            if not entry.response.done():
                io.path_client.finish_plan(count, length=abs(
                    entry.request.goal.pose.position.x - node._robot_pos(io)[0]) + 0.1)
            count += 1


def test_candidates_preserve_frontier_identity_across_robot_approaches():
    msg = _map()
    for robot in [RobotState('r1', 1.25, 1.0), RobotState('r2', 4.75, 1.0)]:
        candidates = candidates_for_robot(msg, robot)
        assert candidates
        assert all(len(c.frontier_cells) == c.group_size for c in candidates)
        assert all(c.frontier_cells for c in candidates)


def test_plans_are_serial_per_robot_and_navigation_waits_for_confirmed_costs(explorer):
    node, _ = explorer
    node._decide()
    assert len(node._robots['r1'].path_client.requests) == 1
    assert len(node._robots['r2'].path_client.requests) == 1
    assert all(not io.nav_client.requests for io in node._robots.values())
    _finish_plans(node)
    assert all(io.goal_active for io in node._robots.values())
    left, right = [node._robots[ns].current_goal for ns in ('r1', 'r2')]
    assert left.x < 2.0 and right.x > 4.0
    assert left.path_len > 0 and right.path_len > 0
    for ns, io in node._robots.items():
        request = io.nav_client.requests[0].request
        assert request.pose.header.frame_id == f'{ns}/map'
        plan = io.path_client.requests[0].request
        assert plan.start.header.frame_id == plan.goal.header.frame_id == f'{ns}/map'


@pytest.mark.parametrize('plan_result', [
    {'accepted': False}, {'status': GoalStatus.STATUS_ABORTED}, {'error_code': 207},
    {'length': None}, {'length': 0.0}, {'length': float('nan')},
])
def test_failed_or_unusable_plans_never_become_navigation_goals(explorer, plan_result):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    planner = node._robots['r1'].path_client
    count = 0
    assert planner.requests
    while count < len(planner.requests):
        planner.finish_plan(count, **plan_result)
        count += 1
    assert not node._robots['r1'].nav_client.requests


def test_ranking_timeout_uses_confirmed_paths_and_cancels_late_acceptance(explorer):
    node, boundary = explorer
    node._decide()
    planner = node._robots['r1'].path_client
    planner.finish_plan(0, length=1.0)
    boundary.seconds = 9.0
    node._decide()
    assert node._robots['r1'].goal_active
    assert not node._robots['r2'].goal_active
    late = node._robots['r2'].path_client.finish_plan(0, length=1.0)
    assert late.cancelled
    assert not node._robots['r2'].nav_client.requests


def test_latest_map_rejects_a_previously_planned_target(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    node._latest_map = _map()
    node._latest_map.data = [95] * len(node._latest_map.data)
    _finish_plans(node)
    assert not node._robots['r1'].nav_client.requests


def test_busy_robot_keeps_its_goal_and_next_round_reserves_its_information(explorer):
    node, _ = explorer
    node._decide()
    _finish_plans(node)
    r1, r2 = node._robots['r1'], node._robots['r2']
    original = r1.current_goal
    r2.goal_active = False
    r2.current_goal = None
    node._decide()
    _finish_plans(node)
    assert len(r1.nav_client.requests) == 1
    assert r1.current_goal is original
    assert r2.current_goal.x > 4.0


def test_stall_cancels_goal_and_does_not_immediately_retry_same_target(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    _finish_plans(node)
    io = node._robots['r1']
    old = io.current_goal
    handle = _Handle()
    io.nav_client.requests[0].response.set_result(handle)
    boundary.seconds = 9.0
    node._supervise_goals()
    assert handle.cancelled and not io.goal_active
    node._decide()
    _finish_plans(node)
    assert io.current_goal is None or (io.current_goal.x, io.current_goal.y) != (old.x, old.y)


def test_late_navigation_result_cannot_clear_a_new_goal(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    _finish_plans(node)
    io = node._robots['r1']
    old = _Handle()
    io.nav_client.requests[0].response.set_result(old)
    boundary.seconds = 9.0
    node._supervise_goals()
    node._decide()
    _finish_plans(node)
    new_goal = io.current_goal
    assert new_goal is not None
    old.result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
    assert io.goal_active and io.current_goal is new_goal


def test_queued_world_goal_survives_a_map_origin_shift(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    first_target = node._robots['r1'].path_client.requests[0].request.goal.pose.position
    previous = node._latest_map
    old_grid = np.asarray(previous.data, dtype=np.int8).reshape(40, 120)
    shifted = np.pad(old_grid, ((0, 0), (20, 0)), constant_values=95)
    msg = _map()
    msg.info.origin.position.x = -1.0
    msg.info.width = 140
    msg.data = shifted.ravel().tolist()
    node._latest_map = msg
    _finish_plans(node)
    goal = node._robots['r1'].current_goal
    assert goal is not None
    assert (goal.x, goal.y) == (first_target.x, first_target.y)


def test_confirmed_path_cost_can_reverse_euclidean_preference(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    planner = node._robots['r1'].path_client
    near_x = planner.requests[0].request.goal.pose.position.x
    planner.finish_plan(0, length=20.0)
    planner.finish_plan(1, length=1.0)
    assert node._robots['r1'].current_goal.x != near_x
    assert node._robots['r1'].current_goal.path_len == 1.0


def test_active_target_that_becomes_occupied_is_cancelled(explorer):
    node, _ = explorer
    node._decide()
    _finish_plans(node)
    io = node._robots['r1']
    handle = _Handle()
    io.nav_client.requests[0].response.set_result(handle)
    node._latest_map.data = [95] * len(node._latest_map.data)
    node._decide()
    assert handle.cancelled and not io.goal_active


def test_episode_completion_cancels_pending_plans_and_navigation(explorer):
    node, _ = explorer
    node._decide()
    planner = node._robots['r1'].path_client
    entry = planner.requests[0]
    handle = _Handle()
    entry.response.set_result(handle)
    node._finish(1.0)
    assert handle.cancelled
    late = node._robots['r2'].path_client.finish_plan(0)
    assert late.cancelled
    assert all(not io.nav_client.requests for io in node._robots.values())


def test_hung_nearest_planner_does_not_starve_other_frontiers(explorer):
    node, boundary = explorer
    boundary.positions.pop('r2')
    node._decide()
    planner = node._robots['r1'].path_client
    nearest = planner.requests[0].request.goal.pose.position.x
    boundary.seconds = 9.0
    node._decide()
    node._decide()
    assert len(planner.requests) == 2
    assert planner.requests[1].request.goal.pose.position.x != nearest
    planner.finish_plan(1, length=3.0)
    assert node._robots['r1'].goal_active


def test_episode_file_end_stops_ranking_without_marking_failures(explorer, tmp_path):
    node, _ = explorer
    node._episode_file = str(tmp_path / 'episode_done')
    node._decide()
    _finish_plans(node)
    assert all(io.goal_active for io in node._robots.values())
    node._robots['r2'].goal_active = False
    node._robots['r2'].current_goal = None
    node._decide()
    assert node._round is not None
    (tmp_path / 'episode_done').write_text('reason=stuck')
    node._watch_episode_end()
    assert node._round is None
    assert not any(io.goal_active for io in node._robots.values())
    assert not any(node._failed.values())
