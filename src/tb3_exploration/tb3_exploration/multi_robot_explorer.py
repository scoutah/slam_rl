"""Standalone classical explorer: verified Nav2 costs and marginal team utility."""
from collections import deque
import math

from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose
import numpy as np
import rclpy
from tb3_exploration.cost_utility import Bid, greedy_assignment, visible_information
from tb3_slam.frontier_utils import (
    GOAL_CLEARANCE_M,
    grid_to_world,
    MIN_GOAL_GRID_DIST,
    reachable_goal_cells,
    world_to_grid,
)
from tb3_slam.multi_robot_explorer import (
    candidates_for_robot,
    Goal,
    MultiRobotExplorerNode,
    RobotState,
)
from tb3_slam.path_cost import path_length


class ClassicExplorer(MultiRobotExplorerNode):
    """Rank feasible pairs, then greedily assign non-redundant frontier goals."""

    def __init__(self) -> None:
        super().__init__('classic_explorer')
        self._sensor_radius = self.declare_parameter('sensor_radius', 3.5).value
        self._alpha = self.declare_parameter('utility_alpha', 1.0).value
        self._beta = self.declare_parameter('utility_beta', 0.2).value
        self._ranking_timeout = self.declare_parameter('ranking_timeout', 8.0).value
        if any(not math.isfinite(v) or v <= 0 for v in (
                self._sensor_radius, self._alpha, self._beta, self._ranking_timeout)):
            raise ValueError('sensor radius, utility weights and ranking timeout must be positive')
        self._round = None
        self._round_generation = 0
        self._pending_plans = {}
        self._planning_candidates = {}
        self._confirmed = []
        self._failed = {ns: [] for ns in self._robots}
        self._assigned_frontiers = {}

    def _decide(self) -> None:
        if self._coverage_done or self._latest_map is None:
            return
        msg = self._latest_map
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        for io in self._robots.values():
            if not io.goal_active or io.current_goal is None:
                continue
            pos = self._robot_pos(io)
            if pos is None:
                continue
            target = world_to_grid(io.current_goal.x, io.current_goal.y, msg)
            reachable = reachable_goal_cells(
                grid, world_to_grid(*pos, msg), math.ceil(GOAL_CLEARANCE_M / msg.info.resolution))
            if target is None or not reachable[target]:
                self._cancel_goal(io, 'target no longer reachable on merged map')
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._round is not None:
            if now - self._round_started >= self._ranking_timeout:
                self.get_logger().warning('[rank] timeout; using only confirmed Nav2 paths')
                self._finish_ranking(timed_out=True)
            return

        self._round_map = self._latest_map
        self._round_starts = {}
        queues = {}
        for ns, io in self._robots.items():
            pos = self._robot_pos(io)
            if io.goal_active or pos is None or not io.path_client.server_is_ready():
                continue
            candidates = [c for c in candidates_for_robot(
                self._round_map, RobotState(ns, *pos)) if not self._was_failed(ns, c.x, c.y)]
            if candidates:
                # Short plans first so a slow planner still yields useful verified pairs.
                candidates.sort(key=lambda c: (math.hypot(c.x - pos[0], c.y - pos[1]), c.cell))
                queues[ns] = deque(candidates)
                self._round_starts[ns] = pos
        if not queues:
            return
        self._round = queues
        self._round_started = now
        self._round_generation += 1
        self._confirmed = []
        self._pending_plans = {ns: None for ns in queues}
        for ns in queues:
            self._plan_next(ns, self._round_generation)

    def _plan_next(self, ns, generation) -> None:
        if generation != self._round_generation or self._round is None:
            return
        if not self._round[ns]:
            self._pending_plans.pop(ns, None)
            self._planning_candidates.pop(ns, None)
            if not self._pending_plans:
                self._finish_ranking()
            return
        candidate = self._round[ns].popleft()
        self._planning_candidates[ns] = candidate
        request = ComputePathToPose.Goal()
        request.use_start = True
        request.start = self._pose_stamped(*self._round_starts[ns])
        request.goal = self._pose_stamped(candidate.x, candidate.y)
        try:
            future = self._robots[ns].path_client.send_goal_async(request)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'[rank] {ns} planner request failed: {error}')
            self._plan_next(ns, generation)
            return
        future.add_done_callback(
            lambda f: self._plan_response(f, ns, candidate, generation))

    def _plan_response(self, future, ns, candidate, generation) -> None:
        try:
            handle = future.result()
        except Exception:  # noqa: BLE001
            self._plan_next(ns, generation)
            return
        if generation != self._round_generation or self._round is None:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self._plan_next(ns, generation)
            return
        self._pending_plans[ns] = handle
        handle.get_result_async().add_done_callback(
            lambda f: self._plan_result(f, ns, candidate, generation))

    def _plan_result(self, future, ns, candidate, generation) -> None:
        if generation != self._round_generation or self._round is None:
            return
        self._pending_plans[ns] = None
        try:
            response = future.result()
            result = response.result
            length = path_length(result.path)
            if (response.status == GoalStatus.STATUS_SUCCEEDED and result.error_code == 0
                    and len(result.path.poses) >= 2 and length is not None
                    and math.isfinite(length) and length > 0):
                self._confirmed.append((ns, candidate, length))
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'[rank] {ns} planner result failed: {error}')
        self._plan_next(ns, generation)

    def _finish_ranking(self, timed_out=False) -> None:
        self._round_generation += 1
        self._round = None
        if timed_out:
            for ns, candidate in self._planning_candidates.items():
                self._failed[ns].append((candidate.x, candidate.y))
        self._planning_candidates.clear()
        for handle in self._pending_plans.values():
            if handle is not None:
                handle.cancel_goal_async()
        self._pending_plans.clear()
        if self._coverage_done:
            return
        msg = self._latest_map
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        resolution = msg.info.resolution
        information = {}

        def visible(cell):
            if cell not in information:
                information[cell] = visible_information(
                    grid, cell, resolution, self._sensor_radius)
            return information[cell]

        covered, reserved = set(), set()
        for ns, io in self._robots.items():
            if io.goal_active and io.current_goal is not None:
                cell = world_to_grid(io.current_goal.x, io.current_goal.y, msg)
                if cell is not None:
                    covered.update(visible(cell))
                    reserved.add(cell)
                for point in self._assigned_frontiers.get(ns, ()):
                    frontier_cell = world_to_grid(*point, msg)
                    if frontier_cell is not None:
                        reserved.add(frontier_cell)

        reachable, poses, bids, targets = {}, {}, [], {}
        for ns, candidate, length in self._confirmed:
            io = self._robots[ns]
            pos = self._robot_pos(io)
            cell = world_to_grid(candidate.x, candidate.y, msg)
            if io.goal_active or pos is None or cell is None:
                continue
            robot_cell = world_to_grid(*pos, msg)
            if ns not in reachable:
                reachable[ns] = reachable_goal_cells(
                    grid, robot_cell, math.ceil(GOAL_CLEARANCE_M / resolution))
                poses[ns] = pos
            if (not reachable[ns][cell] or robot_cell is None
                    or math.dist(cell, robot_cell) < MIN_GOAL_GRID_DIST):
                continue
            frontier = frozenset(
                mapped for old_cell in candidate.frontier_cells
                if (mapped := world_to_grid(*grid_to_world(old_cell, self._round_map), msg))
                is not None)
            bids.append(Bid(ns, cell, length, visible(cell), frontier))
            targets[ns, cell] = candidate

        chosen = greedy_assignment(
            bids, resolution, self._sensor_radius, alpha=self._alpha, beta=self._beta,
            reserved_information=frozenset(covered), reserved_frontiers=frozenset(reserved))
        for bid in chosen:
            candidate = targets[bid.robot, bid.cell]
            self._assigned_frontiers[bid.robot] = [grid_to_world(c, msg) for c in bid.frontier]
            self.get_logger().info(
                f'[assign] {bid.robot} target=({candidate.x:.2f},{candidate.y:.2f}) '
                f'path={bid.path_length:.2f}m '
                f'gain={len(bid.information - covered) * resolution**2:.2f}m2')
            covered.update(bid.information)
            self._dispatch(self._robots[bid.robot], Goal(
                candidate.x, candidate.y, candidate.group_size, bid.path_length), poses[bid.robot])

    def _was_failed(self, ns, x, y) -> bool:
        return any(math.hypot(x - gx, y - gy) < 0.4 for gx, gy in self._failed[ns])

    def _remember_failure(self, io) -> None:
        if io.current_goal is not None:
            self._failed[io.ns].append((io.current_goal.x, io.current_goal.y))

    def _cancel_goal(self, io, reason) -> None:
        if reason != 'episode finished':
            self._remember_failure(io)
        self._assigned_frontiers.pop(io.ns, None)
        super()._cancel_goal(io, reason)

    def _on_goal_response(self, future, ns, generation) -> None:
        try:
            handle = future.result()
        except Exception:  # noqa: BLE001
            if generation == self._robots[ns].goal_generation:
                self._cancel_goal(self._robots[ns], 'navigation request failed')
            return
        if generation == self._robots[ns].goal_generation and not handle.accepted:
            self._remember_failure(self._robots[ns])
        super()._on_goal_response(future, ns, generation)

    def _on_goal_result(self, future, ns, generation) -> None:
        io = self._robots[ns]
        if generation != io.goal_generation:
            return
        try:
            succeeded = future.result().status == GoalStatus.STATUS_SUCCEEDED
        except Exception:  # noqa: BLE001
            self._cancel_goal(io, 'navigation result failed')
            return
        moved = self._progress_since_dispatch(io)
        if succeeded and moved is not None and moved >= self._min_progress:
            self._failed[ns].clear()
        else:
            self._remember_failure(io)
        self._assigned_frontiers.pop(ns, None)
        super()._on_goal_result(future, ns, generation)

    def _finish(self, coverage) -> None:
        super()._finish(coverage)
        if self._round is not None:
            self._finish_ranking()
        for io in self._robots.values():
            if io.goal_active:
                self._cancel_goal(io, 'episode finished')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClassicExplorer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
