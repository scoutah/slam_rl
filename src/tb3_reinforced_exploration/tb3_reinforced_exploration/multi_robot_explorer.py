"""Centralised multi-robot RL explorer (node ``rl_explorer``).

Standalone learned method for the Studienbericht. Subclasses the shared base
node and implements :meth:`select_goals` as a *centralised* frontier allocator
over both robots, following CTDE (centralised training, decentralised
execution): one parameter-sharing Q-network scores frontiers for either robot
and a single shared replay buffer pools both robots' experience. The learning
core lives in ``rl_core.TeamLearner`` (ROS-free, unit-tested); this file is the
ROS node.

Design choices (owned by issue #13):

* **Features** are size-invariant and team-aware
  (``rl_core.make_team_features``): ``[group_size_norm, path_norm, coverage,
  teammate_dist_norm, contested]``. Every distance is divided by the live merged
  grid's diagonal, so no absolute 8x8 coordinate enters the vector and the
  8x8-trained policy runs unchanged on 12x12.
* **Reward is shared** (global coverage gain per second, ``compute_reward``).
  A shared team reward directly rewards joint coverage — the quantity the
  research question measures — at the known cost of harder credit assignment;
  the team-aware features (teammate distance, contested flag) plus CTDE are what
  let the shared net still learn non-redundant, coordinated assignments.
* **Every episode is committed** to replay, including stuck/timeout ones
  (unlike the legacy single-robot node, which discarded aborted episodes). In
  the team setting a shared coverage reward already values partial progress, and
  discarding whole multi-robot episodes would throw away most experience.
* **Persist location is configurable** via the ``data_dir`` parameter
  (``$SLAM_RL_DATA_DIR``); no path is hard-coded.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from tb3_slam.coverage_monitor import check_coverage
from tb3_slam.multi_robot import ROBOT_NAMESPACES
from tb3_slam.multi_robot_explorer import (
    FrontierCandidate,
    Goal,
    MultiRobotExplorerNode,
    RobotState,
)

from .rl_core import (
    compute_reward,
    make_team_features,
    resolve_persist_dir,
    TEAM_FEATURE_DIM,
    TeamLearner,
    Transition,
)

CONTESTED_RADIUS_M = 0.6      # teammate goal this close to a frontier = contested


class RLExplorer(MultiRobotExplorerNode):
    """Learned centralised frontier assignment over both robots (CTDE)."""

    def __init__(self) -> None:
        super().__init__('rl_explorer')

        persist_dir = (Path(self._data_dir).expanduser() if self._data_dir
                       else resolve_persist_dir())
        self._learner = TeamLearner(
            feature_dim=TEAM_FEATURE_DIM, persist_dir=persist_dir,
        )
        self._learner.load()

        # Per-robot pending transition: features of the frontier it was last
        # sent to, plus the team coverage at dispatch. Closed out on the next
        # decision once the coverage gain (the reward) is known.
        self._pending: dict[str, tuple[np.ndarray, float, float]] = {
            ns: None for ns in ROBOT_NAMESPACES
        }
        self._committed = False

        self.create_timer(2.0, self._watch_episode_file)

        self.get_logger().info(
            f'RLExplorer (CTDE) ready – e={self._learner.epsilon:.3f}, '
            f'replay={len(self._learner.replay)}, '
            f'device={self._learner.device}, persist={persist_dir}'
        )

    # ------------------------------------------------------------------
    # Strategy: centralised frontier allocation over both robots
    # ------------------------------------------------------------------

    def select_goals(
        self,
        merged_map,
        robots: list[RobotState],
    ) -> dict[str, Goal | None]:
        """Assign at most one distinct frontier per robot via the shared policy.

        Robots are assigned greedily in turn; a frontier claimed by the first
        robot is removed from the second's candidate set, so the centralised
        allocator never double-books a frontier. Each per-robot decision also
        records the learning transition (reward for the previous goal) and
        trains the shared network.
        """
        diag = self._map_diagonal(merged_map)
        coverage = self._coverage(merged_map)

        goals: dict[str, Goal | None] = {r.name: None for r in robots}
        claimed: list[tuple[float, float]] = []

        for robot in robots:
            # Keep a busy robot on its current goal instead of re-rolling a new
            # (random at high epsilon) frontier every cycle; still reserve its
            # claim so the teammate won't double-book it.
            if robot.busy and robot.current_goal is not None:
                claimed.append((robot.current_goal.x, robot.current_goal.y))
                continue

            # Decentralised execution: this robot is scored on its own
            # candidate set, minus frontiers already claimed this cycle.
            candidates = [
                c for c in self.candidates_for(robot)
                if not self._is_claimed(c, claimed)
            ]
            if not candidates:
                self._close_transition(robot.name, coverage, next_feats=[],
                                       done=self._done(coverage))
                continue

            teammate = self._teammate(robot, robots)
            feats = np.stack([
                self._features(robot, c, teammate, coverage, diag)
                for c in candidates
            ])

            # Close out the previous goal's transition with this cycle's reward
            # and the new candidate set as the next state, then train.
            self._close_transition(robot.name, coverage,
                                   next_feats=list(feats),
                                   done=self._done(coverage))

            idx = self._learner.select_index(feats)
            chosen = candidates[idx]
            goals[robot.name] = Goal(
                x=chosen.x, y=chosen.y, group_size=chosen.group_size,
            )
            claimed.append((chosen.x, chosen.y))
            self._pending[robot.name] = (feats[idx], coverage, diag)

        return goals

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _features(
        self,
        robot: RobotState,
        cand: FrontierCandidate,
        teammate: RobotState | None,
        coverage: float,
        diag: float,
    ) -> np.ndarray:
        """Size-invariant, team-aware features for one (robot, frontier) pair."""
        cost = self.path_cost(robot, cand.x, cand.y)
        if cost is None:
            # Geometric fallback until Nav2 returns a cached path length.
            cost = math.hypot(cand.x - robot.x, cand.y - robot.y)

        if teammate is None:
            teammate_dist = diag          # no teammate known -> treat as far
            contested = False
        else:
            teammate_dist = math.hypot(cand.x - teammate.x, cand.y - teammate.y)
            contested = self._is_contested(cand, teammate)

        return make_team_features(
            group_size=cand.group_size,
            path_length=cost,
            coverage=coverage,
            teammate_dist=teammate_dist,
            map_diagonal=diag,
            contested=contested,
        )

    @staticmethod
    def _is_contested(cand: FrontierCandidate, teammate: RobotState) -> bool:
        """Return True if the teammate is already heading for this frontier."""
        goal = teammate.current_goal
        if goal is None:
            return False
        return math.hypot(cand.x - goal.x, cand.y - goal.y) <= CONTESTED_RADIUS_M

    @staticmethod
    def _is_claimed(cand: FrontierCandidate, claimed: list[tuple[float, float]]) -> bool:
        """Return True if another robot already took this frontier this cycle."""
        return any(
            math.hypot(cand.x - cx, cand.y - cy) <= CONTESTED_RADIUS_M
            for cx, cy in claimed
        )

    @staticmethod
    def _teammate(robot: RobotState, robots: list[RobotState]) -> RobotState | None:
        """Return the other robot (two-robot team), or None if alone."""
        others = [r for r in robots if r.name != robot.name]
        return others[0] if others else None

    @staticmethod
    def _map_diagonal(merged_map) -> float:
        """Diagonal of the merged grid in metres – the size-invariant scale."""
        res = merged_map.info.resolution
        w = merged_map.info.width * res
        h = merged_map.info.height * res
        return math.hypot(w, h)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _coverage(self, merged_map) -> float:
        if self._expected_cells <= 0:
            return 0.0
        coverage, _ = check_coverage(merged_map, self._expected_cells)
        return coverage

    def _done(self, coverage: float) -> bool:
        return coverage >= self._coverage_threshold

    def _close_transition(
        self,
        name: str,
        next_coverage: float,
        next_feats: list[np.ndarray],
        done: bool,
    ) -> None:
        """Turn a robot's pending goal into a replay transition and train.

        The reward is the *shared* team coverage gain per second since the goal
        was dispatched; every robot that acted this cycle receives the same team
        reward (shared-reward CTDE). Decision period is the dt.
        """
        pending = self._pending.get(name)
        if pending is None:
            return
        feats, prev_coverage, _diag = pending
        dt = self.get_parameter('decision_period').value
        reward = compute_reward(prev_coverage, next_coverage, dt)
        self._learner.push(Transition(feats, feats, reward, next_feats, done))
        self._pending[name] = None

        loss = self._learner.train_step()
        if loss is not None and self._learner.step_count % 50 == 0:
            self.get_logger().info(
                f'[rl] step={self._learner.step_count} loss={loss:.4f} '
                f'e={self._learner.epsilon:.3f} r={reward:.3f} '
                f'buf={len(self._learner.replay)}'
            )

    def _watch_episode_file(self) -> None:
        """Persist the shared learner once the episode ends.

        Both success and stuck/timeout episodes are kept: the shared coverage
        reward already discounts aborted episodes by their low gain, and throwing
        away whole team episodes would waste most of the collected experience.
        """
        if self._committed or not self._episode_file:
            return
        ep = Path(self._episode_file)
        if not ep.exists():
            return
        reason = ep.read_text().strip()
        # Close any still-open transitions at the final coverage.
        final_cov = (self._coverage(self._latest_map)
                     if self._latest_map is not None else 0.0)
        for name in list(self._pending):
            self._close_transition(name, final_cov, next_feats=[], done=True)
        self._learner.save()
        self._committed = True
        self.get_logger().info(
            f'[rl] Episode ended ({reason}) – learner saved to '
            f'{self._learner.persist_dir} (buf={len(self._learner.replay)})'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
