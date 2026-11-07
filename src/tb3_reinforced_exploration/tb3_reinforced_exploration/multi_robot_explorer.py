"""Multi-robot RL explorer (node ``rl_explorer``): CTDE with QMIX.

ROS side only; the learning core is ``rl_core.QMixLearner`` (ROS-free, tested).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from tb3_slam.coverage_monitor import check_coverage
from tb3_slam.frontier_utils import world_to_grid
from tb3_slam.information_gain import InformationField
from tb3_slam.multi_robot import ROBOT_NAMESPACES
from tb3_slam.multi_robot_explorer import (
    FrontierCandidate,
    Goal,
    MultiRobotExplorerNode,
    RobotState,
)

from .rl_core import (
    assemble_joint,
    CONTESTED_RADIUS_NORM,
    decentralised_round,
    JointEpisodeRecorder,
    JointTransition,
    make_global_state,
    make_local_observation,
    nearest_intent_dist,
    OBS_DIM,
    Point,
    QMixLearner,
    resolve_persist_dir,
    STATE_DIM,
)


class RLExplorer(MultiRobotExplorerNode):
    """Learned frontier selection for the robot team (Dec-POMDP, CTDE, QMIX).

    Mapping to the theory:

    * Hierarchical action (Hu et al.): the learned layer picks a frontier
      target, Nav2 (base node) drives there.
    * Dec-POMDP / decentralised execution: each robot acts on its own local
      observation o_i (``make_local_observation``) via ``QMixLearner.act``.
    * Intent communication (Tan et al.): robots decide in a fixed order
      (``ROBOT_NAMESPACES``) and announce their goal; busy robots' ongoing goals
      are announced at cycle start. Received intents enter o_i and a local rule
      drops candidates near them (no double-booking).
    * Parameter sharing: one agent Q-net for both homogeneous robots.
    * Centralised training: joint transitions with the global state s
      (``make_global_state``) train the QMIX mixer + agent net on the shared
      team reward (coverage gain per second), i.e. value-factorised credit
      assignment.

    Every episode (also stuck/timeout) is kept; persistence via ``data_dir`` /
    ``$SLAM_RL_DATA_DIR``. With ``eval_mode`` the loaded model is frozen:
    greedy (ε = 0), no replay, no training, no epsilon decay, no saving.
    """

    def __init__(self) -> None:
        super().__init__('rl_explorer')

        persist_dir = (Path(self._data_dir).expanduser() if self._data_dir
                       else resolve_persist_dir())
        self._agents = list(ROBOT_NAMESPACES)
        self._learner = QMixLearner(
            n_agents=len(self._agents), obs_dim=OBS_DIM, state_dim=STATE_DIM,
            persist_dir=persist_dir,
        )
        self._learner.load()
        self._eval_mode = bool(self.declare_parameter('eval_mode', False).value)
        if self._eval_mode:
            if not (self._learner.agent_path.exists()
                    and self._learner.mixer_path.exists()):
                raise RuntimeError(
                    f'eval_mode needs a trained model, none in {persist_dir}')
            self._learner.freeze()
        self._recorder = JointEpisodeRecorder()
        self._committed = False

        self.create_timer(2.0, self._watch_episode_file)

        self.get_logger().info(
            f'RLExplorer (QMIX/CTDE) ready – '
            f'{"EVAL (frozen, greedy)" if self._eval_mode else "training"}, '
            f'e={self._learner.epsilon:.3f}, '
            f'replay={len(self._learner.replay)}, '
            f'device={self._learner.device}, persist={persist_dir}'
        )

    # ------------------------------------------------------------------
    # Decentralised execution with intent communication
    # ------------------------------------------------------------------

    def select_goals(
        self,
        merged_map,
        robots: list[RobotState],
    ) -> dict[str, Goal | None]:
        """One decision cycle: each idle robot picks a frontier on its own o_i.

        Robots act in communication order. An idle robot only receives the
        teammates' intents (ongoing goals of busy robots, goals announced
        earlier this cycle), drops candidates within the contested radius of
        them, scores the rest with the shared agent net and announces its
        choice. Busy robots keep their goal (None). If any robot got a new goal
        this is a joint event and is recorded for centralised training.
        """
        diag = self._map_diagonal(merged_map)
        radius = CONTESTED_RADIUS_NORM * diag
        info = InformationField(
            np.asarray(merged_map.data, dtype=np.int8).reshape(
                merged_map.info.height, merged_map.info.width),
            merged_map.info.resolution)
        order = sorted(robots, key=self._agent_index)

        by_name = {r.name: r for r in order}
        busy = {r.name for r in robots
                if r.busy and r.current_goal is not None}
        busy_intents: dict[str, Point] = {
            r.name: (r.current_goal.x, r.current_goal.y)
            for r in order if r.name in busy
        }
        idle = [r.name for r in order if r.name not in busy]
        cands_cache: dict[str, list[FrontierCandidate]] = {
            name: self.candidates_for(by_name[name]) for name in idle
        }

        choices, picked, intents = decentralised_round(
            order=idle,
            busy_intents=busy_intents,
            candidates=cands_cache,
            observe=lambda name, c, received: self._observe(
                by_name[name], c.x, c.y, c.group_size, received, diag,
                merged_map, info),
            act=self._learner.act,     # local o_i only, shared agent net
            radius=radius,
        )
        goals: dict[str, Goal | None] = {r.name: None for r in robots}
        for name, c in choices.items():
            goals[name] = Goal(x=c.x, y=c.y, group_size=c.group_size)

        if choices and not self._eval_mode:
            options = {n: obs for n, (obs, _) in picked.items()}
            actions = {n: obs[idx] for n, (obs, idx) in picked.items()}
            for name in busy:
                # A busy robot cannot switch under this policy: its action is
                # its ongoing goal, observed with the intents it now receives.
                g = by_name[name].current_goal
                received = [p for n, p in intents.items() if n != name]
                feat = self._observe(by_name[name], g.x, g.y,
                                     g.group_size or 0, received, diag,
                                     merged_map, info)
                options[name], actions[name] = feat[None], feat
            self._record_event(merged_map, order, options, actions,
                               cands_cache, busy, diag)
        return goals

    def _observe(self, robot: RobotState, x: float, y: float,
                 group_size: float, received: list[Point],
                 diag: float, merged_map,
                 info: InformationField) -> np.ndarray:
        """Local observation o_i of frontier (x, y) for ``robot``.

        The information gain is the classical explorer's ray-cast estimate of
        the unseen area revealed at (x, y), minus what the received teammate
        intents already cover (only communicated information is used).
        """
        cost = self.path_cost(robot, x, y)
        if cost is None:
            # Geometric fallback until Nav2 returns a cached path length.
            cost = math.hypot(x - robot.x, y - robot.y)
        return make_local_observation(
            group_size=group_size,
            path_length=cost,
            intent_dist=nearest_intent_dist((x, y), received),
            map_diagonal=diag,
            info_gain=self._info_gain((x, y), received, merged_map, info),
        )

    @staticmethod
    def _info_gain(point: Point, intents: list[Point], merged_map,
                   info: InformationField) -> float:
        """Unseen-area fraction at ``point`` not yet covered by ``intents``."""
        cell = world_to_grid(*point, merged_map)
        if cell is None:
            return 0.0
        covered: set = set()
        for intent in intents:
            intent_cell = world_to_grid(*intent, merged_map)
            if intent_cell is not None:
                covered |= info.visible(intent_cell)
        return info.fraction(cell, frozenset(covered))

    def _agent_index(self, robot: RobotState) -> int:
        try:
            return self._agents.index(robot.name)
        except ValueError:
            return len(self._agents)

    def _now(self) -> float:
        """Return the current ROS time in seconds (sim time under use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _map_diagonal(merged_map) -> float:
        """Diagonal of the merged grid in metres – the size-invariant scale."""
        res = merged_map.info.resolution
        w = merged_map.info.width * res
        h = merged_map.info.height * res
        return math.hypot(w, h)

    # ------------------------------------------------------------------
    # Centralised training (QMIX)
    # ------------------------------------------------------------------

    def _coverage(self, merged_map) -> float:
        if self._expected_cells <= 0:
            return 0.0
        coverage, _ = check_coverage(merged_map, self._expected_cells)
        return coverage

    def _record_event(
        self,
        merged_map,
        robots: list[RobotState],
        options: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        cands_cache: dict[str, list[FrontierCandidate]],
        busy: set[str],
        diag: float,
    ) -> None:
        """Record a joint event (global state = training-only information)."""
        n = len(self._agents)
        act, mask, cand_list = assemble_joint(self._agents, actions, options)

        # Distinct frontier groups visible to the team (unfiltered sets).
        groups: set = set()
        for robot in robots:
            cands = cands_cache.get(robot.name)
            if cands is None:
                cands = self.candidates_for(robot)
            groups.update(c.frontier_cells for c in cands)
        coverage = self._coverage(merged_map)
        state = make_global_state(
            coverage=coverage,
            robot_positions=[(r.x, r.y) for r in robots],
            n_frontier_groups=len(groups),
            n_busy=len(busy),
            n_agents=n,
            map_diagonal=diag,
        )
        self._learn(self._recorder.record_event(
            act, mask, cand_list, state, coverage, self._now()))

    def _learn(self, t: JointTransition | None) -> None:
        """Push a closed joint transition and run one QMIX train step."""
        if t is None:
            return
        self._learner.push(t)
        loss = self._learner.train_step()
        if loss is not None and self._learner.step_count % 50 == 0:
            self.get_logger().info(
                f'[rl] step={self._learner.step_count} loss={loss:.4f} '
                f'e={self._learner.epsilon:.3f} r={t.reward:.3f} '
                f'buf={len(self._learner.replay)}'
            )

    def _watch_episode_file(self) -> None:
        """Close the last joint event (done) and persist once the episode ends.

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
        if self._eval_mode:
            # Frozen model: no learning step, no epsilon decay, no checkpoint.
            self._committed = True
            self.get_logger().info(f'[rl] Episode ended ({reason}) – eval, nothing saved')
            return
        final_cov = (self._coverage(self._latest_map)
                     if self._latest_map is not None else 0.0)
        self._learn(self._recorder.close(final_cov, self._now()))
        # Guarded by _committed, so epsilon decays exactly once per episode.
        self._learner.end_episode()
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
