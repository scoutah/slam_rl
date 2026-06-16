"""
RL-based exploration node – Pointwise DQN.

Each frontier centroid is scored individually by a small Q-network whose
input is a per-frontier feature vector:

    [group_size_normalized, path_length, coverage]

The agent picks the centroid with the highest Q-value (epsilon-greedy).
This lets the network generalise across episodes because the features
have stable semantics regardless of frontier ordering.

The replay buffer and model are persisted to disk so training
accumulates across episodes launched by training_runner.py.
"""
import collections
import math
from pathlib import Path
import pickle
import random
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from tb3_slam.coverage_monitor import (
    check_coverage,
    DEFAULT_COVERAGE_THRESHOLD,
    load_expected_cells,
)
from tb3_slam.frontier_utils import (
    centroids_from_map_debug,
    grid_to_world,
    world_to_grid,
)
from tb3_slam.frontier_viz import build_frontier_markers
from tf2_ros import Buffer, TransformListener
import tf_transformations
from visualization_msgs.msg import MarkerArray

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        'PyTorch is required for the RL explorer.  Install it with:\n'
        '    pip install torch'
    ) from exc

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
FEATURE_DIM = 3             # [group_size_norm, path_length, coverage]

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.98

TARGET_UPDATE_INTERVAL = 50
MIN_REPLAY_SIZE = 50

PERSIST_DIR = (
    Path.home() / 'projects' / 'slam_rl' / 'rl_data'
)
MODEL_PATH = PERSIST_DIR / 'model.pt'
REPLAY_PATH = PERSIST_DIR / 'replay.pkl'
META_PATH = PERSIST_DIR / 'meta.pkl'

RANKING_TIMEOUT_S = 8.0
GOAL_TIMEOUT_S = 45.0
IDLE_SPIN_S = 15.0          # spin in place after this many seconds idle


# ─────────────────────────────────────────────
# Pointwise Q-Network
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """Map per-frontier features to a scalar Q-value."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values. x: (batch, features) -> (batch, 1)."""
        return self.net(x)


# ─────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────

Transition = collections.namedtuple(
    'Transition',
    ['state', 'action_features', 'reward', 'next_candidates', 'done'],
)


class ReplayBuffer:
    """Fixed-capacity replay buffer with disk persistence."""

    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self._buf: collections.deque[Transition] = (
            collections.deque(maxlen=capacity)
        )

    def push(self, t: Transition) -> None:
        """Add a transition."""
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch."""
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def save(self, path: Path) -> None:
        """Persist buffer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self._buf), f)

    def load(self, path: Path) -> None:
        """Load buffer from disk."""
        if not path.exists():
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._buf = collections.deque(data, maxlen=self._buf.maxlen)


# ─────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────

class RLExplorer(Node):
    """Navigate to frontier centroids selected by a pointwise DQN."""

    def __init__(self) -> None:
        super().__init__('rl_explorer')

        self.declare_parameter(
            'coverage_threshold', DEFAULT_COVERAGE_THRESHOLD,
        )
        self.declare_parameter('episode_file', '')

        self._coverage_threshold: float = (
            self.get_parameter('coverage_threshold').value
        )
        self._episode_file: str = (
            self.get_parameter('episode_file').value
        )
        self._coverage_done = False

        # ── Coverage tracking ─────────────────────────────────
        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning(
                '[coverage] No world metadata – disabled'
            )
            self._expected_cells = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected cells: {expected} '
                f'(threshold {self._coverage_threshold:.0%})'
            )

        # ── Navigation state ──────────────────────────────────
        self._goal_active = False
        self._current_map: OccupancyGrid | None = None
        self._current_target_cell: tuple[int, int] | None = None
        self._goal_handle: ClientGoalHandle | None = None
        self._goal_dispatch_time: float | None = None

        # ── Ranking state (Nav2 path lengths) ─────────────────
        self._ranking_in_progress = False
        self._ranking_pending: int = 0
        self._ranking_results: dict[tuple[int, int], float] = {}
        self._ranking_centroids: list[tuple[int, int]] = []
        self._ranking_group_sizes: list[int] = []
        self._ranking_map: OccupancyGrid | None = None
        self._ranking_coverage: float = 0.0
        self._ranking_start_time: float | None = None

        # ── RL state ──────────────────────────────────────────
        self._step_count = 0
        self._last_features: np.ndarray | None = None
        self._last_coverage: float = 0.0
        self._last_time: float = time.time()

        # ── Networks ──────────────────────────────────────────
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._q_net = QNetwork().to(self._device)
        self._target_net = QNetwork().to(self._device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(
            self._q_net.parameters(), lr=LR,
        )
        self._replay = ReplayBuffer()
        self._episode_transitions: list[Transition] = []

        self._epsilon = EPSILON_START
        self._load_state()

        # ── ROS2 interfaces ───────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, 10,
        )
        self._nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
        )
        self._planner_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose',
        )
        self._viz_pub = self.create_publisher(
            MarkerArray, '/frontiers_viz', 10,
        )
        self._cmd_vel_pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10,
        )
        self._last_goal_time: float = time.time()

        self.create_timer(1.0, self._supervise_ranking)
        self.create_timer(5.0, self._supervise_goal)
        self.create_timer(2.0, self._check_idle)
        self.create_timer(2.0, self._watch_episode_file)

        self.get_logger().info(
            f'RLExplorer ready – e={self._epsilon:.3f}, '
            f'replay={len(self._replay)}, device={self._device}'
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if MODEL_PATH.exists():
            self._q_net.load_state_dict(torch.load(
                MODEL_PATH, weights_only=True,
                map_location=self._device,
            ))
            self._target_net.load_state_dict(
                self._q_net.state_dict(),
            )
            self.get_logger().info(
                f'[rl] Loaded model from {MODEL_PATH}'
            )
        self._replay.load(REPLAY_PATH)
        if META_PATH.exists():
            with open(META_PATH, 'rb') as f:
                meta = pickle.load(f)
            self._epsilon = meta.get('epsilon', EPSILON_START)
            self._step_count = meta.get('step_count', 0)
            self.get_logger().info(
                f'[rl] Loaded meta: e={self._epsilon:.3f}, '
                f'steps={self._step_count}, '
                f'replay={len(self._replay)}'
            )

    def _save_state(self) -> None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self._q_net.state_dict(), MODEL_PATH)
        self._replay.save(REPLAY_PATH)
        meta = {
            'epsilon': self._epsilon,
            'step_count': self._step_count,
        }
        with open(META_PATH, 'wb') as f:
            pickle.dump(meta, f)

    # ------------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._coverage_done:
            return

        current_coverage = self._get_coverage(msg)
        if current_coverage is None:
            current_coverage = 0.0

        # Always publish viz
        robot_pos = self._get_robot_pos()
        robot_grid = (
            world_to_grid(robot_pos[0], robot_pos[1], msg)
            if robot_pos is not None else None
        )
        centroids, stats = centroids_from_map_debug(
            msg, robot_grid_pos=robot_grid,
        )
        self._viz_pub.publish(
            build_frontier_markers(
                stats['frontiers'],
                self._current_target_cell,
                msg,
                stamp=self.get_clock().now().to_msg(),
                centroids=centroids,
                goal_color=(0.0, 1.0, 0.0),
            )
        )

        if self._goal_active or self._ranking_in_progress:
            return
        if robot_pos is None:
            return
        if not centroids:
            self.get_logger().info(
                '[rl] No frontier centroids – waiting',
                throttle_duration_sec=5.0,
            )
            return

        self._start_ranking(
            centroids,
            stats['centroid_group_sizes'],
            msg,
            current_coverage,
        )

    # ------------------------------------------------------------------
    # Nav2 path-length ranking
    # ------------------------------------------------------------------

    def _start_ranking(
        self,
        centroids: list[tuple[int, int]],
        group_sizes: list[int],
        map_msg: OccupancyGrid,
        coverage: float,
    ) -> None:
        if not self._planner_client.server_is_ready():
            self.get_logger().warning(
                '[rank] planner not ready – skipping',
                throttle_duration_sec=5.0,
            )
            return

        self._ranking_in_progress = True
        self._ranking_map = map_msg
        self._ranking_centroids = list(centroids)
        self._ranking_group_sizes = list(group_sizes)
        self._ranking_coverage = coverage
        self._ranking_results = {}
        self._ranking_pending = len(centroids)
        self._ranking_start_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )

        for cell in centroids:
            wx, wy = grid_to_world(cell, map_msg)
            goal = ComputePathToPose.Goal()
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            goal.goal = pose
            goal.use_start = False
            future = self._planner_client.send_goal_async(goal)
            future.add_done_callback(
                lambda f, c=cell: self._on_rank_goal_response(f, c)
            )

    def _on_rank_goal_response(
        self, future: Future, cell: tuple[int, int],
    ) -> None:
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            self._record_rank(cell, None)
            return
        handle.get_result_async().add_done_callback(
            lambda f, c=cell: self._on_rank_result(f, c)
        )

    def _on_rank_result(
        self, future: Future, cell: tuple[int, int],
    ) -> None:
        wrapper = future.result()
        length: float | None = None
        if wrapper.status == GoalStatus.STATUS_SUCCEEDED:
            path = wrapper.result.path
            if (wrapper.result.error_code == 0
                    and len(path.poses) >= 2):
                length = 0.0
                prev = path.poses[0].pose.position
                for ps in path.poses[1:]:
                    cur = ps.pose.position
                    length += math.hypot(
                        cur.x - prev.x, cur.y - prev.y,
                    )
                    prev = cur
        self._record_rank(cell, length)

    def _record_rank(
        self, cell: tuple[int, int], length: float | None,
    ) -> None:
        if length is not None:
            self._ranking_results[cell] = length
        self._ranking_pending -= 1
        if self._ranking_pending <= 0:
            self._finish_ranking()

    def _finish_ranking(self) -> None:
        self._ranking_in_progress = False
        self._ranking_start_time = None

        # Build feature vectors for reachable centroids
        candidates = []
        for i, cell in enumerate(self._ranking_centroids):
            if cell not in self._ranking_results:
                continue
            features = self._make_features(
                self._ranking_group_sizes[i],
                self._ranking_results[cell],
                self._ranking_coverage,
            )
            candidates.append((cell, features))

        if not candidates:
            self.get_logger().warning(
                '[rank] No reachable centroids'
            )
            return

        # Record reward for previous action
        now = time.time()
        if self._last_features is not None:
            dt = max(now - self._last_time, 1.0)
            delta_cov = (
                self._ranking_coverage - self._last_coverage
            )
            reward = delta_cov / dt * 100.0
            done = (
                self._ranking_coverage
                >= self._coverage_threshold
            )
            next_feats = [f for _, f in candidates]
            self._episode_transitions.append(Transition(
                self._last_features,
                self._last_features,
                reward,
                next_feats,
                done,
            ))
            self.get_logger().info(
                f'[rl] reward={reward:.4f}  '
                f'dcov={delta_cov:.4f}  dt={dt:.1f}s'
            )
            self._update_network()

        # Select action
        all_features = np.stack([f for _, f in candidates])
        action_idx = self._select_action(all_features)
        chosen_cell, chosen_features = candidates[action_idx]

        self._current_map = self._ranking_map
        self._current_target_cell = chosen_cell
        self._last_features = chosen_features
        self._last_coverage = self._ranking_coverage
        self._last_time = now

        wx, wy = grid_to_world(
            chosen_cell, self._ranking_map,
        )
        path_len = self._ranking_results[chosen_cell]
        idx = self._ranking_centroids.index(chosen_cell)
        gs = self._ranking_group_sizes[idx]
        self.get_logger().info(
            f'[rl] act={action_idx}/{len(candidates)} '
            f'-> ({wx:.2f},{wy:.2f}) '
            f'grp={gs} path={path_len:.2f}m '
            f'e={self._epsilon:.3f} '
            f'buf={len(self._replay)} '
            f'step={self._step_count}'
        )
        self._send_goal(wx, wy)

    def _supervise_ranking(self) -> None:
        """Cancel stalled ranking after timeout."""
        if (not self._ranking_in_progress
                or self._ranking_start_time is None):
            return
        elapsed = (
            self.get_clock().now().nanoseconds * 1e-9
            - self._ranking_start_time
        )
        if elapsed < RANKING_TIMEOUT_S:
            return
        self.get_logger().warning(
            f'[rank] TIMEOUT {elapsed:.0f}s – '
            f'using {len(self._ranking_results)} results'
        )
        self._ranking_pending = 0
        self._finish_ranking()

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _make_features(
        self,
        group_size: int,
        path_length: float,
        coverage: float,
    ) -> np.ndarray:
        """Build normalised feature vector for one frontier."""
        return np.array([
            group_size / 100.0,
            path_length / 10.0,
            coverage,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # RL core
    # ------------------------------------------------------------------

    def _select_action(self, candidates: np.ndarray) -> int:
        """Epsilon-greedy over pointwise Q-values."""
        if random.random() < self._epsilon:
            return random.randrange(len(candidates))
        with torch.no_grad():
            t = torch.FloatTensor(candidates).to(self._device)
            q_vals = self._q_net(t).squeeze(-1)
            return int(q_vals.argmax().item())

    def _update_network(self) -> None:
        """One gradient step on a mini-batch."""
        if len(self._replay) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
            return

        batch = self._replay.sample(BATCH_SIZE)

        states = torch.FloatTensor(
            np.stack([t.action_features for t in batch])
        ).to(self._device)
        rewards = torch.FloatTensor(
            [t.reward for t in batch]
        ).to(self._device)
        dones = torch.FloatTensor(
            [float(t.done) for t in batch]
        ).to(self._device)

        q_current = self._q_net(states).squeeze(-1)

        with torch.no_grad():
            q_next = torch.zeros(
                BATCH_SIZE, device=self._device,
            )
            for i, t in enumerate(batch):
                if t.done or not t.next_candidates:
                    continue
                nc = torch.FloatTensor(
                    np.stack(t.next_candidates)
                ).to(self._device)
                q_next[i] = (
                    self._target_net(nc).squeeze(-1).max()
                )
            q_target = (
                rewards + GAMMA * q_next * (1.0 - dones)
            )

        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._step_count += 1
        self._epsilon = max(
            EPSILON_END, self._epsilon * EPSILON_DECAY,
        )

        if self._step_count % TARGET_UPDATE_INTERVAL == 0:
            self._target_net.load_state_dict(
                self._q_net.state_dict(),
            )
            self.get_logger().info(
                f'[rl] Target sync step={self._step_count} '
                f'loss={loss.item():.4f} '
                f'e={self._epsilon:.3f}'
            )

        if self._step_count % 100 == 0:
            self._save_state()
            self.get_logger().info(
                f'[rl] State saved (step {self._step_count})'
            )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _send_goal(self, x: float, y: float) -> None:
        """Send a NavigateToPose goal to Nav2."""
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('[goal] Nav2 not reachable')
            return

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        q = tf_transformations.quaternion_from_euler(0, 0, 0.0)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        goal.pose = pose

        self._goal_active = True
        self._goal_dispatch_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )
        self._last_goal_time = time.time()
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future: Future) -> None:
        """Handle Nav2 goal acceptance."""
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            self.get_logger().warning('[goal] REJECTED')
            self._goal_active = False
            self._goal_handle = None
            return
        self._goal_handle = handle
        self.get_logger().info('[goal] ACCEPTED')
        handle.get_result_async().add_done_callback(
            self._on_goal_result,
        )

    def _on_goal_result(self, future: Future) -> None:
        """Handle Nav2 goal completion."""
        status = future.result().status
        name = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'status={status}')
        self.get_logger().info(f'[goal] {name}')
        self._goal_active = False
        self._goal_handle = None
        self._current_target_cell = None

    def _supervise_goal(self) -> None:
        """Cancel goals stuck in Nav2 recovery loops."""
        if (not self._goal_active
                or self._goal_dispatch_time is None):
            return
        elapsed = (
            self.get_clock().now().nanoseconds * 1e-9
            - self._goal_dispatch_time
        )
        if elapsed < GOAL_TIMEOUT_S:
            return
        self.get_logger().warning(
            f'[goal] TIMEOUT {elapsed:.0f}s – cancelling'
        )
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._goal_active = False
        self._goal_handle = None
        self._current_target_cell = None

    def _check_idle(self) -> None:
        """Spin in place when idle too long to generate new map data."""
        if self._coverage_done or self._goal_active:
            return
        idle = time.time() - self._last_goal_time
        if idle < IDLE_SPIN_S:
            return
        self.get_logger().warning(
            f'[idle] No goal for {idle:.0f}s – spinning'
        )
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.angular.z = 1.5
        self._cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_coverage(
        self, msg: OccupancyGrid,
    ) -> float | None:
        """Return current coverage or None if unavailable."""
        if self._expected_cells <= 0:
            return None
        coverage, known = check_coverage(
            msg, self._expected_cells,
        )
        self.get_logger().info(
            f'[coverage] {coverage:.1%} '
            f'({known}/{self._expected_cells})',
            throttle_duration_sec=5.0,
        )
        return coverage

    def _get_robot_pos(self) -> tuple[float, float] | None:
        """Look up current robot position from TF."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
            )
            return (
                t.transform.translation.x,
                t.transform.translation.y,
            )
        except Exception:
            return None

    def _watch_episode_file(self) -> None:
        """React to episode-done file written by EpisodeManager."""
        if self._coverage_done or not self._episode_file:
            return
        ep = Path(self._episode_file)
        if not ep.exists():
            return
        reason = ep.read_text().strip()
        self._coverage_done = True
        if 'stuck' in reason or reason == 'timeout':
            n = len(self._episode_transitions)
            self._episode_transitions.clear()
            self.get_logger().info(
                f'[rl] Episode aborted ({reason}) – discarded {n} transitions'
            )
        else:
            self._commit_episode()
            self._save_state()
            self.get_logger().info(
                f'[rl] Episode succeeded ({reason}) – state saved'
            )

    def _commit_episode(self) -> None:
        """Move episode transitions into replay buffer and train."""
        n = len(self._episode_transitions)
        for t in self._episode_transitions:
            self._replay.push(t)
        self._episode_transitions.clear()
        self.get_logger().info(
            f'[rl] Committed {n} transitions to replay buffer'
        )
        for _ in range(min(n, 10)):
            self._update_network()


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = RLExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
