"""RL-based exploration node.

Uses a Deep Q-Network (DQN) to select which frontier centroid to navigate to.
The state is the list of centroid positions relative to the robot (padded to
MAX_CENTROIDS entries). The action is the index of the chosen centroid.
The reward is the map coverage gained after reaching the target.

Requirements:
    pip install torch   (PyTorch, CPU build is sufficient)
"""
import collections
import random
from pathlib import Path

import numpy as np
import rclpy
import tf_transformations
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener

from tb3_slam.coverage_monitor import (
    DEFAULT_COVERAGE_THRESHOLD,
    check_coverage,
    load_expected_cells,
)
from tb3_slam.frontier_utils import centroids_from_map, grid_to_world, world_to_grid

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
MAX_CENTROIDS = 10          # action space size (extra slots are masked)
STATE_DIM = MAX_CENTROIDS * 2   # (Δx, Δy) per centroid slot

REPLAY_CAPACITY = 10_000
BATCH_SIZE = 64
GAMMA = 0.99                # discount factor
LR = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995       # multiplied after each action

TARGET_UPDATE_INTERVAL = 50  # steps between target-network sync
MIN_REPLAY_SIZE = 200        # start learning after this many transitions

MODEL_PATH = Path('/tmp/rl_explorer_model.pt')


# ─────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """Fully-connected Q-network: state → Q-values for each centroid slot."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, MAX_CENTROIDS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────

Transition = collections.namedtuple(
    'Transition', ['state', 'action', 'reward', 'next_state', 'done']
)


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self._buf: collections.deque[Transition] = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────

class RLExplorer(Node):
    """Navigate to frontier centroids selected by a DQN policy."""

    def __init__(self) -> None:
        super().__init__('rl_explorer')

        self.declare_parameter('coverage_threshold', DEFAULT_COVERAGE_THRESHOLD)
        self.declare_parameter('episode_file', '')

        self._coverage_threshold: float = self.get_parameter('coverage_threshold').value
        self._episode_file: str = self.get_parameter('episode_file').value
        self._coverage_done = False

        # ── Coverage tracking ──────────────────────────────────────
        expected = load_expected_cells()
        if expected is None:
            self.get_logger().warning('[coverage] No world metadata – coverage monitoring disabled')
            self._expected_cells = 0
        else:
            self._expected_cells = expected
            self.get_logger().info(
                f'[coverage] Expected cells: {expected} (threshold {self._coverage_threshold:.0%})'
            )

        # ── Navigation state ───────────────────────────────────────
        self._goal_active = False
        self._current_map: OccupancyGrid | None = None

        # ── RL state ───────────────────────────────────────────────
        self._epsilon = EPSILON_START
        self._step_count = 0
        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_coverage: float = 0.0

        # ── Networks ───────────────────────────────────────────────
        self._q_net = QNetwork()
        self._target_net = QNetwork()
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(self._q_net.parameters(), lr=LR)
        self._replay = ReplayBuffer()

        if MODEL_PATH.exists():
            self._q_net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            self._target_net.load_state_dict(self._q_net.state_dict())
            self.get_logger().info(f'[rl] Loaded model from {MODEL_PATH}')

        # ── ROS2 interfaces ────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self._nav_client: ActionClient = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('RLExplorer ready – waiting for /map ...')

    # ------------------------------------------------------------------
    # Map callback  (main RL loop)
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if self._coverage_done:
            return

        current_coverage = self._get_coverage(msg)
        if current_coverage is None:
            return

        if current_coverage >= self._coverage_threshold:
            self._finish(current_coverage)
            return

        if self._goal_active:
            return

        # ── Record reward for the previous transition ──────────────
        robot_pos = self._get_robot_pos()
        robot_grid_pos = world_to_grid(robot_pos[0], robot_pos[1], msg) if robot_pos is not None else None
        centroids = centroids_from_map(msg, robot_grid_pos=robot_grid_pos)
        if robot_pos is None:
            self.get_logger().warning('[rl] Could not get robot position – skipping step')
            return

        next_state = self._build_state(centroids, robot_pos)

        if self._last_state is not None:
            reward = current_coverage - self._last_coverage
            done = current_coverage >= self._coverage_threshold
            self._replay.push(self._last_state, self._last_action, reward, next_state, done)
            self.get_logger().debug(f'[rl] reward={reward:.4f}  ε={self._epsilon:.3f}')
            self._update_network()

        # ── Select and execute next action ─────────────────────────
        if not centroids:
            self.get_logger().info('[rl] No frontier centroids – waiting for new frontiers')
            return

        n_valid = min(len(centroids), MAX_CENTROIDS)
        action = self._select_action(next_state, n_valid)
        target_cell = centroids[action]
        self._current_map = msg

        x, y = grid_to_world(target_cell, msg)
        self.get_logger().info(
            f'[rl] action={action}  → ({x:.2f}, {y:.2f})'
            f'  ε={self._epsilon:.3f}  replay={len(self._replay)}'
        )
        self._send_goal(x, y)

        # Store current step info for the next reward computation
        self._last_state = next_state
        self._last_action = action
        self._last_coverage = current_coverage
        self._goal_active = True

    # ------------------------------------------------------------------
    # RL helpers
    # ------------------------------------------------------------------

    def _build_state(
        self, centroids: list[tuple[int, int]], robot_pos: tuple[float, float]
    ) -> np.ndarray:
        """Build a fixed-size state vector of centroid positions relative to the robot.

        Each of the MAX_CENTROIDS slots holds (Δx, Δy).  Unused slots are zeroed.
        Centroids are sorted by distance (nearest first) so that slot 0 is always
        the most immediately reachable frontier.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        if self._current_map is None and not centroids:
            return state

        rx, ry = robot_pos

        # Sort by distance to robot (use the last known map for conversion)
        map_msg = self._current_map
        if map_msg is None:
            return state

        def dist(cell: tuple[int, int]) -> float:
            cx, cy = grid_to_world(cell, map_msg)
            return (cx - rx) ** 2 + (cy - ry) ** 2

        sorted_centroids = sorted(centroids, key=dist)

        for i, cell in enumerate(sorted_centroids[:MAX_CENTROIDS]):
            cx, cy = grid_to_world(cell, map_msg)
            state[i * 2] = cx - rx
            state[i * 2 + 1] = cy - ry

        return state

    def _select_action(self, state: np.ndarray, n_valid: int) -> int:
        """Epsilon-greedy action selection over the n_valid filled centroid slots."""
        if random.random() < self._epsilon:
            return random.randrange(n_valid)

        with torch.no_grad():
            q_values: torch.Tensor = self._q_net(torch.FloatTensor(state).unsqueeze(0))[0]
            # Mask slots beyond the available centroids
            if n_valid < MAX_CENTROIDS:
                q_values[n_valid:] = float('-inf')
            return int(q_values.argmax().item())

    def _update_network(self) -> None:
        """Sample a mini-batch from the replay buffer and perform one gradient step."""
        if len(self._replay) < MIN_REPLAY_SIZE:
            return

        batch = self._replay.sample(BATCH_SIZE)
        states = torch.FloatTensor(np.stack([t.state for t in batch]))
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch]))
        dones = torch.FloatTensor([float(t.done) for t in batch])

        # Current Q-values for the taken actions
        q_current = self._q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            best_actions = self._q_net(next_states).argmax(dim=1)
            q_next = self._target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._step_count += 1
        self._epsilon = max(EPSILON_END, self._epsilon * EPSILON_DECAY)

        if self._step_count % TARGET_UPDATE_INTERVAL == 0:
            self._target_net.load_state_dict(self._q_net.state_dict())
            self.get_logger().info(
                f'[rl] Target network synced  (step {self._step_count}, loss {loss.item():.4f})'
            )

        if self._step_count % 100 == 0:
            torch.save(self._q_net.state_dict(), MODEL_PATH)
            self.get_logger().info(f'[rl] Model saved to {MODEL_PATH}')

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _send_goal(self, x: float, y: float, theta: float = 0.0) -> None:
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('[goal] Nav2 action server not reachable')
            self._goal_active = False
            return

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        goal.pose = pose

        future: Future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future: Future) -> None:
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            self.get_logger().warning('[goal] REJECTED by Nav2')
            self._goal_active = False
            return
        self.get_logger().info('[goal] ACCEPTED – navigating ...')
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future: Future) -> None:
        status: int = future.result().status
        status_name = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'status={status}')
        self.get_logger().info(f'[goal] {status_name}')
        self._goal_active = False
        # The next _map_callback will compute the reward and select a new action.

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_coverage(self, msg: OccupancyGrid) -> float | None:
        if self._expected_cells <= 0:
            return None
        coverage, known = check_coverage(msg, self._expected_cells)
        self.get_logger().info(
            f'[coverage] {coverage:.1%}  ({known}/{self._expected_cells} cells)',
            throttle_duration_sec=5.0,
        )
        return coverage

    def _get_robot_pos(self) -> tuple[float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            self.get_logger().warning(f'[tf] Could not get robot position: {e}')
            return None

    def _finish(self, coverage: float) -> None:
        self.get_logger().info(
            f'[coverage] TARGET REACHED {coverage:.1%} >= {self._coverage_threshold:.0%}'
        )
        self._coverage_done = True
        torch.save(self._q_net.state_dict(), MODEL_PATH)
        self.get_logger().info(f'[rl] Final model saved to {MODEL_PATH}')
        if self._episode_file:
            Path(self._episode_file).write_text(f'coverage={coverage:.1%}')


def main(args=None):
    rclpy.init(args=args)
    node = RLExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
