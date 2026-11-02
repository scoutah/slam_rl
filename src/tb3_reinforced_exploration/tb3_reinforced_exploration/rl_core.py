"""ROS-free learning core of the pointwise-DQN explorer.

Features, epsilon-greedy policy + decay, reward, replay buffer with disk
persistence, the Q-network and persist-dir resolution — everything that does not
need rclpy/tf2/Nav2, so it stays unit-testable (torch is imported lazily).
``rl_agent`` imports from here.
"""
from __future__ import annotations

import collections
import os
from pathlib import Path
import pickle
import random

import numpy as np

FEATURE_DIM = 3             # [group_size_norm, path_length_norm, coverage]

# Multi-agent (team-aware) per-frontier features for the centralised explorer.
# [group_size_norm, path_norm, coverage, teammate_dist_norm, contested]. All
# normalisers are either already in [0, 1] (coverage, contested) or divide a
# distance by the *map's own diagonal*, which is read from the live grid — so
# the same relative frontier yields the same features on 8x8 and 12x12. That
# size-invariance is what lets the 8x8-trained policy run unchanged on 12x12.
TEAM_FEATURE_DIM = 5

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.98

TARGET_UPDATE_INTERVAL = 50
MIN_REPLAY_SIZE = 50

# Size-invariant normalisation (relative, not absolute coords) so an 8x8-trained
# policy transfers to the unseen 12x12 world. Single source of truth.
GROUP_SIZE_NORM = 100.0
PATH_LENGTH_NORM = 10.0


def make_features(group_size: float,
                  path_length: float,
                  coverage: float) -> np.ndarray:
    """Normalised per-frontier features: [group_size/100, path_length/10, coverage]."""
    return np.array([
        group_size / GROUP_SIZE_NORM,
        path_length / PATH_LENGTH_NORM,
        coverage,
    ], dtype=np.float32)


CONTESTED_RADIUS_NORM = 0.1   # teammate goal within 10 % of the diagonal = contested


def make_team_features(group_size: float,
                       path_length: float,
                       coverage: float,
                       teammate_dist: float,
                       map_diagonal: float,
                       contested: bool) -> np.ndarray:
    """Size-invariant, team-aware per-frontier features for the team explorer.

    ``path_length`` (this robot's cost to the frontier) and ``teammate_dist``
    (the other robot's straight-line distance to the same frontier) are both
    divided by ``map_diagonal`` — the diagonal of the live merged grid in metres
    — so they stay in roughly [0, 1] regardless of world size. No absolute world
    coordinate ever enters the vector. ``contested`` is the assignment-status
    flag (teammate already heading for this frontier).

    Returns ``[group_size_norm, path_norm, coverage, teammate_dist_norm, contested]``.
    """
    diag = map_diagonal if map_diagonal > 1e-6 else 1.0
    return np.array([
        group_size / GROUP_SIZE_NORM,
        path_length / diag,
        coverage,
        teammate_dist / diag,
        1.0 if contested else 0.0,
    ], dtype=np.float32)


def compute_reward(prev_coverage: float,
                   next_coverage: float,
                   dt_seconds: float) -> float:
    """Coverage gained per second x100; dt floored at 1 s (matches the node)."""
    dt = max(dt_seconds, 1.0)
    return (next_coverage - prev_coverage) / dt * 100.0


def decay_epsilon(epsilon: float) -> float:
    """One multiplicative epsilon decay step, clamped at EPSILON_END."""
    return max(EPSILON_END, epsilon * EPSILON_DECAY)


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
        """Add a transition (oldest evicted once capacity is reached)."""
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch without replacement."""
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def save(self, path: Path) -> None:
        """Persist the buffer to disk, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self._buf), f)

    def load(self, path: Path) -> None:
        """Load a buffer from disk; a missing file is a silent no-op."""
        if not path.exists():
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._buf = collections.deque(data, maxlen=self._buf.maxlen)


def resolve_persist_dir(environ: dict | None = None,
                        cwd: Path | None = None) -> Path:
    """Locate the model/replay/meta dir.

    SLAM_RL_DATA_DIR wins; else <ws>/rl_data via COLCON_PREFIX_PATH; else
    ./rl_data. environ/cwd injectable for tests.
    """
    env = os.environ if environ is None else environ

    env_dir = env.get('SLAM_RL_DATA_DIR')
    if env_dir:
        return Path(env_dir).expanduser()

    prefix_path = env.get('COLCON_PREFIX_PATH', '')
    if prefix_path:
        first_prefix = prefix_path.split(os.pathsep)[0]
        ws_root = Path(first_prefix).parent
        return ws_root / 'rl_data'

    base = Path.cwd() if cwd is None else cwd
    return base / 'rl_data'


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            'PyTorch is required for the RL explorer.  Install it with:\n'
            '    pip install torch'
        ) from exc
    return torch, nn


def make_q_network(feature_dim: int = FEATURE_DIM):
    """Build the pointwise Q-network feature_dim->64->32->1 as an nn.Module."""
    torch, nn = _require_torch()

    class QNetwork(nn.Module):
        """Map per-frontier features to a scalar Q-value."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            """Return Q-values. x: (batch, features) -> (batch, 1)."""
            return self.net(x)

    return QNetwork()


def select_action(candidates: np.ndarray,
                  epsilon: float,
                  q_net=None,
                  device=None,
                  rng: random.Random | None = None) -> int:
    """Epsilon-greedy index over pointwise Q-values; rng injectable for tests."""
    n = len(candidates)
    if n == 0:
        raise ValueError('select_action called with no candidates')
    r = random.random() if rng is None else rng.random()
    if r < epsilon or q_net is None:
        upper = n - 1
        return (random.randint(0, upper) if rng is None
                else rng.randint(0, upper))

    torch, _ = _require_torch()
    with torch.no_grad():
        t = torch.FloatTensor(np.asarray(candidates))
        if device is not None:
            t = t.to(device)
        q_vals = q_net(t).squeeze(-1)
        return int(q_vals.argmax().item())


class TeamLearner:
    """CTDE learner for the centralised multi-robot explorer.

    Centralised training, decentralised execution: a *single* Q-network scores
    frontiers for either robot (parameter sharing — the burgers are homogeneous
    and interchangeable, so one shared set of weights halves the parameters and
    lets both robots' experience train the same policy), and *one* shared replay
    buffer pools transitions from both agents. Because the per-robot features are
    team-aware (teammate distance + contested flag), the shared net still learns
    to coordinate rather than collide, while execution stays decentralised: at
    decision time each robot is scored independently on its own candidate set.

    ROS-free so it is unit-testable; torch is imported lazily. The ROS node owns
    pose/map/Nav2 I/O and feeds (features, reward, next_candidates) in here.
    """

    def __init__(self,
                 feature_dim: int = TEAM_FEATURE_DIM,
                 persist_dir: Path | None = None,
                 device=None) -> None:
        torch, _ = _require_torch()
        import torch.optim as optim

        self.feature_dim = feature_dim
        self.persist_dir = (persist_dir if persist_dir is not None
                            else resolve_persist_dir())
        self.device = (device if device is not None
                       else torch.device('cuda' if torch.cuda.is_available()
                                         else 'cpu'))

        self.q_net = make_q_network(feature_dim).to(self.device)
        self.target_net = make_q_network(feature_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)

        self.replay = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.step_count = 0

    # -- paths ---------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self.persist_dir / 'model.pt'

    @property
    def replay_path(self) -> Path:
        return self.persist_dir / 'replay.pkl'

    @property
    def meta_path(self) -> Path:
        return self.persist_dir / 'meta.pkl'

    # -- persistence ---------------------------------------------------

    def load(self) -> None:
        """Restore network, replay buffer and meta from the persist dir.

        A feature-dim mismatch (e.g. an old 3-feature model) is skipped rather
        than crashing, so switching feature sets starts fresh instead of failing.
        """
        torch, _ = _require_torch()
        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, weights_only=True,
                                   map_location=self.device)
                self.q_net.load_state_dict(state)
                self.target_net.load_state_dict(self.q_net.state_dict())
            except (RuntimeError, KeyError):
                pass  # incompatible checkpoint – keep the fresh network
        self.replay.load(self.replay_path)
        if self.meta_path.exists():
            with open(self.meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.epsilon = meta.get('epsilon', EPSILON_START)
            self.step_count = meta.get('step_count', 0)

    def save(self) -> None:
        """Persist network, replay buffer and meta to the persist dir."""
        torch, _ = _require_torch()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), self.model_path)
        self.replay.save(self.replay_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump({'epsilon': self.epsilon,
                         'step_count': self.step_count}, f)

    # -- acting --------------------------------------------------------

    def select_index(self, candidates: np.ndarray,
                     rng: random.Random | None = None) -> int:
        """Epsilon-greedy frontier index for one robot's candidate features."""
        return select_action(candidates, self.epsilon, self.q_net,
                             self.device, rng)

    # -- learning ------------------------------------------------------

    def push(self, t: Transition) -> None:
        """Add one agent's transition to the shared replay buffer."""
        self.replay.push(t)

    def train_step(self) -> float | None:
        """One gradient step on a shared mini-batch; returns loss or None.

        Returns None (no update) until the buffer holds enough samples. Steps the
        epsilon schedule and syncs the target net on the usual interval.
        """
        torch, nn = _require_torch()
        if len(self.replay) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
            return None

        batch = self.replay.sample(BATCH_SIZE)
        states = torch.FloatTensor(
            np.stack([t.action_features for t in batch])).to(self.device)
        rewards = torch.FloatTensor(
            [t.reward for t in batch]).to(self.device)
        dones = torch.FloatTensor(
            [float(t.done) for t in batch]).to(self.device)

        q_current = self.q_net(states).squeeze(-1)
        with torch.no_grad():
            q_next = torch.zeros(BATCH_SIZE, device=self.device)
            for i, t in enumerate(batch):
                if t.done or not t.next_candidates:
                    continue
                nc = torch.FloatTensor(
                    np.stack(t.next_candidates)).to(self.device)
                q_next[i] = self.target_net(nc).squeeze(-1).max()
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = decay_epsilon(self.epsilon)
        if self.step_count % TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())
