"""ROS-free core of the pointwise-DQN RL explorer.

Holds the learning logic that does not depend on rclpy/tf2/Nav2/Gazebo:
hyper-parameters, feature construction, the epsilon-greedy policy and its decay,
the reward formula, the replay buffer with disk persistence, the Q-network and
the persist-directory resolution. rl_agent.py imports everything from here, so
the node behaves as before while this logic stays unit-testable without ROS
(torch is imported lazily – only the network/policy need it).
"""
from __future__ import annotations

import collections
import os
from pathlib import Path
import pickle
import random

import numpy as np

FEATURE_DIM = 3             # [group_size_norm, path_length_norm, coverage]

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


def make_q_network():
    """Build the pointwise Q-network FEATURE_DIM->64->32->1 as an nn.Module."""
    torch, nn = _require_torch()

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
