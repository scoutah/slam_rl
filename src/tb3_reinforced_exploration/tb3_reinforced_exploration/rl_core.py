"""ROS-free learning core of the RL explorers (multi-robot QMIX + legacy DQN).

Everything that needs no rclpy/tf2/Nav2, so it stays unit-testable; torch is
imported lazily via ``_require_torch``. ``rl_agent`` uses the legacy part.
"""
from __future__ import annotations

import collections
import csv
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np

FEATURE_DIM = 3             # legacy rl_agent: [group_size_norm, path_length_norm, coverage]

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
# QMixLearner decays once per *episode* (end_episode), not per gradient step:
# 0.98 ** 150 ≈ 0.05, so ε reaches its floor after ≈150 training episodes.
# (The legacy single-robot rl_agent still decays per step.)
EPSILON_DECAY = 0.98

TARGET_UPDATE_INTERVAL = 50
MIN_REPLAY_SIZE = 50

# Training telemetry (QMixLearner): append-only CSV plus versioned weights.
SNAPSHOT_INTERVAL = 10      # episodes between weight snapshots
TRAINING_LOG_NAME = 'training_log.csv'
TRAINING_LOG_FIELDS = ('episode', 'step', 'loss', 'epsilon', 'reward', 'buffer')
SNAPSHOT_DIR_NAME = 'snapshots'

# Size-invariant normalisation (relative, not absolute coords) so an 8x8-trained
# policy transfers to the unseen 12x12 world. Single source of truth.
GROUP_SIZE_NORM = 100.0
PATH_LENGTH_NORM = 10.0     # legacy rl_agent only (absolute metres)

# Multi-robot (QMIX) dimensions. OBS_DIM: per-candidate local observation o_i,
# see make_local_observation. STATE_DIM: global state s, training only, see
# make_global_state.
OBS_DIM = 5
STATE_DIM = 4
FRONTIER_COUNT_NORM = 20.0  # frontier-group count that maps to 1.0 in s
MIXER_EMBED_DIM = 16
GRAD_CLIP_NORM = 10.0       # usual QMIX setting; keeps the hypernetworks stable

CONTESTED_RADIUS_NORM = 0.1   # intent within 10 % of the diagonal = taken

Point = tuple[float, float]


def make_features(group_size: float,
                  path_length: float,
                  coverage: float) -> np.ndarray:
    """Normalised per-frontier features: [group_size/100, path_length/10, coverage]."""
    return np.array([
        group_size / GROUP_SIZE_NORM,
        path_length / PATH_LENGTH_NORM,
        coverage,
    ], dtype=np.float32)


# ── Intent communication + local observation (Dec-POMDP o_i) ───────

def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def near_any(x: float, y: float, points: list[Point], radius: float) -> bool:
    """Return True if (x, y) lies within ``radius`` of any of ``points``."""
    return any(_dist((x, y), p) <= radius for p in points)


def filter_claimed(candidates: list, claimed: list[Point],
                   radius: float) -> list:
    """Drop candidates (objects with ``.x``/``.y``) near a received intent.

    Local rule each robot applies to the intents it has *received* (teammates'
    ongoing or just-announced goals), not a central allocator. It is what
    guarantees no frontier is double-booked within a decision cycle.
    """
    return [c for c in candidates if not near_any(c.x, c.y, claimed, radius)]


def nearest_intent_dist(point: Point, intents: list[Point]) -> float | None:
    """Distance from ``point`` to the closest received intent, None if none."""
    if not intents:
        return None
    return min(_dist(point, p) for p in intents)


def make_local_observation(group_size: float,
                           path_length: float,
                           intent_dist: float | None,
                           map_diagonal: float,
                           info_gain: float = 0.0) -> np.ndarray:
    """Local observation o_i of one candidate frontier for one robot.

    Dec-POMDP: built only from the robot's own perspective (its own path cost)
    plus the teammate's communicated intent (Tan et al.) — never the teammate's
    pose or global quantities such as team coverage. The frontier itself comes
    from the merged map, which counts as a shared, communicated world model.
    Every distance is divided by the live map diagonal, so the same relative
    geometry gives the same o_i on 8x8 and 12x12 (size invariance for transfer).

    ``info_gain`` is the estimated unseen area the frontier reveals as a
    fraction of the sensor disc (``tb3_slam.information_gain``), net of what the
    teammate's announced intent already covers — the same raw quantity the
    classical explorer uses; the net learns its own trade-off against the path.

    Returns ``[group_size_norm, path_norm, intent_dist_norm, has_intent,
    info_gain]``; ``intent_dist_norm`` is clipped to [0, 1] and is 1.0 without
    an intent, ``info_gain`` is clipped to [0, 1].
    """
    diag = map_diagonal if map_diagonal > 1e-6 else 1.0
    if intent_dist is None:
        intent_norm, has_intent = 1.0, 0.0
    else:
        intent_norm, has_intent = min(max(intent_dist / diag, 0.0), 1.0), 1.0
    return np.array([
        group_size / GROUP_SIZE_NORM,
        path_length / diag,
        intent_norm,
        has_intent,
        min(max(info_gain, 0.0), 1.0),
    ], dtype=np.float32)


def decentralised_round(order: list[str],
                        busy_intents: dict[str, Point],
                        candidates: dict[str, list],
                        observe,
                        act,
                        radius: float) -> tuple[dict, dict, dict]:
    """One decision cycle of decentralised execution with intent messages.

    ``order`` is the fixed communication order of the *idle* robots;
    ``busy_intents`` the ongoing goals announced at cycle start. Each idle
    robot receives the teammates' intents so far, drops candidates (objects
    with ``.x``/``.y``) within ``radius`` of them, builds its observation
    matrix via ``observe(name, cand, received) -> o_i row``, picks
    ``act(obs) -> index`` and announces the choice as its intent.

    Returns ``(choices, options, intents)``: chosen candidate per robot that
    picked one, its observation matrix + chosen index ``(obs, idx)``, and all
    intents at the end of the cycle.
    """
    intents = dict(busy_intents)
    choices: dict = {}
    options: dict = {}
    for name in order:
        received = [p for n, p in intents.items() if n != name]
        cands = filter_claimed(candidates.get(name, []), received, radius)
        if not cands:
            continue
        obs = np.stack([observe(name, c, received) for c in cands])
        idx = act(obs)
        choices[name] = cands[idx]
        options[name] = (obs, idx)
        intents[name] = (cands[idx].x, cands[idx].y)
    return choices, options, intents


def assemble_joint(agents: list[str],
                   actions: dict[str, np.ndarray],
                   options: dict[str, np.ndarray],
                   obs_dim: int = OBS_DIM
                   ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Stack per-robot action rows/option sets in fixed agent order.

    Returns ``(actions [n, obs_dim], mask [n], candidates)``; a robot without an
    action (absent, or idle with nothing to pick) gets zeros, mask 0 and an
    empty ``[0, obs_dim]`` candidate set.
    """
    n = len(agents)
    act = np.zeros((n, obs_dim), np.float32)
    mask = np.zeros(n, np.float32)
    cands = [np.zeros((0, obs_dim), np.float32) for _ in range(n)]
    for i, ns in enumerate(agents):
        if ns in actions:
            act[i], mask[i] = actions[ns], 1.0
            cands[i] = np.asarray(options[ns], np.float32).reshape(-1, obs_dim)
    return act, mask, cands


def make_global_state(coverage: float,
                      robot_positions: list[Point],
                      n_frontier_groups: int,
                      n_busy: int,
                      n_agents: int,
                      map_diagonal: float) -> np.ndarray:
    """Global state s for centralised training only (CTDE); never used to act.

    Conditions the QMIX mixer. Size-invariant:
    ``[coverage, inter_robot_dist/diag, min(groups/FRONTIER_COUNT_NORM, 1),
    busy_fraction]``. With fewer than two known robots the distance is 0 (the
    agent mask already marks the missing robot).
    """
    diag = map_diagonal if map_diagonal > 1e-6 else 1.0
    pairs = [_dist(a, b) for i, a in enumerate(robot_positions)
             for b in robot_positions[i + 1:]]
    inter = min(sum(pairs) / len(pairs) / diag, 1.0) if pairs else 0.0
    return np.array([
        min(max(coverage, 0.0), 1.0),
        inter,
        min(n_frontier_groups / FRONTIER_COUNT_NORM, 1.0),
        n_busy / n_agents if n_agents > 0 else 0.0,
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

# One joint decision event of the team (QMIX replay entry). actions:
# [n_agents, OBS_DIM] observation features of each agent's chosen/ongoing goal;
# mask: [n_agents] 1 = agent acted/held a goal; next_candidates: per agent an
# array [k, OBS_DIM] (k may be 0) of what it could pick at the next event.
JointTransition = collections.namedtuple(
    'JointTransition',
    ['actions', 'mask', 'state', 'reward', 'next_candidates', 'next_state',
     'next_mask', 'done'],
)


class ReplayBuffer:
    """Fixed-capacity replay buffer with disk persistence."""

    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, t: Transition | JointTransition) -> None:
        """Add a transition (oldest evicted once capacity is reached)."""
        self._buf.append(t)

    def sample(self, batch_size: int) -> list:
        """Sample a random mini-batch without replacement."""
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        """Drop all stored transitions."""
        self._buf.clear()

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


class JointEpisodeRecorder:
    """Turn the team's joint decision events into QMIX ``JointTransition`` s.

    A joint event is a decision cycle in which at least one robot received a
    new goal (semi-MDP step: the hierarchical action is a frontier and Nav2
    executes it, so steps have variable length). Each event stays pending until
    the next one (or episode end) closes it with the *shared* team reward
    ``compute_reward(prev_cov, next_cov, real elapsed seconds)``.
    """

    def __init__(self) -> None:
        # (actions, mask, state, coverage, time) of the open event, or None.
        self._pending: tuple | None = None

    @property
    def has_pending(self) -> bool:
        """Return True while an event awaits its reward."""
        return self._pending is not None

    def record_event(self,
                     actions: np.ndarray,
                     mask: np.ndarray,
                     candidates: list[np.ndarray],
                     state: np.ndarray,
                     coverage: float,
                     now: float) -> JointTransition | None:
        """Close the pending event with this one as successor; open this one.

        ``candidates[i]`` is what agent i could choose from at this event
        (idle: its candidate set; busy: its ongoing goal only; absent: empty).
        Returns the closed transition, or None for the episode's first event.
        """
        next_mask = np.array(
            [1.0 if m > 0 and len(c) > 0 else 0.0
             for m, c in zip(mask, candidates)], dtype=np.float32)
        out = self._close(coverage, now, list(candidates), state, next_mask,
                          done=False)
        self._pending = (np.asarray(actions, np.float32),
                         np.asarray(mask, np.float32),
                         np.asarray(state, np.float32), coverage, now)
        return out

    def close(self, coverage: float, now: float) -> JointTransition | None:
        """Close the pending event at episode end (done, nothing to bootstrap)."""
        if self._pending is None:
            return None
        n, obs_dim = self._pending[0].shape
        empty = [np.zeros((0, obs_dim), np.float32) for _ in range(n)]
        # next_state is never used when done; keep a valid shape anyway.
        out = self._close(coverage, now, empty,
                          np.zeros_like(self._pending[2]),
                          np.zeros(n, np.float32), done=True)
        self._pending = None
        return out

    def _close(self, coverage, now, next_candidates, next_state, next_mask,
               done) -> JointTransition | None:
        if self._pending is None:
            return None
        actions, mask, state, prev_cov, prev_t = self._pending
        reward = compute_reward(prev_cov, coverage, now - prev_t)
        return JointTransition(
            actions, mask, state, reward,
            [np.asarray(c, np.float32).reshape(-1, actions.shape[1])
             for c in next_candidates],
            np.asarray(next_state, np.float32), next_mask, bool(done),
        )


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


def make_mixer(n_agents: int,
               state_dim: int = STATE_DIM,
               embed_dim: int = MIXER_EMBED_DIM):
    """Build the QMIX mixing network as an nn.Module.

    ``forward(agent_qs [B, n], states [B, S], mask [B, n] | None) -> [B]``.
    Hypernetworks map the global state s to the mixing weights; ``abs()`` makes
    them non-negative and ELU is monotonic, so ∂Q_tot/∂Q_i ≥ 0 (QMIX's
    monotonicity, which makes per-agent argmax consistent with the joint
    argmax). Biases come from s, the output bias via a small MLP. A masked
    (absent) agent's Q_i is zeroed and therefore contributes nothing.
    """
    torch, nn = _require_torch()

    class QMixer(nn.Module):
        """Monotonic state-conditioned mixer Q_tot = f_s(Q_1, ..., Q_n)."""

        def __init__(self) -> None:
            super().__init__()
            self.n_agents = n_agents
            self.embed_dim = embed_dim
            self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
            self.hyper_b1 = nn.Linear(state_dim, embed_dim)
            self.hyper_w2 = nn.Linear(state_dim, embed_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

        def forward(self, agent_qs, states, mask=None):
            """Mix per-agent Q-values into Q_tot (one value per batch row)."""
            b = agent_qs.shape[0]
            if mask is not None:
                agent_qs = agent_qs * mask
            w1 = torch.abs(self.hyper_w1(states)).view(
                b, self.n_agents, self.embed_dim)
            b1 = self.hyper_b1(states).view(b, 1, self.embed_dim)
            hidden = nn.functional.elu(
                torch.bmm(agent_qs.view(b, 1, self.n_agents), w1) + b1)
            w2 = torch.abs(self.hyper_w2(states)).view(b, self.embed_dim, 1)
            b2 = self.hyper_b2(states).view(b, 1, 1)
            return (torch.bmm(hidden, w2) + b2).view(b)

    return QMixer()


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


class QMixLearner:
    """CTDE learner: shared agent Q-net for execution, QMIX mixer for training.

    * Parameter sharing: one pointwise agent net Q(o_i, a) for all homogeneous
      robots; the action a is a frontier, scored through its local observation
      features (hierarchical: the net picks the target, Nav2 drives there).
    * Decentralised execution: :meth:`act` sees only one robot's local
      observation matrix and uses only the agent net.
    * Centralised training: :meth:`train_step` mixes the agents' Q_i with the
      monotonic mixer conditioned on the global state s and regresses Q_tot on
      the shared team reward (QMIX value factorisation, i.e. credit assignment
      of the team reward to each robot's own Q_i).

    ROS-free; the node feeds ``JointTransition`` s (see JointEpisodeRecorder).
    """

    def __init__(self,
                 n_agents: int = 2,
                 obs_dim: int = OBS_DIM,
                 state_dim: int = STATE_DIM,
                 persist_dir: Path | None = None,
                 device=None) -> None:
        torch, _ = _require_torch()
        import torch.optim as optim

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.persist_dir = (persist_dir if persist_dir is not None
                            else resolve_persist_dir())
        self.device = (device if device is not None
                       else torch.device('cuda' if torch.cuda.is_available()
                                         else 'cpu'))

        self.agent_net = make_q_network(obs_dim).to(self.device)
        self.mixer = make_mixer(n_agents, state_dim).to(self.device)
        self.target_agent_net = make_q_network(obs_dim).to(self.device)
        self.target_mixer = make_mixer(n_agents, state_dim).to(self.device)
        self._sync_targets()
        self.target_agent_net.eval()
        self.target_mixer.eval()
        self.optimizer = optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=LR,
        )

        self.replay = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.step_count = 0
        self.episode_count = 0
        self.frozen = False

    def freeze(self) -> None:
        """Switch to evaluation: greedy policy, no learning, nothing written.

        Call after :meth:`load`. Sets epsilon to 0 and turns :meth:`push`,
        :meth:`train_step`, :meth:`end_episode` and :meth:`save` into no-ops,
        so the loaded model and its files stay untouched.
        """
        self.frozen = True
        self.epsilon = 0.0

    def _sync_targets(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    # -- paths (new names, so old 5-feature TeamLearner files are ignored) --

    @property
    def agent_path(self) -> Path:
        return self.persist_dir / 'agent.pt'

    @property
    def mixer_path(self) -> Path:
        return self.persist_dir / 'mixer.pt'

    @property
    def replay_path(self) -> Path:
        return self.persist_dir / 'joint_replay.pkl'

    @property
    def meta_path(self) -> Path:
        return self.persist_dir / 'qmix_meta.pkl'

    @property
    def log_path(self) -> Path:
        return self.persist_dir / TRAINING_LOG_NAME

    @property
    def snapshot_dir(self) -> Path:
        return self.persist_dir / SNAPSHOT_DIR_NAME

    # -- persistence ---------------------------------------------------

    def load(self) -> None:
        """Restore agent net, mixer, replay and meta from the persist dir.

        Agent net and mixer are loaded all-or-nothing: an incompatible or
        unreadable checkpoint (e.g. other dims) keeps both fresh instead of
        crashing. Replay entries with a different shape are discarded too.
        """
        torch, _ = _require_torch()
        if self.agent_path.exists() and self.mixer_path.exists():
            # Clone: load_state_dict copies matching tensors in place before
            # raising on a mismatch, so a plain reference would be corrupted.
            fresh = tuple({k: v.clone() for k, v in net.state_dict().items()}
                          for net in (self.agent_net, self.mixer))
            try:
                self.agent_net.load_state_dict(torch.load(
                    self.agent_path, weights_only=True,
                    map_location=self.device))
                self.mixer.load_state_dict(torch.load(
                    self.mixer_path, weights_only=True,
                    map_location=self.device))
            except (RuntimeError, KeyError, EOFError, pickle.UnpicklingError):
                self.agent_net.load_state_dict(fresh[0])
                self.mixer.load_state_dict(fresh[1])
            self._sync_targets()
        try:
            self.replay.load(self.replay_path)
        except (EOFError, pickle.UnpicklingError, AttributeError):
            self.replay.clear()
        if len(self.replay) and not self._replay_compatible():
            self.replay.clear()
        if self.meta_path.exists():
            with open(self.meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.epsilon = meta.get('epsilon', EPSILON_START)
            self.step_count = meta.get('step_count', 0)
            self.episode_count = meta.get('episode_count', 0)

    def _replay_compatible(self) -> bool:
        t = self.replay.sample(1)[0]
        return (isinstance(t, JointTransition)
                and np.shape(t.actions) == (self.n_agents, self.obs_dim)
                and np.shape(t.state) == (self.state_dim,))

    def save(self) -> None:
        """Persist agent net, mixer, replay and meta to the persist dir."""
        if self.frozen:
            return
        torch, _ = _require_torch()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent_net.state_dict(), self.agent_path)
        torch.save(self.mixer.state_dict(), self.mixer_path)
        self.replay.save(self.replay_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump({'epsilon': self.epsilon,
                         'step_count': self.step_count,
                         'episode_count': self.episode_count}, f)

    def log_step(self, loss: float, reward: float) -> None:
        """Append one telemetry row (episode, step, loss, ε, reward, buffer).

        The file is append-only and survives restarts, so the whole training
        history stays in one CSV (see tools/plot_training.py). No-op when frozen.
        """
        if self.frozen:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with open(self.log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(TRAINING_LOG_FIELDS)
            w.writerow([self.episode_count + 1, self.step_count, f'{loss:.6f}',
                        f'{self.epsilon:.5f}', f'{reward:.6f}', len(self.replay)])

    def save_snapshot(self) -> Path | None:
        """Write versioned weights to ``snapshots/ep_<NNNN>/`` (never overwritten).

        Returns the directory, or ``None`` when frozen.
        """
        if self.frozen:
            return None
        torch, _ = _require_torch()
        out = self.snapshot_dir / f'ep_{self.episode_count:04d}'
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent_net.state_dict(), out / 'agent.pt')
        torch.save(self.mixer.state_dict(), out / 'mixer.pt')
        with open(out / 'meta.json', 'w') as f:
            json.dump({'episode': self.episode_count, 'step': self.step_count,
                       'epsilon': self.epsilon}, f)
        return out

    # -- decentralised execution ---------------------------------------

    def act(self, obs_feats: np.ndarray,
            rng: random.Random | None = None) -> int:
        """Epsilon-greedy candidate index from ONE robot's local observation.

        Uses only the shared agent net — never the mixer or the global state —
        so execution is decentralised by construction.
        """
        return select_action(obs_feats, self.epsilon, self.agent_net,
                             self.device, rng)

    # -- centralised training ------------------------------------------

    def push(self, t: JointTransition) -> None:
        """Add one joint transition to the replay buffer (no-op when frozen)."""
        if not self.frozen:
            self.replay.push(t)

    def _next_agent_qs(self, batch: list[JointTransition]):
        """[B, n] of max_a' Q_i_target(o'_i, a'); 0 where no candidate exists."""
        torch, _ = _require_torch()
        sizes = [len(c) for t in batch for c in t.next_candidates]
        out = torch.zeros(len(batch) * self.n_agents, device=self.device)
        if sum(sizes) == 0:
            return out.view(len(batch), self.n_agents)
        flat = np.concatenate([np.asarray(c, np.float32).reshape(-1, self.obs_dim)
                               for t in batch for c in t.next_candidates])
        q = self.target_agent_net(
            torch.from_numpy(flat).to(self.device)).squeeze(-1)
        start = 0
        for j, k in enumerate(sizes):
            if k:
                out[j] = q[start:start + k].max()
            start += k
        return out.view(len(batch), self.n_agents)

    def train_step(self) -> float | None:
        """One QMIX gradient step on a joint mini-batch; returns loss or None.

        y = r + γ(1−done)·Q_tot_target(s', [max_a' Q_i_target(o'_i, a')]_i),
        Q_tot = mixer(s, [Q_i(o_i, a_i)]_i), smooth-L1, Adam over agent net +
        mixer. Returns None until the buffer holds enough samples; epsilon is
        not touched here (see :meth:`end_episode`).
        """
        if self.frozen:
            return None
        torch, nn = _require_torch()
        if len(self.replay) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
            return None

        batch = self.replay.sample(BATCH_SIZE)
        b = len(batch)

        def tensor(key):
            return torch.from_numpy(np.stack(
                [np.asarray(getattr(t, key), np.float32) for t in batch]
            )).to(self.device)

        actions, mask, states = tensor('actions'), tensor('mask'), tensor('state')
        next_states, next_mask = tensor('next_state'), tensor('next_mask')
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32,
                               device=self.device)
        dones = torch.tensor([float(t.done) for t in batch],
                             dtype=torch.float32, device=self.device)

        agent_qs = self.agent_net(
            actions.view(b * self.n_agents, self.obs_dim)).view(b, self.n_agents)
        q_tot = self.mixer(agent_qs, states, mask)
        with torch.no_grad():
            next_qs = self._next_agent_qs(batch)
            q_tot_next = self.target_mixer(next_qs, next_states, next_mask)
            target = rewards + GAMMA * (1.0 - dones) * q_tot_next

        loss = nn.functional.smooth_l1_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            GRAD_CLIP_NORM)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_INTERVAL == 0:
            self._sync_targets()
        return float(loss.item())

    def end_episode(self) -> None:
        """Advance the epsilon schedule by one step; call once per episode.

        Per-episode (not per-gradient-step) decay keeps exploration high for
        ≈150 episodes instead of collapsing within the first few. No-op when
        frozen.
        """
        if self.frozen:
            return
        self.epsilon = decay_epsilon(self.epsilon)
        self.episode_count += 1
        if self.episode_count % SNAPSHOT_INTERVAL == 0:
            self.save_snapshot()
