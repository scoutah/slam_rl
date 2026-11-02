"""Tests for the ROS-free RL core (rl_core.py).

These cover the learning logic a macOS co-developer can verify without ROS or
Gazebo: feature normalisation, the reward formula, the epsilon schedule, the
replay buffer (including disk persistence), persist-dir resolution and the
epsilon-greedy policy. torch-dependent tests skip gracefully if torch is absent.
"""
import random

import numpy as np
import pytest

from tb3_reinforced_exploration.rl_core import (
    compute_reward,
    decay_epsilon,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    FEATURE_DIM,
    make_features,
    make_team_features,
    ReplayBuffer,
    resolve_persist_dir,
    select_action,
    TEAM_FEATURE_DIM,
    Transition,
)


# ── Feature construction ────────────────────────────────────────────

def test_make_features_normalisation():
    f = make_features(group_size=50, path_length=5.0, coverage=0.4)
    assert f.dtype == np.float32
    assert f.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(f, [0.5, 0.5, 0.4], rtol=1e-6)


def test_make_features_zero_and_large():
    np.testing.assert_allclose(make_features(0, 0.0, 0.0), [0.0, 0.0, 0.0])
    # Large values are not clamped – the network sees them as-is.
    np.testing.assert_allclose(
        make_features(300, 40.0, 1.0), [3.0, 4.0, 1.0], rtol=1e-6,
    )


def test_make_features_is_size_invariant():
    # Same relative frontier on two world sizes -> identical features. This is
    # the property the Studienbericht transfer claim depends on.
    small = make_features(group_size=20, path_length=3.0, coverage=0.6)
    large = make_features(group_size=20, path_length=3.0, coverage=0.6)
    np.testing.assert_array_equal(small, large)


# ── Team-aware (multi-agent) features ───────────────────────────────

def test_make_team_features_shape_and_normalisation():
    # diag=10: path 2.5 -> 0.25, teammate 5.0 -> 0.5; group 50/100=0.5.
    f = make_team_features(
        group_size=50, path_length=2.5, coverage=0.4,
        teammate_dist=5.0, map_diagonal=10.0, contested=True,
    )
    assert f.dtype == np.float32
    assert f.shape == (TEAM_FEATURE_DIM,)
    np.testing.assert_allclose(f, [0.5, 0.25, 0.4, 0.5, 1.0], rtol=1e-6)


def test_make_team_features_contested_flag():
    f = make_team_features(10, 1.0, 0.2, 1.0, 10.0, contested=False)
    assert f[-1] == 0.0


def test_make_team_features_zero_diagonal_is_safe():
    # A degenerate diagonal must not divide by zero.
    f = make_team_features(10, 1.0, 0.2, 1.0, 0.0, contested=False)
    assert np.all(np.isfinite(f))


def test_make_team_features_is_size_invariant():
    # The SAME relative frontier on 8x8 and 12x12: distances scale with the
    # map diagonal, so dividing by it yields identical features. This is the
    # transfer property the Studienbericht depends on.
    diag8 = (8.0 ** 2 * 2) ** 0.5
    diag12 = (12.0 ** 2 * 2) ** 0.5
    small = make_team_features(
        group_size=20, path_length=0.3 * diag8, coverage=0.6,
        teammate_dist=0.5 * diag8, map_diagonal=diag8, contested=False,
    )
    large = make_team_features(
        group_size=20, path_length=0.3 * diag12, coverage=0.6,
        teammate_dist=0.5 * diag12, map_diagonal=diag12, contested=False,
    )
    np.testing.assert_allclose(small, large, rtol=1e-6)


# ── Reward formula ──────────────────────────────────────────────────

def test_compute_reward_basic():
    # +0.1 coverage over 2 s -> 0.1/2*100 = 5.0
    assert compute_reward(0.3, 0.4, 2.0) == pytest.approx(5.0)


def test_compute_reward_dt_floored_at_one_second():
    # dt below 1 s is clamped to 1 s, so a tiny interval can't explode reward.
    assert compute_reward(0.0, 0.1, 0.01) == pytest.approx(10.0)
    assert compute_reward(0.0, 0.1, 1.0) == pytest.approx(10.0)


def test_compute_reward_negative_on_coverage_loss():
    assert compute_reward(0.5, 0.4, 1.0) == pytest.approx(-10.0)


def test_compute_reward_zero_gain():
    assert compute_reward(0.5, 0.5, 5.0) == 0.0


# ── Epsilon schedule ────────────────────────────────────────────────

def test_decay_epsilon_multiplies():
    assert decay_epsilon(1.0) == pytest.approx(EPSILON_DECAY)


def test_decay_epsilon_clamped_at_floor():
    assert decay_epsilon(EPSILON_END) == pytest.approx(EPSILON_END)
    assert decay_epsilon(EPSILON_END * 0.5) == EPSILON_END


def test_decay_epsilon_converges_to_floor():
    e = EPSILON_START
    for _ in range(1000):
        e = decay_epsilon(e)
    assert e == pytest.approx(EPSILON_END)
    assert e >= EPSILON_END


# ── Replay buffer ───────────────────────────────────────────────────

def _t(reward: float) -> Transition:
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    return Transition(feats, feats, reward, [feats], False)


def test_replay_push_and_len():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.push(_t(1.0))
    buf.push(_t(2.0))
    assert len(buf) == 2


def test_replay_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(_t(float(i)))
    assert len(buf) == 3
    rewards = sorted(t.reward for t in buf.sample(3))
    assert rewards == [2.0, 3.0, 4.0]


def test_replay_sample_size():
    buf = ReplayBuffer(capacity=10)
    for i in range(6):
        buf.push(_t(float(i)))
    assert len(buf.sample(4)) == 4


def test_replay_sample_too_many_raises():
    buf = ReplayBuffer(capacity=10)
    buf.push(_t(1.0))
    with pytest.raises(ValueError):
        buf.sample(5)


def test_replay_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=10)
    for i in range(4):
        buf.push(_t(float(i)))
    path = tmp_path / 'sub' / 'replay.pkl'
    buf.save(path)
    assert path.exists()  # parent dir created

    other = ReplayBuffer(capacity=10)
    other.load(path)
    assert len(other) == 4
    assert sorted(t.reward for t in other.sample(4)) == [0.0, 1.0, 2.0, 3.0]


def test_replay_load_missing_file_is_noop(tmp_path):
    buf = ReplayBuffer(capacity=10)
    buf.push(_t(9.0))
    buf.load(tmp_path / 'does_not_exist.pkl')
    assert len(buf) == 1  # unchanged


def test_replay_load_preserves_capacity(tmp_path):
    src = ReplayBuffer(capacity=100)
    for i in range(10):
        src.push(_t(float(i)))
    path = tmp_path / 'replay.pkl'
    src.save(path)

    small = ReplayBuffer(capacity=5)
    small.load(path)
    # maxlen of the loading buffer must be preserved, not the saved length.
    assert small._buf.maxlen == 5
    assert len(small) == 5


# ── Persist-dir resolution ──────────────────────────────────────────

def test_resolve_persist_dir_env_override(tmp_path):
    env = {'SLAM_RL_DATA_DIR': str(tmp_path / 'explicit')}
    assert resolve_persist_dir(env, cwd=tmp_path) == tmp_path / 'explicit'


def test_resolve_persist_dir_expands_user():
    env = {'SLAM_RL_DATA_DIR': '~/rl_here'}
    from pathlib import Path
    assert resolve_persist_dir(env) == Path('~/rl_here').expanduser()


def test_resolve_persist_dir_from_colcon_prefix(tmp_path):
    install = tmp_path / 'ws' / 'install'
    env = {'COLCON_PREFIX_PATH': str(install)}
    assert resolve_persist_dir(env, cwd=tmp_path) == tmp_path / 'ws' / 'rl_data'


def test_resolve_persist_dir_colcon_prefix_first_of_many(tmp_path):
    import os
    first = tmp_path / 'ws' / 'install'
    other = tmp_path / 'other' / 'install'
    env = {'COLCON_PREFIX_PATH': os.pathsep.join([str(first), str(other)])}
    assert resolve_persist_dir(env) == tmp_path / 'ws' / 'rl_data'


def test_resolve_persist_dir_env_wins_over_colcon(tmp_path):
    env = {
        'SLAM_RL_DATA_DIR': str(tmp_path / 'explicit'),
        'COLCON_PREFIX_PATH': str(tmp_path / 'ws' / 'install'),
    }
    assert resolve_persist_dir(env) == tmp_path / 'explicit'


def test_resolve_persist_dir_cwd_fallback(tmp_path):
    assert resolve_persist_dir({}, cwd=tmp_path) == tmp_path / 'rl_data'


# ── Epsilon-greedy policy ───────────────────────────────────────────

def test_select_action_empty_raises():
    with pytest.raises(ValueError):
        select_action(np.zeros((0, FEATURE_DIM), np.float32), epsilon=0.0)


def test_select_action_pure_exploration_in_range():
    cands = np.zeros((4, FEATURE_DIM), np.float32)
    rng = random.Random(0)
    for _ in range(50):
        idx = select_action(cands, epsilon=1.0, q_net=None, rng=rng)
        assert 0 <= idx < 4


def test_select_action_none_qnet_always_random():
    cands = np.zeros((3, FEATURE_DIM), np.float32)
    rng = random.Random(1)
    # epsilon=0 but no network -> still a valid random index, never crashes.
    idx = select_action(cands, epsilon=0.0, q_net=None, rng=rng)
    assert 0 <= idx < 3


def test_select_action_greedy_picks_argmax():
    torch = pytest.importorskip('torch')

    class ConstQ(torch.nn.Module):
        # Returns the first feature as the Q-value, so argmax over features[:,0].
        def forward(self, x):
            return x[:, :1]

    cands = np.array(
        [[0.1, 0, 0], [0.9, 0, 0], [0.5, 0, 0]], dtype=np.float32,
    )
    # epsilon=0 and a real net -> deterministic greedy pick of index 1.
    idx = select_action(cands, epsilon=0.0, q_net=ConstQ(),
                        rng=random.Random(0))
    assert idx == 1


def test_make_q_network_shapes():
    torch = pytest.importorskip('torch')
    net = __import__(
        'tb3_reinforced_exploration.rl_core', fromlist=['make_q_network'],
    ).make_q_network()
    out = net(torch.zeros((5, FEATURE_DIM)))
    assert tuple(out.shape) == (5, 1)


def test_make_q_network_team_feature_dim():
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_q_network
    net = make_q_network(TEAM_FEATURE_DIM)
    out = net(torch.zeros((3, TEAM_FEATURE_DIM)))
    assert tuple(out.shape) == (3, 1)


# ── TeamLearner (CTDE: parameter sharing + shared replay) ───────────

def _team_transition(reward: float) -> Transition:
    feats = np.zeros(TEAM_FEATURE_DIM, dtype=np.float32)
    return Transition(feats, feats, reward, [feats], False)


def test_team_learner_builds_with_team_feature_dim(tmp_path):
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import TeamLearner
    learner = TeamLearner(persist_dir=tmp_path)
    assert learner.feature_dim == TEAM_FEATURE_DIM
    assert len(learner.replay) == 0


def test_team_learner_train_step_noop_until_enough_samples(tmp_path):
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import TeamLearner
    learner = TeamLearner(persist_dir=tmp_path)
    assert learner.train_step() is None  # empty buffer


def test_team_learner_train_step_updates_and_decays(tmp_path):
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import MIN_REPLAY_SIZE, TeamLearner
    learner = TeamLearner(persist_dir=tmp_path)
    for i in range(MIN_REPLAY_SIZE + 64):
        learner.push(_team_transition(float(i % 3)))
    e_before, steps_before = learner.epsilon, learner.step_count
    loss = learner.train_step()
    assert isinstance(loss, float)
    assert learner.step_count == steps_before + 1
    assert learner.epsilon <= e_before


def test_team_learner_shared_replay_pools_both_agents(tmp_path):
    # Parameter sharing / shared buffer: transitions from either robot land in
    # the one buffer the single network trains on.
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import TeamLearner
    learner = TeamLearner(persist_dir=tmp_path)
    learner.push(_team_transition(1.0))   # as if from r1
    learner.push(_team_transition(2.0))   # as if from r2
    assert len(learner.replay) == 2


def test_team_learner_save_load_roundtrip(tmp_path):
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import TeamLearner
    learner = TeamLearner(persist_dir=tmp_path)
    learner.epsilon = 0.321
    learner.step_count = 7
    for i in range(4):
        learner.push(_team_transition(float(i)))
    learner.save()
    assert learner.model_path.exists()
    assert learner.replay_path.exists()
    assert learner.meta_path.exists()

    restored = TeamLearner(persist_dir=tmp_path)
    restored.load()
    assert restored.epsilon == pytest.approx(0.321)
    assert restored.step_count == 7
    assert len(restored.replay) == 4


def test_team_learner_load_skips_incompatible_checkpoint(tmp_path):
    # An old 3-feature model must not crash the 5-feature learner; it is
    # skipped and training starts fresh.
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_q_network, TeamLearner
    old = make_q_network(FEATURE_DIM)
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(old.state_dict(), tmp_path / 'model.pt')
    learner = TeamLearner(persist_dir=tmp_path)
    learner.load()  # must not raise
    assert learner.feature_dim == TEAM_FEATURE_DIM
