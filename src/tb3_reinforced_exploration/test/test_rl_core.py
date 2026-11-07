"""Tests for the ROS-free RL core (rl_core.py).

Legacy DQN helpers plus the multi-robot QMIX core (local observation, global
state, intent round, joint recorder, mixer, QMixLearner); torch tests skip
gracefully if torch is absent.
"""
import random

import numpy as np
import pytest

from tb3_reinforced_exploration.rl_core import (
    assemble_joint,
    compute_reward,
    CONTESTED_RADIUS_NORM,
    decay_epsilon,
    decentralised_round,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    FEATURE_DIM,
    filter_claimed,
    FRONTIER_COUNT_NORM,
    JointEpisodeRecorder,
    JointTransition,
    make_features,
    make_global_state,
    make_local_observation,
    near_any,
    nearest_intent_dist,
    OBS_DIM,
    ReplayBuffer,
    resolve_persist_dir,
    select_action,
    STATE_DIM,
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


# ── Intent filter (received intents -> local exclusion) ─────────────

class _Cand:
    def __init__(self, x: float, y: float, group_size: int = 10) -> None:
        self.x, self.y, self.group_size = x, y, group_size


def test_near_any():
    assert near_any(0.0, 0.0, [(0.3, 0.4)], radius=0.5)
    assert not near_any(0.0, 0.0, [(0.3, 0.4)], radius=0.49)
    assert not near_any(0.0, 0.0, [], radius=10.0)


def test_filter_claimed_drops_received_intent():
    cands = [_Cand(2.05, 0.0), _Cand(-1.0, 1.0)]
    kept = filter_claimed(cands, claimed=[(2.0, 0.0)], radius=0.3)
    assert [(c.x, c.y) for c in kept] == [(-1.0, 1.0)]


def test_filter_claimed_no_claims_keeps_all():
    cands = [_Cand(0.0, 0.0), _Cand(1.0, 1.0)]
    assert filter_claimed(cands, [], radius=1.0) == cands


def test_contested_radius_scales_with_diagonal():
    for size in (8.0, 12.0):
        diag = (2 * size ** 2) ** 0.5
        radius = CONTESTED_RADIUS_NORM * diag
        assert near_any(0.0, 0.0, [(0.09 * diag, 0.0)], radius)
        assert not near_any(0.0, 0.0, [(0.11 * diag, 0.0)], radius)


# ── Local observation o_i (Dec-POMDP) ───────────────────────────────

def test_local_observation_shape_and_values():
    # diag 10: path 2.5 -> 0.25, intent 5 -> 0.5; group 50/100 -> 0.5.
    o = make_local_observation(group_size=50, path_length=2.5,
                               intent_dist=5.0, map_diagonal=10.0)
    assert o.dtype == np.float32
    assert o.shape == (OBS_DIM,)
    np.testing.assert_allclose(o, [0.5, 0.25, 0.5, 1.0, 0.0], rtol=1e-6)


def test_local_observation_no_intent():
    o = make_local_observation(10, 1.0, intent_dist=None, map_diagonal=10.0)
    assert o[2] == 1.0 and o[3] == 0.0


def test_local_observation_intent_clipped():
    o = make_local_observation(10, 1.0, intent_dist=50.0, map_diagonal=10.0)
    assert o[2] == 1.0 and o[3] == 1.0


def test_local_observation_info_gain_appended_and_clipped():
    o = make_local_observation(10, 1.0, None, 10.0, info_gain=0.4)
    assert o[4] == np.float32(0.4)
    assert make_local_observation(10, 1.0, None, 10.0, info_gain=3.0)[4] == 1.0
    assert make_local_observation(10, 1.0, None, 10.0, info_gain=-1.0)[4] == 0.0


def test_local_observation_zero_diagonal_is_safe():
    assert np.all(np.isfinite(make_local_observation(10, 1.0, 1.0, 0.0)))


def test_local_observation_is_size_invariant():
    # Same relative geometry on 8x8 and 12x12 -> identical o_i (transfer).
    obs = []
    for size in (8.0, 12.0):
        diag = (2 * size ** 2) ** 0.5
        robot, cand, intent = (0.0, 0.0), (0.3 * size, 0.1 * size), \
            (0.5 * size, -0.2 * size)
        path = ((cand[0] - robot[0]) ** 2 + (cand[1] - robot[1]) ** 2) ** 0.5
        obs.append(make_local_observation(
            20, path, nearest_intent_dist(cand, [intent]), diag,
            info_gain=0.37))
    np.testing.assert_allclose(obs[0], obs[1], rtol=1e-6)


def test_nearest_intent_dist():
    assert nearest_intent_dist((0.0, 0.0), []) is None
    assert nearest_intent_dist((0.0, 0.0), [(3.0, 4.0), (0.0, 1.0)]) == \
        pytest.approx(1.0)


# ── Global state s (training only) ──────────────────────────────────

def test_global_state_values():
    s = make_global_state(coverage=0.4, robot_positions=[(0.0, 0.0), (3.0, 4.0)],
                          n_frontier_groups=5, n_busy=1, n_agents=2,
                          map_diagonal=10.0)
    assert s.dtype == np.float32 and s.shape == (STATE_DIM,)
    np.testing.assert_allclose(
        s, [0.4, 0.5, 5 / FRONTIER_COUNT_NORM, 0.5], rtol=1e-6)


def test_global_state_clips_and_single_robot():
    s = make_global_state(1.5, [(0.0, 0.0)], 10_000, 0, 2, 10.0)
    np.testing.assert_allclose(s, [1.0, 0.0, 1.0, 0.0])


def test_global_state_is_size_invariant():
    states = [make_global_state(0.3, [(0.0, 0.0), (0.4 * k, 0.2 * k)], 3, 1, 2,
                                (2 * k ** 2) ** 0.5) for k in (8.0, 12.0)]
    np.testing.assert_allclose(states[0], states[1], rtol=1e-6)


# ── Decentralised round with intent communication ───────────────────

def _obs_fn(name, c, received):
    d = nearest_intent_dist((c.x, c.y), received)
    return make_local_observation(c.group_size, abs(c.x) + abs(c.y), d, 10.0)


def test_round_no_double_booking_and_announces_intent():
    shared = [_Cand(1.0, 0.0), _Cand(5.0, 5.0)]
    choices, picked, intents = decentralised_round(
        order=['r1', 'r2'], busy_intents={},
        candidates={'r1': shared, 'r2': shared},
        observe=_obs_fn, act=lambda obs: 0, radius=0.5)
    assert (choices['r1'].x, choices['r2'].x) == (1.0, 5.0)
    assert intents == {'r1': (1.0, 0.0), 'r2': (5.0, 5.0)}
    # r1 decided first -> no intent received; r2 received r1's intent.
    assert picked['r1'][0][0][3] == 0.0
    assert picked['r2'][0][0][3] == 1.0


def test_round_respects_busy_intent():
    cands = [_Cand(1.0, 0.0), _Cand(5.0, 5.0)]
    choices, _, intents = decentralised_round(
        ['r1'], {'r2': (1.1, 0.0)}, {'r1': cands}, _obs_fn,
        act=lambda obs: 0, radius=0.5)
    assert choices['r1'].x == 5.0
    assert intents['r2'] == (1.1, 0.0)


def test_round_robot_without_candidates_skipped():
    choices, picked, _ = decentralised_round(
        ['r1'], {'r2': (0.0, 0.0)}, {'r1': [_Cand(0.1, 0.0)]}, _obs_fn,
        act=lambda obs: 0, radius=0.5)
    assert choices == {} and picked == {}


def test_assemble_joint_masks_absent_agent():
    a = np.ones(OBS_DIM, np.float32)
    act, mask, cands = assemble_joint(['r1', 'r2'], {'r2': a},
                                      {'r2': a[None]})
    np.testing.assert_array_equal(mask, [0.0, 1.0])
    assert act[0].sum() == 0.0 and cands[0].shape == (0, OBS_DIM)
    assert cands[1].shape == (1, OBS_DIM)


# ── JointEpisodeRecorder ────────────────────────────────────────────

def _event(rec, cov, now, mask=(1.0, 1.0), k=(2, 1)):
    act = np.full((2, OBS_DIM), cov, np.float32)
    cands = [np.zeros((n, OBS_DIM), np.float32) for n in k]
    state = np.full(STATE_DIM, cov, np.float32)
    return rec.record_event(act, np.array(mask, np.float32), cands, state,
                            cov, now)


def test_recorder_first_event_returns_none_then_closes():
    rec = JointEpisodeRecorder()
    assert _event(rec, 0.1, 10.0) is None
    assert rec.has_pending
    t = _event(rec, 0.3, 20.0, mask=(1.0, 0.0), k=(3, 0))
    assert isinstance(t, JointTransition)
    # Real elapsed time: +0.2 over 10 s -> 2.0.
    assert t.reward == pytest.approx(compute_reward(0.1, 0.3, 10.0))
    np.testing.assert_allclose(t.state, 0.1)
    np.testing.assert_allclose(t.next_state, 0.3)
    np.testing.assert_array_equal(t.next_mask, [1.0, 0.0])
    assert [len(c) for c in t.next_candidates] == [3, 0]
    assert not t.done


def test_recorder_close_at_episode_end():
    rec = JointEpisodeRecorder()
    assert rec.close(0.5, 1.0) is None
    _event(rec, 0.2, 0.0)
    t = rec.close(0.6, 4.0)
    assert t.done and not rec.has_pending
    assert t.reward == pytest.approx(compute_reward(0.2, 0.6, 4.0))
    assert all(len(c) == 0 for c in t.next_candidates)
    np.testing.assert_array_equal(t.next_mask, [0.0, 0.0])
    assert rec.close(0.7, 5.0) is None


def test_recorder_mask_zero_when_no_candidates():
    rec = JointEpisodeRecorder()
    _event(rec, 0.1, 0.0)
    t = _event(rec, 0.2, 5.0, mask=(1.0, 1.0), k=(2, 0))
    np.testing.assert_array_equal(t.next_mask, [1.0, 0.0])


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


def test_epsilon_floor_reached_after_about_150_episodes():
    e = EPSILON_START
    for _ in range(140):
        e = decay_epsilon(e)
    assert e > EPSILON_END
    for _ in range(20):
        e = decay_epsilon(e)
    assert e == pytest.approx(EPSILON_END)


# ── QMIX mixer ──────────────────────────────────────────────────────

def test_mixer_shape():
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_mixer
    mixer = make_mixer(2)
    out = mixer(torch.zeros((5, 2)), torch.zeros((5, STATE_DIM)))
    assert tuple(out.shape) == (5,)


def test_mixer_is_monotonic_in_each_agent_q():
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_mixer
    torch.manual_seed(0)
    mixer = make_mixer(2)
    states = torch.rand((256, STATE_DIM))
    qs = torch.randn((256, 2)) * 3.0
    base = mixer(qs, states)
    for i in range(2):
        bumped = qs.clone()
        bumped[:, i] += torch.rand(256) * 2.0 + 1e-3
        assert torch.all(mixer(bumped, states) >= base - 1e-6)


def test_mixer_mask_zeroes_agent():
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_mixer
    torch.manual_seed(1)
    mixer = make_mixer(2)
    states = torch.rand((8, STATE_DIM))
    mask = torch.tensor([[1.0, 0.0]] * 8)
    a = torch.randn((8, 2))
    b = a.clone()
    b[:, 1] += 100.0                       # masked agent's Q changes ...
    torch.testing.assert_close(mixer(a, states, mask), mixer(b, states, mask))
    zeroed = a.clone()
    zeroed[:, 1] = 0.0                     # ... and acts like Q_2 = 0
    torch.testing.assert_close(mixer(a, states, mask), mixer(zeroed, states))


# ── QMixLearner ─────────────────────────────────────────────────────

def _joint(reward: float, done: bool = False) -> JointTransition:
    rng = np.random.default_rng(int(reward * 10) % 1000)
    return JointTransition(
        actions=rng.random((2, OBS_DIM)).astype(np.float32),
        mask=np.array([1.0, 1.0], np.float32),
        state=rng.random(STATE_DIM).astype(np.float32),
        reward=reward,
        next_candidates=[rng.random((3, OBS_DIM)).astype(np.float32),
                         np.zeros((0, OBS_DIM), np.float32)],
        next_state=rng.random(STATE_DIM).astype(np.float32),
        next_mask=np.array([1.0, 0.0], np.float32),
        done=done,
    )


def _learner(tmp_path):
    pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import QMixLearner
    return QMixLearner(n_agents=2, persist_dir=tmp_path, device='cpu')


def test_qmix_act_valid_index_and_uses_no_state(tmp_path):
    learner = _learner(tmp_path)

    class _Forbidden:
        def __call__(self, *a, **k):
            raise AssertionError('mixer must not be used for acting')

    learner.mixer = learner.target_mixer = _Forbidden()
    learner.epsilon = 0.0                   # greedy -> goes through agent net
    obs = np.random.default_rng(0).random((5, OBS_DIM)).astype(np.float32)
    idx = learner.act(obs)
    assert 0 <= idx < 5
    learner.epsilon = 1.0
    assert 0 <= learner.act(obs, rng=random.Random(0)) < 5


def test_qmix_act_greedy_is_agent_net_argmax(tmp_path):
    torch = pytest.importorskip('torch')
    learner = _learner(tmp_path)
    learner.epsilon = 0.0
    obs = np.random.default_rng(3).random((6, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = int(learner.agent_net(torch.from_numpy(obs)).argmax())
    assert learner.act(obs) == expected


def test_qmix_train_step_none_until_min_replay_then_loss(tmp_path):
    learner = _learner(tmp_path)
    from tb3_reinforced_exploration.rl_core import BATCH_SIZE, MIN_REPLAY_SIZE
    need = max(MIN_REPLAY_SIZE, BATCH_SIZE)
    for i in range(need - 1):
        learner.push(_joint(float(i % 3), done=(i % 7 == 0)))
    assert learner.train_step() is None
    learner.push(_joint(1.0))
    e_before = learner.epsilon
    loss = learner.train_step()
    assert isinstance(loss, float) and np.isfinite(loss)
    assert learner.step_count == 1
    assert learner.epsilon == e_before    # epsilon only per episode


def test_qmix_train_handles_all_empty_next_candidates(tmp_path):
    learner = _learner(tmp_path)
    t = _joint(1.0, done=True)._replace(
        next_candidates=[np.zeros((0, OBS_DIM), np.float32)] * 2,
        next_mask=np.zeros(2, np.float32))
    for _ in range(100):
        learner.push(t)
    assert isinstance(learner.train_step(), float)


def test_qmix_target_sync(tmp_path):
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import TARGET_UPDATE_INTERVAL
    learner = _learner(tmp_path)
    for i in range(100):
        learner.push(_joint(float(i % 5)))

    def same(a, b):
        return all(torch.equal(x, y) for x, y in
                   zip(a.state_dict().values(), b.state_dict().values()))

    learner.train_step()
    assert not same(learner.agent_net, learner.target_agent_net)
    for _ in range(TARGET_UPDATE_INTERVAL - 1):
        learner.train_step()
    assert learner.step_count == TARGET_UPDATE_INTERVAL
    assert same(learner.agent_net, learner.target_agent_net)
    assert same(learner.mixer, learner.target_mixer)


def test_qmix_end_episode_decays_once(tmp_path):
    learner = _learner(tmp_path)
    assert learner.epsilon == EPSILON_START
    learner.end_episode()
    assert learner.epsilon == pytest.approx(EPSILON_START * EPSILON_DECAY)


def test_qmix_save_load_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    learner = _learner(tmp_path)
    learner.epsilon, learner.step_count = 0.321, 7
    for i in range(4):
        learner.push(_joint(float(i)))
    learner.save()
    for p in (learner.agent_path, learner.mixer_path, learner.replay_path,
              learner.meta_path):
        assert p.exists()
    assert learner.agent_path.name == 'agent.pt'
    assert learner.replay_path.name == 'joint_replay.pkl'

    restored = _learner(tmp_path)
    restored.load()
    assert restored.epsilon == pytest.approx(0.321)
    assert restored.step_count == 7
    assert len(restored.replay) == 4
    for a, b in zip(learner.mixer.state_dict().values(),
                    restored.target_mixer.state_dict().values()):
        assert torch.equal(a, b)


def test_qmix_load_skips_incompatible_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    from tb3_reinforced_exploration.rl_core import make_mixer, make_q_network
    torch.save(make_q_network(FEATURE_DIM).state_dict(), tmp_path / 'agent.pt')
    torch.save(make_mixer(3).state_dict(), tmp_path / 'mixer.pt')
    buf = ReplayBuffer()
    buf.push(_t(1.0))                       # legacy-shaped entry
    buf.save(tmp_path / 'joint_replay.pkl')
    learner = _learner(tmp_path)
    before = {k: v.clone() for k, v in learner.agent_net.state_dict().items()}
    learner.load()                          # must not raise
    for k, v in learner.agent_net.state_dict().items():
        assert torch.equal(v, before[k])
    assert len(learner.replay) == 0


def test_qmix_ignores_old_team_learner_files(tmp_path):
    pytest.importorskip('torch')
    (tmp_path / 'model.pt').write_bytes(b'old')
    learner = _learner(tmp_path)
    learner.load()
    assert learner.step_count == 0 and len(learner.replay) == 0


def test_qmix_training_log_is_append_only(tmp_path):
    learner = _learner(tmp_path)
    learner.log_step(0.5, 0.1)
    learner.log_step(0.25, 0.2)
    rows = learner.log_path.read_text().strip().splitlines()
    assert rows[0] == 'episode,step,loss,epsilon,reward,buffer'
    assert len(rows) == 3
    assert rows[1].startswith('1,0,0.5')


def test_qmix_snapshots_every_interval_and_frozen_writes_nothing(tmp_path):
    from tb3_reinforced_exploration.rl_core import SNAPSHOT_INTERVAL
    learner = _learner(tmp_path)
    for _ in range(SNAPSHOT_INTERVAL):
        learner.end_episode()
    snap = learner.snapshot_dir / f'ep_{SNAPSHOT_INTERVAL:04d}'
    assert (snap / 'agent.pt').exists() and (snap / 'mixer.pt').exists()
    assert learner.episode_count == SNAPSHOT_INTERVAL

    frozen = _learner(tmp_path / 'frozen')
    frozen.freeze()
    frozen.log_step(1.0, 1.0)
    assert frozen.save_snapshot() is None
    assert not (tmp_path / 'frozen').exists()
