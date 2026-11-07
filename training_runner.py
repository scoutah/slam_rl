#!/usr/bin/env python3
"""Episodic SLAM training runner.

Generates/reuses a world, launches the sim, waits for the episode-end signal and
writes a per-run CSV in results/. Run ``python3 training_runner.py --help`` for the
options; the episode timeout is derived from world size (#10), not a CLI flag.
"""

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

EPISODE_FILE = Path('/tmp/slam_episode_done')
# episode_manager mirrors its sim-time elapsed here (see EpisodeManager docstring).
SIM_ELAPSED_FILE = Path(str(EPISODE_FILE) + '.sim_s')
SHUTDOWN_GRACE = 15      # seconds to wait for ros2 launch to exit cleanly
INTER_EPISODE_PAUSE = 3  # seconds to wait after shutdown before next launch
TRAIN_WORLD_SIZE = 8.0   # the RL policy is trained on this edge length (in-distribution)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / 'results'
WORLDS_DIR = PROJECT_ROOT / 'worlds'   # overridden by --worlds-dir

# ROS node name substrings whose log lines are suppressed in the terminal
LOG_FILTER: list[str] = [
    'cartographer_node-',
    'controller_server-',
    'rviz2-',
    'gazebo-',
    'parameter_bridge-',
    'create-',
    'cmd_vel_relay-',
    'static_transform_publisher-',
    'tf_relay-',
    'map_merger-',
    'robot_state_publisher-',
    'lifecycle_manager-',
    'smoother_server-',
    'route_server-',
    'behavior_server-',
    'waypoint_follower-',
    'velocity_smoother-',
    'collision_monitor-',
    'opennav_docking-',
    'bt_navigator-',
    'planner_server-',
]


# ---------------------------------------------------------------------------
# Log filtering
# ---------------------------------------------------------------------------

def _stream_filter(src: io.RawIOBase, dst: io.TextIOWrapper) -> None:
    """Read *src* line by line, writing to *dst* unless it matches a filter."""
    for raw in iter(src.readline, b''):
        line = raw.decode('utf-8', errors='replace')
        if any(f in line for f in LOG_FILTER):
            continue
        dst.write(line)
        dst.flush()


def start_filtered_readers(proc: subprocess.Popen) -> list[threading.Thread]:
    """Spawn daemon threads that forward filtered subprocess output to the terminal."""
    threads = [
        threading.Thread(target=_stream_filter, args=(proc.stdout, sys.stdout), daemon=True),
        threading.Thread(target=_stream_filter, args=(proc.stderr, sys.stderr), daemon=True),
    ]
    for t in threads:
        t.start()
    return threads


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------

def prepare_world(
    world_arg: str | None,
    run_id: str,
    save: bool = False,
    world_size: float | None = None,
    seed: int | None = None,
) -> Path:
    """
    Return the path to the .sdf world file to use for this run.

    If *world_arg* is given, use that file directly (no copy).
    Otherwise generate a new random world via tb3_slam.utils, with edge
    length *world_size* (generator default if None).
    When *save* is True, persist to worlds/<run_id>.sdf; otherwise
    write to /tmp/ (ephemeral). With *seed*, the world is the one for episode 1
    of that seed (see :func:`seed_world_rng`).
    """
    if world_arg:
        p = Path(world_arg).resolve()
        if not p.exists():
            print(f'[runner] ERROR: world file not found: {p}', file=sys.stderr)
            sys.exit(1)
        return p

    # Import here so the script can still --help without a colcon workspace
    from tb3_slam.utils import DEFAULT_WORLD_SIZE, generate_random_world

    if save:
        WORLDS_DIR.mkdir(parents=True, exist_ok=True)
        world_path = WORLDS_DIR / f'{run_id}.sdf'
    else:
        world_path = Path(f'/tmp/slam_world_{run_id}.sdf')
    seed_world_rng(seed, 1)
    success, _ = generate_random_world(str(world_path), world_size or DEFAULT_WORLD_SIZE)
    if not success:
        print('[runner] ERROR: world generation failed', file=sys.stderr)
        sys.exit(1)
    print(f'[runner] World saved to {world_path}')
    return world_path


def seed_world_rng(seed: int | None, episode: int) -> None:
    """Make the next generated world depend only on (*seed*, *episode*).

    Running two methods with the same ``--seed`` and ``--randomize-worlds``
    therefore gives both the same world sequence. No-op without a seed.
    """
    if seed is not None:
        random.seed(f'{seed}-{episode}')


def distribution_label(world_size: float | None) -> str:
    """``ID`` for the training size, ``OOD`` for any other; empty if unknown."""
    if world_size is None:
        return ''
    return 'ID' if abs(world_size - TRAIN_WORLD_SIZE) < 1e-6 else 'OOD'


def read_world_size(world_file: Path) -> float | None:
    """World edge length from the world's metadata sidecar, if recorded."""
    try:
        meta = json.loads(world_file.with_suffix('.meta.json').read_text())
        return float(meta['world_size'])
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def _parse_coverage(reason: str) -> float | None:
    """Extract the coverage float from a reason string like 'coverage=95.0%'."""
    m = re.search(r'coverage=([0-9.]+)%', reason)
    return float(m.group(1)) / 100.0 if m else None


def keep_world_for_rerun(world_file: Path) -> Path | None:
    """Copy a world (SDF + metadata sidecar) into ``worlds/`` unless it is already there.

    Called for ``exhausted`` episodes so they can be replayed later with
    ``--world worlds/<id>.sdf`` even if the run was started without
    ``--save-worlds``. Returns the saved path, or ``None`` if copying failed.
    """
    try:
        WORLDS_DIR.mkdir(parents=True, exist_ok=True)
        target = WORLDS_DIR / world_file.name
        if target.resolve() != world_file.resolve():
            shutil.copy2(world_file, target)
            meta = world_file.with_suffix('.meta.json')
            if meta.exists():
                shutil.copy2(meta, target.with_suffix('.meta.json'))
        return target
    except OSError as exc:
        print(f'[runner] WARNING: could not keep world {world_file}: {exc}', file=sys.stderr)
        return None


def make_result_path(mode: str, run_id: str) -> Path:
    """Build the per-run CSV path: results/<dd-mm-yyyy>-<mode>-<id>.csv"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%d-%m-%Y')
    return RESULTS_DIR / f'{date_str}-{mode}-{run_id}.csv'


def read_sim_elapsed(path: Path = SIM_ELAPSED_FILE) -> float | None:
    """Sim-time seconds of the finished episode, or None if not (yet) written."""
    try:
        return float(path.read_text())
    except (OSError, ValueError):
        return None


def append_result(
    result_file: Path,
    episode: int,
    mode: str,
    reason: str,
    elapsed: float,
    world_file: Path | None = None,
    wall_s: float | None = None,
    sim_speed: float = 1.0,
) -> None:
    """Append one episode result row to the CSV result file.

    ``elapsed`` is the episode duration in *simulation* seconds (the wall-clock
    value if the sim-time one was unavailable); ``wall_s`` is the real duration.
    """
    try:
        coverage = _parse_coverage(reason)
        world_size = read_world_size(world_file) if world_file is not None else None
        write_header = not result_file.exists()

        with result_file.open('a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            if write_header:
                writer.writerow(['timestamp', 'episode', 'mode', 'reason',
                                 'elapsed_s', 'coverage', 'world', 'world_size',
                                 'wall_s', 'sim_speed', 'distribution'])
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                episode,
                mode,
                reason,
                f'{elapsed:.1f}',
                f'{coverage:.4f}' if coverage is not None else '',
                world_file.name if world_file is not None else '',
                world_size if world_size is not None else '',
                f'{wall_s:.1f}' if wall_s is not None else '',
                f'{sim_speed:g}',
                distribution_label(world_size),
            ])
    except Exception as exc:
        print(f'[runner] ERROR writing result to {result_file}: {exc}', file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_launch_cmd(
    mode: str,
    episode_file: Path,
    threshold: float,
    headless: bool,
    rviz: bool = True,
    rl_eval: bool = False,
) -> list[str]:
    # The episode timeout is derived from the world size by episode_manager
    # (issue #10) and is deliberately not configurable here.
    cmd = [
        'ros2', 'launch', 'tb3_slam', 'sim_slam.launch.py',
        f'exploration_mode:={mode}',
        f'episode_file:={episode_file}',
        f'coverage_threshold:={threshold}',
    ]
    if rl_eval:
        cmd.append('rl_eval:=true')
    if headless:
        cmd.append('headless:=true')
    elif not rviz:
        cmd.append('rviz:=false')
    return cmd


def wait_for_episode_done(
    episode_file: Path,
    proc: subprocess.Popen,
    poll_interval: float = 1.0,
) -> str:
    """Block until the episode_done file appears or the subprocess exits."""
    while True:
        if episode_file.exists():
            return episode_file.read_text().strip()
        if proc.poll() is not None:
            time.sleep(0.2)
            if episode_file.exists():
                return episode_file.read_text().strip()
            return 'process_exited'
        time.sleep(poll_interval)


def kill_launch(proc: subprocess.Popen) -> None:
    """Send SIGINT to the launch process group (graceful), then SIGKILL if needed."""
    if proc.poll() is not None:
        _kill_stale_ros_processes()
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        _kill_stale_ros_processes()
        return
    try:
        proc.wait(timeout=SHUTDOWN_GRACE)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    _kill_stale_ros_processes()


_STALE_PATTERNS = [
    'cartographer_node',
    'ros_gz_bridge',
    'parameter_bridge',
    'controller_server',
    'planner_server',
    'smoother_server',
    'behavior_server',
    'bt_navigator',
    'waypoint_follower',
    'velocity_smoother',
    'lifecycle_manager',
    'gz sim',
    'gzserver',
    'gzclient',
    'frontier_explorer',
    'reinforced_exploration',
    'episode_manager',
    'cmd_vel_relay',
    'map_merger',
    'static_transform_publisher',
    'robot_state_publisher',
    'classic_explorer',
    'rl_explorer',
]


def _kill_stale_ros_processes() -> None:
    for pat in _STALE_PATTERNS:
        subprocess.run(
            ['pkill', '-9', '-f', pat],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    mode: str,
    n_episodes: int,
    threshold: float,
    headless: bool,
    world_file: Path,
    result_file: Path,
    randomize_worlds: bool = False,
    save_worlds: bool = False,
    world_size: float | None = None,
    rviz: bool = True,
    sim_speed: float = 1.0,
    rl_eval: bool = False,
    seed: int | None = None,
) -> None:
    print(f'[runner] Starting episodic {"EVALUATION (frozen RL)" if rl_eval else "training"}: '
          f'mode={mode}, episodes={n_episodes}')
    print(f'         threshold={threshold:.0%}, timeout=auto (world size), '
          f'headless={headless}, sim speed={sim_speed:g}x'
          + (' (unlimited)' if sim_speed == 0 else ''))
    print(f'         world    → {world_file}' + (' (new each episode)' if randomize_worlds else ''))
    print(f'         world size: {read_world_size(world_file) or "unknown"} m')
    print(f'         results  → {result_file}')
    print(f'         filtered nodes: {len(LOG_FILTER)} suppressed')

    stats: list[dict] = []

    for episode in range(1, n_episodes + 1):
        print(f'\n[runner] ═══ Episode {episode}/{n_episodes} ═══')

        if randomize_worlds and episode > 1:
            from tb3_slam.utils import DEFAULT_WORLD_SIZE, generate_random_world
            ep_id = uuid.uuid4().hex[:8]
            if save_worlds:
                WORLDS_DIR.mkdir(parents=True, exist_ok=True)
                world_file = WORLDS_DIR / f'{ep_id}.sdf'
            else:
                world_file = Path(f'/tmp/slam_world_{ep_id}.sdf')
            seed_world_rng(seed, episode)
            success, _ = generate_random_world(str(world_file), world_size or DEFAULT_WORLD_SIZE)
            if not success:
                print('[runner] ERROR: world generation failed', file=sys.stderr)
                continue
            print(f'[runner] New world: {world_file}')

        EPISODE_FILE.unlink(missing_ok=True)
        SIM_ELAPSED_FILE.unlink(missing_ok=True)

        # Pass world file to launch via environment variable
        env = os.environ.copy()
        env['SLAM_WORLD_FILE'] = str(world_file)
        env['SLAM_SIM_SPEED'] = f'{sim_speed:g}'

        launch_cmd = build_launch_cmd(mode, EPISODE_FILE, threshold, headless, rviz,
                                      rl_eval)
        print(f'[runner] Launching: {" ".join(launch_cmd)}')

        ep_start = time.time()
        proc = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        start_filtered_readers(proc)

        reason = wait_for_episode_done(EPISODE_FILE, proc)
        wall_s = time.time() - ep_start
        # Durations are reported in sim seconds so results do not depend on the
        # achieved real-time factor; fall back to wall time if unavailable.
        sim_s = read_sim_elapsed()
        if sim_s is None:
            print('[runner] WARNING: no sim-time elapsed available, using wall time',
                  file=sys.stderr)
        elapsed = sim_s if sim_s is not None else wall_s

        print(f'[runner] Episode {episode} finished – reason="{reason}", '
              f'elapsed={elapsed:.1f}s sim, {wall_s:.1f}s wall')
        stats.append({'episode': episode, 'reason': reason, 'elapsed': elapsed})

        kill_launch(proc)
        if reason.startswith('exhausted'):
            kept = keep_world_for_rerun(world_file)
            if kept is not None:
                print(f'[runner] Exhausted – world kept for re-run: {kept}')
        append_result(result_file, episode, mode, reason, elapsed, world_file,
                      wall_s=wall_s, sim_speed=sim_speed)
        print(f'[runner] Result appended to {result_file}')
        print(f'[runner] Simulation stopped. Waiting {INTER_EPISODE_PAUSE}s …')
        time.sleep(INTER_EPISODE_PAUSE)

    # Summary
    print('\n[runner] ══════════════════════════ Training complete ══════════════════════════')
    print(f'  Episodes  : {len(stats)}')
    avg_time = sum(s['elapsed'] for s in stats) / len(stats)
    print(f'  Avg time  : {avg_time:.1f}s (sim time)')
    timeouts = sum(1 for s in stats if 'timeout' in s['reason'])
    print(f'  Timeouts  : {timeouts}/{len(stats)}')
    stuck = sum(1 for s in stats if 'stuck' in s['reason'])
    print(f'  Stuck     : {stuck}/{len(stats)}')
    exhausted = sum(1 for s in stats if s['reason'].startswith('exhausted'))
    print(f'  Exhausted : {exhausted}/{len(stats)}')
    coverages = [_parse_coverage(s['reason']) for s in stats if _parse_coverage(s['reason']) is not None]
    if coverages:
        print(f'  Avg coverage: {sum(coverages)/len(coverages):.1%}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global WORLDS_DIR
    parser = argparse.ArgumentParser(description='Episodic SLAM training runner')
    parser.add_argument('--mode', choices=['classic', 'rl'], default='classic',
                        help='Exploration strategy (default: classic)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.97,
                        help='Coverage threshold to end episode (default: 0.97)')
    parser.add_argument('--headless', action='store_true',
                        help='Skip Gazebo GUI client')
    parser.add_argument('--no-rviz', action='store_true',
                        help='Do not open RViz with the Gazebo GUI (ignored with --headless)')
    parser.add_argument('--world', type=str, default=None,
                        help='Path to an existing .sdf world file (skip generation)')
    parser.add_argument('--world-size', type=float, default=None,
                        help='Edge length in metres of generated square worlds; '
                             'one of tb3_slam.utils.WORLD_SIZES (8, 12; default 8)')
    parser.add_argument('--randomize-worlds', action='store_true',
                        help='Generate a new random world for each episode')
    parser.add_argument('--save-worlds', action='store_true',
                        help='Persist generated worlds to worlds/ (default: /tmp/)')
    parser.add_argument('--worlds-dir', type=Path, default=None,
                        help='Directory for saved worlds (default: worlds/); use e.g. '
                             'worlds/train_8, worlds/eval_8, worlds/eval_12 to keep sets apart')
    parser.add_argument('--sim-speed', type=float, default=1.0,
                        help='Target Gazebo real-time factor: 1 = real time (default), '
                             '4 = up to 4x faster, 0 = unlimited. The achieved speed is '
                             'CPU-bound; measure it with `gz topic -e -t /stats`.')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the trained RL model frozen (greedy, no learning, '
                             'nothing saved); needs --mode rl and a model in rl_data/')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed world generation: the same seed gives the same world '
                             'sequence for every method (use with --randomize-worlds)')
    args = parser.parse_args()
    if args.eval and args.mode != 'rl':
        parser.error('--eval needs --mode rl')
    if args.sim_speed < 0:
        parser.error('--sim-speed must be >= 0')
    if args.world and args.world_size is not None and not args.randomize_worlds:
        parser.error('--world-size has no effect with --world (size is baked into the SDF)')
    if args.world_size is not None:
        # Import here so --help still works without a colcon workspace.
        from tb3_slam.utils import check_world_size
        try:
            check_world_size(args.world_size)
        except ValueError as exc:
            parser.error(str(exc))

    if args.worlds_dir:
        WORLDS_DIR = args.worlds_dir.resolve()
    run_id = uuid.uuid4().hex[:8]
    world_file = prepare_world(args.world, run_id, save=args.save_worlds,
                               world_size=args.world_size, seed=args.seed)
    result_file = make_result_path('rl-eval' if args.eval else args.mode, run_id)

    try:
        run(
            mode=args.mode,
            n_episodes=args.episodes,
            threshold=args.threshold,
            headless=args.headless,
            world_file=world_file,
            result_file=result_file,
            randomize_worlds=args.randomize_worlds,
            save_worlds=args.save_worlds,
            world_size=args.world_size,
            rviz=not args.no_rviz,
            sim_speed=args.sim_speed,
            rl_eval=args.eval,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user.')
    except Exception as exc:
        print(f'[runner] Unexpected error: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
