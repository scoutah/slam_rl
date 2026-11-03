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
SHUTDOWN_GRACE = 15      # seconds to wait for ros2 launch to exit cleanly
INTER_EPISODE_PAUSE = 3  # seconds to wait after shutdown before next launch

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / 'results'
WORLDS_DIR = PROJECT_ROOT / 'worlds'

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
) -> Path:
    """
    Return the path to the .sdf world file to use for this run.

    If *world_arg* is given, use that file directly (no copy).
    Otherwise generate a new random world via tb3_slam.utils, with edge
    length *world_size* (generator default if None).
    When *save* is True, persist to worlds/<run_id>.sdf; otherwise
    write to /tmp/ (ephemeral).
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
    success, _ = generate_random_world(str(world_path), world_size or DEFAULT_WORLD_SIZE)
    if not success:
        print('[runner] ERROR: world generation failed', file=sys.stderr)
        sys.exit(1)
    print(f'[runner] World saved to {world_path}')
    return world_path


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


def make_result_path(mode: str, run_id: str) -> Path:
    """Build the per-run CSV path: results/<dd-mm-yyyy>-<mode>-<id>.csv"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%d-%m-%Y')
    return RESULTS_DIR / f'{date_str}-{mode}-{run_id}.csv'


def append_result(
    result_file: Path,
    episode: int,
    mode: str,
    reason: str,
    elapsed: float,
    world_file: Path | None = None,
) -> None:
    """Append one episode result row to the CSV result file."""
    try:
        coverage = _parse_coverage(reason)
        world_size = read_world_size(world_file) if world_file is not None else None
        write_header = not result_file.exists()

        with result_file.open('a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            if write_header:
                writer.writerow(['timestamp', 'episode', 'mode', 'reason',
                                 'elapsed_s', 'coverage', 'world', 'world_size'])
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                episode,
                mode,
                reason,
                f'{elapsed:.1f}',
                f'{coverage:.4f}' if coverage is not None else '',
                world_file.name if world_file is not None else '',
                world_size if world_size is not None else '',
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
) -> list[str]:
    # The episode timeout is derived from the world size by episode_manager
    # (issue #10) and is deliberately not configurable here.
    cmd = [
        'ros2', 'launch', 'tb3_slam', 'sim_slam.launch.py',
        f'exploration_mode:={mode}',
        f'episode_file:={episode_file}',
        f'coverage_threshold:={threshold}',
    ]
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
) -> None:
    print(f'[runner] Starting episodic training: mode={mode}, episodes={n_episodes}')
    print(f'         threshold={threshold:.0%}, timeout=auto (world size), '
          f'headless={headless}')
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
            success, _ = generate_random_world(str(world_file), world_size or DEFAULT_WORLD_SIZE)
            if not success:
                print('[runner] ERROR: world generation failed', file=sys.stderr)
                continue
            print(f'[runner] New world: {world_file}')

        EPISODE_FILE.unlink(missing_ok=True)

        # Pass world file to launch via environment variable
        env = os.environ.copy()
        env['SLAM_WORLD_FILE'] = str(world_file)

        launch_cmd = build_launch_cmd(mode, EPISODE_FILE, threshold, headless, rviz)
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
        elapsed = time.time() - ep_start

        print(f'[runner] Episode {episode} finished – reason="{reason}", elapsed={elapsed:.1f}s')
        stats.append({'episode': episode, 'reason': reason, 'elapsed': elapsed})

        kill_launch(proc)
        append_result(result_file, episode, mode, reason, elapsed, world_file)
        print(f'[runner] Result appended to {result_file}')
        print(f'[runner] Simulation stopped. Waiting {INTER_EPISODE_PAUSE}s …')
        time.sleep(INTER_EPISODE_PAUSE)

    # Summary
    print('\n[runner] ══════════════════════════ Training complete ══════════════════════════')
    print(f'  Episodes  : {len(stats)}')
    avg_time = sum(s['elapsed'] for s in stats) / len(stats)
    print(f'  Avg time  : {avg_time:.1f}s')
    timeouts = sum(1 for s in stats if 'timeout' in s['reason'])
    print(f'  Timeouts  : {timeouts}/{len(stats)}')
    stuck = sum(1 for s in stats if 'stuck' in s['reason'])
    print(f'  Stuck     : {stuck}/{len(stats)}')
    coverages = [_parse_coverage(s['reason']) for s in stats if _parse_coverage(s['reason']) is not None]
    if coverages:
        print(f'  Avg coverage: {sum(coverages)/len(coverages):.1%}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Episodic SLAM training runner')
    parser.add_argument('--mode', choices=['classic', 'rl'], default='classic',
                        help='Exploration strategy (default: classic)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='Coverage threshold to end episode (default: 0.99)')
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
    args = parser.parse_args()
    if args.world and args.world_size is not None and not args.randomize_worlds:
        parser.error('--world-size has no effect with --world (size is baked into the SDF)')
    if args.world_size is not None:
        # Import here so --help still works without a colcon workspace.
        from tb3_slam.utils import check_world_size
        try:
            check_world_size(args.world_size)
        except ValueError as exc:
            parser.error(str(exc))

    run_id = uuid.uuid4().hex[:8]
    world_file = prepare_world(args.world, run_id, save=args.save_worlds,
                               world_size=args.world_size)
    result_file = make_result_path(args.mode, run_id)

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
        )
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user.')
    except Exception as exc:
        print(f'[runner] Unexpected error: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
