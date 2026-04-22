#!/usr/bin/env python3
"""
Episodic SLAM training runner.

Launches the simulation once per episode, waits until the episode_manager
signals completion (95 % map coverage or timeout), then kills everything
and starts a fresh episode with a newly generated world.

Usage:
    python3 training_runner.py [--mode classic|rl] [--episodes N]
                               [--timeout SECONDS] [--threshold FLOAT]
                               [--headless]

Examples:
    python3 training_runner.py --mode classic --episodes 20
    python3 training_runner.py --mode rl --episodes 100 --headless
"""

import argparse
import csv
import io
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

EPISODE_FILE = Path('/tmp/slam_episode_done')
SHUTDOWN_GRACE = 15      # seconds to wait for ros2 launch to exit cleanly
INTER_EPISODE_PAUSE = 3  # seconds to wait after shutdown before next launch
DEFAULT_RESULT_FILE = Path(__file__).resolve().parent / 'results' / 'training_results.csv'

# ROS node name substrings whose log lines are suppressed in the terminal
LOG_FILTER: list[str] = ['cartographer_node-8', 'controller_server-12']


# ---------------------------------------------------------------------------
# Log filtering
# ---------------------------------------------------------------------------

def _stream_filter(src: io.RawIOBase, dst: io.TextIOWrapper) -> None:
    """Read *src* line by line, writing to *dst* unless the line matches LOG_FILTER."""
    for raw in iter(src.readline, b''):
        line = raw.decode('utf-8', errors='replace')
        if not any(f in line for f in LOG_FILTER):
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
# Result logging
# ---------------------------------------------------------------------------

def _parse_coverage(reason: str) -> float | None:
    """Extract the coverage float from a reason string like 'coverage=95.0%'."""
    m = re.search(r'coverage=([0-9.]+)%', reason)
    return float(m.group(1)) / 100.0 if m else None


def append_result(
    result_file: Path,
    episode: int,
    mode: str,
    reason: str,
    elapsed: float,
) -> None:
    """Append one episode result row to the CSV result file."""
    try:
        result_file.parent.mkdir(parents=True, exist_ok=True)
        coverage = _parse_coverage(reason)
        write_header = not result_file.exists()

        with result_file.open('a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'episode', 'mode', 'reason', 'elapsed_s', 'coverage'])
            writer.writerow([
                datetime.now().isoformat(timespec='seconds'),
                episode,
                mode,
                reason,
                f'{elapsed:.1f}',
                f'{coverage:.4f}' if coverage is not None else '',
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
    timeout: float,
    headless: bool,
) -> list[str]:
    cmd = [
        'ros2', 'launch', 'tb3_slam', 'sim_slam.launch.py',
        f'exploration_mode:={mode}',
        f'episode_file:={episode_file}',
        f'coverage_threshold:={threshold}',
        f'max_episode_duration:={timeout}',
    ]
    if headless:
        cmd.append('headless:=true')
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
            # Process died – check file one final time in case it was written
            # just before the process exited (e.g. episode_manager signal flow).
            time.sleep(0.2)
            if episode_file.exists():
                return episode_file.read_text().strip()
            return 'process_exited'
        time.sleep(poll_interval)


def kill_launch(proc: subprocess.Popen) -> None:
    """Send SIGINT to the launch process group (graceful), then SIGKILL if needed."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=SHUTDOWN_GRACE)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    mode: str,
    n_episodes: int,
    timeout: float,
    threshold: float,
    headless: bool,
    result_file: Path,
) -> None:
    print(f'[runner] Starting episodic training: mode={mode}, episodes={n_episodes}')
    print(f'         threshold={threshold:.0%}, timeout={timeout:.0f}s, headless={headless}')
    print(f'         results → {result_file}')
    print(f'         filtered nodes: {LOG_FILTER}')

    stats: list[dict] = []

    for episode in range(1, n_episodes + 1):
        print(f'\n[runner] ═══ Episode {episode}/{n_episodes} ═══')

        EPISODE_FILE.unlink(missing_ok=True)

        launch_cmd = build_launch_cmd(mode, EPISODE_FILE, threshold, timeout, headless)
        print(f'[runner] Launching: {" ".join(launch_cmd)}')

        ep_start = time.time()
        proc = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # isolates the launch group so SIGINT from
        )                            # episode_manager doesn't reach this runner
        start_filtered_readers(proc)

        reason = wait_for_episode_done(EPISODE_FILE, proc)
        elapsed = time.time() - ep_start

        print(f'[runner] Episode {episode} finished – reason="{reason}", elapsed={elapsed:.1f}s')
        stats.append({'episode': episode, 'reason': reason, 'elapsed': elapsed})

        kill_launch(proc)
        append_result(result_file, episode, mode, reason, elapsed)
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
    parser.add_argument('--timeout', type=float, default=600.0,
                        help='Max episode duration in seconds (default: 600)')
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='Coverage threshold to end episode (default: 0.99)')
    parser.add_argument('--headless', action='store_true',
                        help='Skip Gazebo GUI client')
    parser.add_argument('--result-file', type=Path, default=DEFAULT_RESULT_FILE,
                        help=f'CSV file to append results to (default: {DEFAULT_RESULT_FILE})')
    args = parser.parse_args()

    try:
        run(
            mode=args.mode,
            n_episodes=args.episodes,
            timeout=args.timeout,
            threshold=args.threshold,
            headless=args.headless,
            result_file=args.result_file,
        )
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user.')
    except Exception as exc:
        print(f'[runner] Unexpected error: {exc}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
