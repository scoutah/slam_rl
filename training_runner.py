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
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

EPISODE_FILE = Path('/tmp/slam_episode_done')
SHUTDOWN_GRACE = 15      # seconds to wait for ros2 launch to exit cleanly
INTER_EPISODE_PAUSE = 3  # seconds to wait after shutdown before next launch


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
        # Suppress the Gazebo GUI client
        cmd.append('gz_headless:=true')
    return cmd


def wait_for_episode_done(episode_file: Path, proc: subprocess.Popen, poll_interval: float = 1.0) -> str:
    """Block until the episode_done file appears or the subprocess exits."""
    while True:
        if episode_file.exists():
            return episode_file.read_text().strip()
        if proc.poll() is not None:
            return 'process_exited'
        time.sleep(poll_interval)


def kill_launch(proc: subprocess.Popen) -> None:
    """Send SIGINT (graceful) then SIGKILL if needed."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=SHUTDOWN_GRACE)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(mode: str, n_episodes: int, timeout: float, threshold: float, headless: bool) -> None:
    print(f'[runner] Starting episodic training: mode={mode}, episodes={n_episodes}')
    print(f'         threshold={threshold:.0%}, timeout={timeout:.0f}s, headless={headless}')

    stats: list[dict] = []

    for episode in range(1, n_episodes + 1):
        print(f'\n[runner] ═══ Episode {episode}/{n_episodes} ═══')

        # Clean up any leftover signal file from the previous episode
        EPISODE_FILE.unlink(missing_ok=True)

        launch_cmd = build_launch_cmd(mode, EPISODE_FILE, threshold, timeout, headless)
        print(f'[runner] Launching: {" ".join(launch_cmd)}')

        ep_start = time.time()
        proc = subprocess.Popen(
            launch_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        reason = wait_for_episode_done(EPISODE_FILE, proc)
        elapsed = time.time() - ep_start

        print(f'[runner] Episode {episode} finished – reason="{reason}", elapsed={elapsed:.1f}s')
        stats.append({'episode': episode, 'reason': reason, 'elapsed': elapsed})

        kill_launch(proc)
        print(f'[runner] Simulation stopped. Waiting {INTER_EPISODE_PAUSE}s …')
        time.sleep(INTER_EPISODE_PAUSE)

    # Summary
    print('\n[runner] ══════════════════════════ Training complete ══════════════════════════')
    print(f'  Episodes  : {len(stats)}')
    avg_time = sum(s['elapsed'] for s in stats) / len(stats)
    print(f'  Avg time  : {avg_time:.1f}s')
    timeouts = sum(1 for s in stats if 'timeout' in s['reason'])
    print(f'  Timeouts  : {timeouts}/{len(stats)}')


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
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Coverage threshold to end episode (default: 0.95)')
    parser.add_argument('--headless', action='store_true',
                        help='Skip Gazebo GUI client')
    args = parser.parse_args()

    try:
        run(
            mode=args.mode,
            n_episodes=args.episodes,
            timeout=args.timeout,
            threshold=args.threshold,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user.')


if __name__ == '__main__':
    main()
