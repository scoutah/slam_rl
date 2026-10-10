#!/usr/bin/env bash
# Training entrypoint for the SLAM-RL container.
#
# Sources ROS + the workspace overlay (like entrypoint.sh), then runs
# training_runner.py, forwarding ALL arguments to it. This lets
#   docker compose run --rm train --episodes 1 --headless
# work as expected: Compose replaces the service command with those flags, and
# this wrapper passes them straight to the runner instead of trying to exec
# them as a program.
set -e

WORKSPACE="${WORKSPACE:-/root/projects/slam_rl}"

source /opt/ros/jazzy/setup.bash
if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source "${WORKSPACE}/install/setup.bash"
fi

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export SLAM_RL_DATA_DIR="${SLAM_RL_DATA_DIR:-${WORKSPACE}/rl_data}"
mkdir -p "${SLAM_RL_DATA_DIR}"

cd "${WORKSPACE}"
exec python3 training_runner.py "$@"
